import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import accelerate
import torch
import yaml
from einops import rearrange
from torch.distributed.elastic.multiprocessing.errors import record

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, launch_training_task
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_dit_stage2_v6_1 import (
    enable_stage2_causal_attention,
    set_stage2_grid,
    stage2_streaming_block_forward,
)
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_ShapeChecker,
)
from wanvideo.data.flashvsr.datasets.streaming_dataset import VIDEO_EXTENSIONS, FlashVSRStreamingDataset
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53 import FlashVSRTarStreamingDatasetV53
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5


class FlashVSRStage2VideoOnlyDataset(FlashVSRStreamingDataset):
    """Video-only tar/manifest dataset for Stage2.

    It reuses the v5.3 video discovery/iteration path, but never constructs an
    image branch. This keeps Stage2 aligned with the paper: video data only.
    """

    def __init__(
        self,
        *,
        yubari_video_tar_url: Optional[str],
        takano_video_tar_url: Optional[str],
        internal_url: Optional[str],
        yubari_video_prob: Optional[float],
        takano_video_prob: Optional[float],
        height: int,
        width: int,
        num_frames: int,
        stride: int = 1,
        max_source_frames: int = 160,
        enable_degradation: bool = True,
        degradation_config_path: Optional[str] = None,
        degradation_seed: Optional[int] = None,
        hq_prefix_frames: int = 0,
        control_dropout_prob: float = 0.0,
        shuffle_buffer: int = 100,
        global_seed: Optional[int] = None,
        output_tensors: bool = True,
    ):
        if internal_url and (yubari_video_tar_url or takano_video_tar_url):
            raise ValueError("Set either internal_url or yubari/takano video URLs for Stage2, not both.")
        self._stage2_internal_only = bool(internal_url)
        # The base class requires at least one source during construction. Use
        # one lightweight source here, then replace iteration with the explicit
        # Stage2 video-source logic below. This avoids double-scanning every root.
        bootstrap_url = internal_url or takano_video_tar_url or yubari_video_tar_url
        super().__init__(
            internal_url=bootstrap_url,
            image_internal_url=None,
            image_dataset_prob=0.0,
            metadata_url=None,
            metadata_source="auto",
            max_parquet_records=None,
            min_overall_score=None,
            require_qwen35_parse_success=False,
            height=height,
            width=width,
            num_frames=num_frames,
            stride=stride,
            max_source_frames=max_source_frames,
            enable_degradation=enable_degradation,
            degradation_config_path=degradation_config_path,
            degradation_seed=degradation_seed,
            hq_prefix_frames=hq_prefix_frames,
            control_dropout_prob=control_dropout_prob,
            shuffle_buffer=shuffle_buffer,
            global_seed=global_seed,
            output_tensors=output_tensors,
        )
        if self._stage2_internal_only:
            self.yubari_video_prob, self.takano_video_prob = 0.0, 0.0
        else:
            self.yubari_video_prob, self.takano_video_prob = FlashVSRTarStreamingDatasetV53._resolve_video_probs(
                yubari_video_tar_url=yubari_video_tar_url,
                takano_video_tar_url=takano_video_tar_url,
                yubari_video_prob=yubari_video_prob,
                takano_video_prob=takano_video_prob,
            )
        (
            self.yubari_video_manifest_urls,
            self.yubari_video_urls,
            self.yubari_video_tar_urls,
            self.yubari_video_file_urls,
        ) = self._discover_video_source(yubari_video_tar_url)
        (
            self.takano_video_manifest_urls,
            self.takano_video_urls,
            self.takano_video_tar_urls,
            self.takano_video_file_urls,
        ) = self._discover_video_source(takano_video_tar_url)
        self.custom_collate_fn = self.tensor_collate_fn

    def _discover_video_source(self, base_url: Optional[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        if not base_url:
            return [], [], [], []
        manifest_urls, urls = self._discover_sample_sources(base_url, VIDEO_EXTENSIONS + (".tar",))
        if manifest_urls and not urls:
            expanded_urls: List[str] = []
            for manifest_path in manifest_urls:
                expanded_urls.extend(FlashVSRTarStreamingDatasetV53._load_manifest_entries_once(manifest_path))
            urls = expanded_urls
            manifest_urls = []
        tar_urls = [url for url in urls if str(url).endswith(".tar")]
        file_urls = [url for url in urls if not str(url).endswith(".tar")]
        return manifest_urls, urls, tar_urls, file_urls

    def _iterate_video_source(
        self,
        source_dataset: str,
        tar_urls: Sequence[str],
        file_urls: Sequence[str],
        manifest_urls: Sequence[str],
        rng,
    ) -> Iterator[Dict[str, Any]]:
        iterators: List[Iterator[Dict[str, Any]]] = []
        if tar_urls:
            datapipe = self._make_torchdata_tar_pipe(list(tar_urls), rng=rng if self.global_seed is not None else None)

            def tar_iter() -> Iterator[Dict[str, Any]]:
                while True:
                    for file_name, stream_item in datapipe:
                        if not str(file_name).endswith(VIDEO_EXTENSIONS):
                            continue
                        sample = self._process_video_bytes(
                            stream_item.read(),
                            sample_id=os.path.basename(str(file_name)),
                            rng=rng,
                        )
                        if sample is not None:
                            sample["source_dataset"] = source_dataset
                            yield sample

            iterators.append(tar_iter())
        if file_urls or manifest_urls:
            urls = self._split_for_process_and_worker(list(file_urls))

            def direct_iter() -> Iterator[Dict[str, Any]]:
                for url in self._iter_deterministic_permutation(urls, rng) if urls else []:
                    sample = self._process_video_bytes(self._open_binary(url), sample_id=os.path.basename(url), rng=rng)
                    if sample is not None:
                        sample["source_dataset"] = source_dataset
                        yield sample
                if manifest_urls:
                    for url in self._iter_manifest_entries(list(manifest_urls)):
                        sample = self._process_video_bytes(self._open_binary(url), sample_id=os.path.basename(url), rng=rng)
                        if sample is not None:
                            sample["source_dataset"] = source_dataset
                            yield sample

            iterators.append(direct_iter())
        if not iterators:
            return
        if len(iterators) == 1:
            while True:
                yield next(iterators[0])
        while True:
            yield next(iterators[rng.randrange(len(iterators))])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = self._make_iteration_rng()
        iterators: List[Tuple[float, Iterator[Dict[str, Any]]]] = []
        if self._stage2_internal_only:
            yield from super().__iter__()
            return
        if self.yubari_video_prob > 0 and (self.yubari_video_urls or self.yubari_video_manifest_urls):
            iterators.append(
                (
                    self.yubari_video_prob,
                    self._iterate_video_source(
                        "yubari",
                        self.yubari_video_tar_urls,
                        self.yubari_video_file_urls,
                        self.yubari_video_manifest_urls,
                        rng,
                    ),
                )
            )
        if self.takano_video_prob > 0 and (self.takano_video_urls or self.takano_video_manifest_urls):
            iterators.append(
                (
                    self.takano_video_prob,
                    self._iterate_video_source(
                        "takano",
                        self.takano_video_tar_urls,
                        self.takano_video_file_urls,
                        self.takano_video_manifest_urls,
                        rng,
                    ),
                )
            )
        if not iterators:
            raise ValueError("Stage2 dataset discovered no video samples.")
        if len(iterators) == 1:
            yield from iterators[0][1]
            return
        weights = [item[0] for item in iterators]
        sources = [item[1] for item in iterators]
        while True:
            yield next(rng.choices(sources, weights=weights, k=1)[0])

    def validation_video_iterator(self, rng: Optional[Any] = None) -> Iterator[Dict[str, Any]]:
        rng = rng or self._make_iteration_rng()
        yield from self.__iter__()


class WanVideoUnit_NoiseInitializerStage2(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = max(1, int(num_frames) - 1) // 4
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnit_InputVideoEmbedderStage2(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "framewise_decoding"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, noise, tiled, tile_size, tile_stride, framewise_decoding):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        if framewise_decoding:
            input_latents = pipe.vae.encode_framewise(input_video, device=pipe.device)
        else:
            input_latents = pipe.vae.encode(
                input_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if input_latents.shape[2] < 2:
            raise ValueError(f"Stage2 needs at least two GT latent frames, got {tuple(input_latents.shape)}")
        input_latents = input_latents[:, :, 1:].contiguous()
        if noise.shape != input_latents.shape:
            raise ValueError(f"Stage2 noise/target mismatch: noise={tuple(noise.shape)} target={tuple(input_latents.shape)}")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents}


def FlowMatchSFTLossStage2(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    return loss * pipe.scheduler.training_weight(timestep)


class FlashVSRStage2Pipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        prompt_tensor_path=None,
        lq_proj_checkpoint=None,
        lq_proj_layer_num=None,
        zero_init_lq_proj_in=True,
        stage2_attention_mode: str = "block_sparse_chunk_causal",
        stage2_topk_ratio: float = 2.0,
        stage2_local_num: int = -1,
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs or [],
            tokenizer_config=None,
        )
        pipe.__class__ = FlashVSRStage2Pipeline
        pipe.prompt_tensor_path = prompt_tensor_path
        pipe.fixed_prompt_tensor = None
        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializerStage2(),
            v5.FlashVSRUnit_FixedPrompt(),
            WanVideoUnit_InputVideoEmbedderStage2(),
            v5.FlashVSRUnit_LQVideoEmbedder(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.model_fn = flashvsr_stage2_model_fn
        pipe.compilable_models = ["dit"]
        pipe.lq_proj_scale = 1.0
        pipe.debug_tensor_dump_dir = None
        enable_stage2_causal_attention(
            pipe.dit,
            mode=stage2_attention_mode,
            topk_ratio=stage2_topk_ratio,
            local_num=None if int(stage2_local_num) < 0 else int(stage2_local_num),
        )

        effective_lq_proj_layers = 1 if lq_proj_layer_num is None else int(lq_proj_layer_num)
        pipe.lq_proj_in = v5.FlashVSRLQProjIn(
            in_dim=3,
            out_dim=pipe.dit.dim,
            layer_num=effective_lq_proj_layers,
            zero_init_output=zero_init_lq_proj_in and lq_proj_checkpoint is None,
            temporal_mode="streaming",
        ).to(device=device, dtype=torch_dtype)
        if lq_proj_checkpoint is not None:
            state_dict = torch.load(lq_proj_checkpoint, map_location="cpu")
            pipe.lq_proj_in.load_state_dict(state_dict, strict=True)
        return pipe

    @torch.no_grad()
    def infer_from_lq(
        self,
        lq_video,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        num_inference_steps: int = 10,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        framewise_decoding: bool = False,
        output_type: str = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        inputs_shared = {
            "input_video": None,
            "lq_video": lq_video,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "framewise_decoding": framewise_decoding,
            "lq_proj_scale": self.lq_proj_scale,
        }
        for unit in self.units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, {}, {})
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.model_fn(**models, **inputs_shared, timestep=timestep)
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video

    @torch.no_grad()
    def infer_from_lq_streaming(
        self,
        lq_video,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        tiled: bool = True,
        tile_size: Tuple[int, int] = (30, 52),
        tile_stride: Tuple[int, int] = (15, 26),
        output_type: str = "quantized",
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
    ):
        """Official-style Stage2 streaming/cache inference.

        Stage2 predicts only the post-first-frame latent stream: 89 raw frames
        produce 22 latent-time positions, processed as 6 + 2 + ... chunks.
        """
        if int(num_frames) % 8 != 1:
            raise ValueError(f"Stage2 streaming inference expects num_frames % 8 == 1, got {num_frames}")
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        lq_video = self.preprocess_video(lq_video).to(device=self.device, dtype=self.torch_dtype)
        lq_video = lq_video[:, :, :num_frames]
        latent_frames = max(1, (int(num_frames) - 1) // 4)
        latents = self.generate_noise(
            (
                1,
                self.vae.model.z_dim,
                latent_frames,
                int(height) // self.vae.upsampling_factor,
                int(width) // self.vae.upsampling_factor,
            ),
            seed=seed,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)

        context = v5.FlashVSRUnit_FixedPrompt().process(self)["context"]
        self.load_models_to_device(("dit", "lq_proj_in"))
        models = {"dit": self.dit}
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep_tensor = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            self.lq_proj_in.clear_cache()
            pre_cache_k: List[Optional[torch.Tensor]] = [None] * len(self.dit.blocks)
            pre_cache_v: List[Optional[torch.Tensor]] = [None] * len(self.dit.blocks)
            updated_chunks: List[torch.Tensor] = []
            process_total_num = (int(num_frames) - 1) // 8 - 2
            for cur_process_idx in range(process_total_num):
                if cur_process_idx == 0:
                    lq_latents = None
                    for inner_idx in range(7):
                        start = max(0, inner_idx * 4 - 3)
                        end = (inner_idx + 1) * 4 - 3
                        cur = self.lq_proj_in.stream_forward(lq_video[:, :, start:end])
                        if cur is None:
                            continue
                        if lq_latents is None:
                            lq_latents = cur
                        else:
                            for layer_idx in range(len(lq_latents)):
                                lq_latents[layer_idx] = torch.cat([lq_latents[layer_idx], cur[layer_idx]], dim=1)
                    cur_latents = latents[:, :, :6]
                else:
                    lq_latents = None
                    for inner_idx in range(2):
                        start = cur_process_idx * 8 + 17 + inner_idx * 4
                        end = cur_process_idx * 8 + 21 + inner_idx * 4
                        cur = self.lq_proj_in.stream_forward(lq_video[:, :, start:end])
                        if cur is None:
                            continue
                        if lq_latents is None:
                            lq_latents = cur
                        else:
                            for layer_idx in range(len(lq_latents)):
                                lq_latents[layer_idx] = torch.cat([lq_latents[layer_idx], cur[layer_idx]], dim=1)
                    cur_latents = latents[:, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2]

                noise_pred, pre_cache_k, pre_cache_v = flashvsr_stage2_streaming_model_fn(
                    **models,
                    latents=cur_latents,
                    timestep=timestep_tensor,
                    context=context,
                    lq_latents=lq_latents,
                    lq_proj_scale=self.lq_proj_scale,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    cur_process_idx=cur_process_idx,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                )
                updated_chunks.append(
                    self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], cur_latents)
                )
            latents = torch.cat(updated_chunks, dim=2)

        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


def flashvsr_stage2_model_fn(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    lq_latents=None,
    lq_proj_scale: float = 1.0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    batch_size = latents.shape[0]
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.shape[0] == 1 and batch_size > 1:
        context = context.expand(batch_size, -1, -1)
    elif context.shape[0] != batch_size:
        raise ValueError(f"Context batch mismatch: context={tuple(context.shape)} latents={tuple(latents.shape)}")
    context = dit.text_embedding(context)

    x = dit.patch_embedding(latents)
    f, h, w = x.shape[2:]
    if f % 2 != 0:
        raise ValueError(f"Stage2 DiT latent-time must be even, got f={f} from latents={tuple(latents.shape)}")
    x = rearrange(x, "b c f h w -> b (f h w) c")
    set_stage2_grid(dit, (int(f), int(h), int(w)))
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if lq_latents is not None:
        expected_tokens = x.shape[1]
        for layer_idx, layer_latents in enumerate(lq_latents):
            if layer_latents.shape[1] != expected_tokens:
                raise ValueError(
                    f"Stage2 requires exact LQ/DiT token match at layer {layer_idx}: "
                    f"x={expected_tokens}, lq={layer_latents.shape[1]}, grid={(f, h, w)}"
                )

    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + lq_latents[block_id] * lq_proj_scale
        if dit.training:
            def block_forward(hidden_states, block=block, context=context, t_mod=t_mod, freqs=freqs):
                return block(hidden_states, context, t_mod, freqs)

            x = gradient_checkpoint_forward(
                block_forward,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


def flashvsr_stage2_streaming_model_fn(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    lq_latents=None,
    lq_proj_scale: float = 1.0,
    pre_cache_k: Optional[List[Optional[torch.Tensor]]] = None,
    pre_cache_v: Optional[List[Optional[torch.Tensor]]] = None,
    cur_process_idx: int = 0,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    **kwargs,
):
    batch_size = latents.shape[0]
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.shape[0] == 1 and batch_size > 1:
        context = context.expand(batch_size, -1, -1)
    elif context.shape[0] != batch_size:
        raise ValueError(f"Context batch mismatch: context={tuple(context.shape)} latents={tuple(latents.shape)}")
    context = dit.text_embedding(context)

    x = dit.patch_embedding(latents)
    f, h, w = x.shape[2:]
    if f not in (2, 6):
        raise ValueError(f"Stage2 streaming chunks must be 6 or 2 latent frames, got f={f}")
    x = rearrange(x, "b c f h w -> b (f h w) c")
    time_start = 0 if int(cur_process_idx) == 0 else 4 + int(cur_process_idx) * 2
    freqs = torch.cat(
        [
            dit.freqs[0][time_start : time_start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if pre_cache_k is None:
        pre_cache_k = [None] * len(dit.blocks)
    if pre_cache_v is None:
        pre_cache_v = [None] * len(dit.blocks)
    next_cache_k: List[Optional[torch.Tensor]] = [None] * len(dit.blocks)
    next_cache_v: List[Optional[torch.Tensor]] = [None] * len(dit.blocks)

    expected_tokens = x.shape[1]
    if lq_latents is not None:
        for layer_idx, layer_latents in enumerate(lq_latents):
            if layer_latents.shape[1] != expected_tokens:
                raise ValueError(
                    f"Stage2 streaming requires exact LQ/chunk token match at layer {layer_idx}: "
                    f"x={expected_tokens}, lq={layer_latents.shape[1]}, grid={(f, h, w)}"
                )

    grid = (int(f), int(h), int(w))
    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + lq_latents[block_id] * lq_proj_scale
        x, cache_k, cache_v = stage2_streaming_block_forward(
            block,
            x,
            context,
            t_mod,
            freqs,
            grid=grid,
            pre_cache_k=pre_cache_k[block_id],
            pre_cache_v=pre_cache_v[block_id],
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
        )
        next_cache_k[block_id] = cache_k
        next_cache_v[block_id] = cache_v

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w)), next_cache_k, next_cache_v


class FlashVSRStage2TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        prompt_tensor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=384,
        lora_checkpoint=None,
        lq_proj_checkpoint=None,
        resume_stage1_checkpoint=None,
        lq_proj_layer_num=None,
        lq_proj_scale: float = 1.0,
        zero_init_lq_proj_in=True,
        freeze_lq_proj_in: bool = False,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        stage2_attention_mode: str = "block_sparse_chunk_causal",
        stage2_topk_ratio: float = 2.0,
        stage2_local_num: int = -1,
        fp8_models=None,
        offload_models=None,
        device="cpu",
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            raise ValueError("Stage2 v6 requires gradient checkpointing for practical memory use.")
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        self.pipe = FlashVSRStage2Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            prompt_tensor_path=prompt_tensor_path,
            lq_proj_checkpoint=lq_proj_checkpoint,
            lq_proj_layer_num=lq_proj_layer_num,
            zero_init_lq_proj_in=zero_init_lq_proj_in,
            stage2_attention_mode=stage2_attention_mode,
            stage2_topk_ratio=stage2_topk_ratio,
            stage2_local_num=stage2_local_num,
        )
        self.pipe.lq_proj_scale = float(lq_proj_scale)
        self.pipe = self.split_pipeline_units("sft", self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            task="sft",
        )
        if resume_stage1_checkpoint is not None:
            if lora_base_model is None:
                raise ValueError("resume_stage1_checkpoint requires lora_base_model.")
            self._load_stage1_resume_checkpoint(resume_stage1_checkpoint, lora_base_model)
        if freeze_lq_proj_in:
            for param in self.pipe.lq_proj_in.parameters():
                param.requires_grad = False
            self.pipe.lq_proj_in.eval()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        export_names = self.trainable_param_names()
        export_names.update(name for name, _ in self.named_parameters() if name.startswith("pipe.lq_proj_in."))
        state_dict = {name: param for name, param in state_dict.items() if name in export_names}
        if remove_prefix is not None:
            state_dict = {
                (name[len(remove_prefix):] if name.startswith(remove_prefix) else name): param
                for name, param in state_dict.items()
            }
        return state_dict

    def _load_stage1_resume_checkpoint(self, checkpoint_path: str, lora_base_model: str) -> None:
        state_dict = v5.load_state_dict(checkpoint_path, device="cpu")
        lq_proj_state, lora_state, _ = v5.flashvsr_stage1_split_exported_state(state_dict)
        if not lq_proj_state and not lora_state:
            raise ValueError(f"Stage2 resume checkpoint has no lq_proj_in or LoRA weights: {checkpoint_path}")
        if lq_proj_state:
            result = self.pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
            print(
                f"Stage2 v6 loaded lq_proj_in from {checkpoint_path}, "
                f"keys={len(lq_proj_state)}, missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}",
                flush=True,
            )
        if lora_state:
            lora_model = getattr(self.pipe, lora_base_model)
            mapped_lora_state = self.mapping_lora_state_dict(lora_state)
            result = lora_model.load_state_dict(mapped_lora_state, strict=False)
            print(
                f"Stage2 v6 loaded LoRA from {checkpoint_path}, "
                f"keys={len(mapped_lora_state)}, missing={len(result[0])}, unexpected={len(result[1])}",
                flush=True,
            )

    def get_pipeline_inputs(self, data):
        if not torch.is_tensor(data["video"]) or not torch.is_tensor(data["lq_video"]):
            raise ValueError("Stage2 v6 expects tensor-collated video/lq_video.")
        video = data["video"]
        return (
            {
                "input_video": video,
                "lq_video": data["lq_video"],
                "height": int(video.shape[-2]),
                "width": int(video.shape[-1]),
                "num_frames": int(video.shape[1]),
                "cfg_scale": 1.0,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "framewise_decoding": False,
                "seed": 0,
                "lq_proj_scale": self.pipe.lq_proj_scale,
            },
            {},
            {},
        )

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        merged_inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            merged_inputs = self.pipe.unit_runner(unit, self.pipe, *merged_inputs)
        return FlowMatchSFTLossStage2(self.pipe, **merged_inputs[0], **merged_inputs[1])


class FlashVSRStage2ValidationCallback:
    def __init__(
        self,
        *,
        output_path: str,
        validation_samples: List[Dict[str, Any]],
        num_inference_steps: int,
        fps: int,
        seed_base: int,
        use_wandb: bool = False,
    ):
        self.output_path = output_path
        self.validation_samples = validation_samples
        self.num_inference_steps = int(num_inference_steps)
        self.fps = int(fps)
        self.seed_base = int(seed_base)
        self.use_wandb = bool(use_wandb)

    def __call__(self, accelerator, model, checkpoint_path: str, step: int):
        if not self.validation_samples:
            return
        validation_dir = os.path.join(self.output_path, "validation", f"step-{step}")
        os.makedirs(validation_dir, exist_ok=True)

        pipe = model.pipe
        scheduler_state = {
            "timesteps": pipe.scheduler.timesteps.clone() if getattr(pipe.scheduler, "timesteps", None) is not None else None,
            "training": getattr(pipe.scheduler, "training", None),
        }
        training_mode = model.training
        model.eval()
        try:
            for sample_index, sample in enumerate(self.validation_samples):
                sample_dir = os.path.join(validation_dir, f"sample_{sample_index:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                hr_tensor = sample["video"]
                lq_tensor = sample["lq_video"]
                hr_frames = v5._tensor_video_to_pil_frames(hr_tensor)
                lq_frames = v5._tensor_video_to_pil_frames(lq_tensor)
                v5.save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                v5.save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                sr_frames = pipe.infer_from_lq_streaming(
                    lq_video=lq_tensor.unsqueeze(0),
                    height=int(hr_tensor.shape[-2]),
                    width=int(hr_tensor.shape[-1]),
                    num_frames=int(hr_tensor.shape[0]),
                    seed=self.seed_base + sample_index,
                    rand_device="cpu",
                    num_inference_steps=self.num_inference_steps,
                    tiled=True,
                    output_type="quantized",
                )
                v5.save_video(sr_frames, os.path.join(sample_dir, "sr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "checkpoint_path": checkpoint_path,
                            "step": int(step),
                            "sample_index": int(sample_index),
                            "validation_mode": "stage2_v6_video_only",
                            "input_num_frames": int(hr_tensor.shape[0]),
                            "output_num_frames": len(sr_frames),
                            "sample_seed": v5._serialize_sample_seed(sample.get("sample_seed")),
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
                if self.use_wandb and sample_index == 0:
                    try:
                        import wandb

                        if wandb.run is not None:
                            wandb.log(
                                {
                                    "validation/step": step,
                                    "validation/hr_video": wandb.Video(os.path.join(sample_dir, "hr.mp4"), fps=self.fps, format="mp4"),
                                    "validation/lq_video": wandb.Video(os.path.join(sample_dir, "lq.mp4"), fps=self.fps, format="mp4"),
                                    "validation/sr_video": wandb.Video(os.path.join(sample_dir, "sr.mp4"), fps=self.fps, format="mp4"),
                                },
                                step=step,
                            )
                    except Exception as error:
                        print(f"[wandb] stage2 validation log failed: {error}", flush=True)
        finally:
            model.train(training_mode)
            if scheduler_state["training"]:
                pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
            else:
                if scheduler_state["timesteps"] is not None:
                    pipe.scheduler.timesteps = scheduler_state["timesteps"]
                if scheduler_state["training"] is not None:
                    pipe.scheduler.training = scheduler_state["training"]


def _stage2_parser():
    parser = v5.flashvsr_parser()
    parser.add_argument(
        "--stage2_attention_mode",
        type=str,
        default="block_sparse_chunk_causal",
        choices=("block_sparse_chunk_causal", "block_sparse_official_mask", "dense_full"),
        help="Stage2 self-attention backend. block_sparse_chunk_causal is the author-aligned training path.",
    )
    parser.add_argument("--stage2_topk_ratio", type=float, default=2.0, help="Fixed top-k ratio for Stage2 block sparse attention.")
    parser.add_argument(
        "--stage2_local_num",
        type=int,
        default=-1,
        help="Official FlashVSR local history window in chunk units. -1 follows official random local_num sampling.",
    )
    return parser


def parse_stage2_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = _stage2_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**v5._flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if args.image_tar_url is not None:
        args.picked17k_image_tar_url = args.image_tar_url
    else:
        args.image_tar_url = args.picked17k_image_tar_url
    if args.num_frames % 8 != 1:
        parser.error("Stage2 v6 requires num_frames % 8 == 1 so dropping the first GT latent leaves an even latent-time.")
    return args


@record
def main(argv=None):
    def _excepthook(exc_type, exc_value, exc_traceback):
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[fatal rank={rank} local_rank={local_rank}] {exc_type.__name__}: {exc_value}", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook
    args = parse_stage2_args(argv)
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    }
    data_loader_config_cls = getattr(accelerate, "DataLoaderConfiguration", None)
    if data_loader_config_cls is not None:
        accelerator_kwargs["dataloader_config"] = data_loader_config_cls(
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    v5.configure_deepspeed_runtime(accelerator, args)
    if accelerator.is_main_process:
        v5.dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)

    dataset = FlashVSRStage2VideoOnlyDataset(
        yubari_video_tar_url=args.yubari_video_tar_url,
        takano_video_tar_url=args.takano_video_tar_url,
        internal_url=args.internal_url,
        yubari_video_prob=args.yubari_video_prob,
        takano_video_prob=args.takano_video_prob,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        enable_degradation=args.enable_degradation,
        degradation_config_path=args.degradation_config_path,
        degradation_seed=args.degradation_seed,
        hq_prefix_frames=args.hq_prefix_frames,
        control_dropout_prob=args.control_dropout_prob,
        shuffle_buffer=args.shuffle_buffer,
        global_seed=args.global_seed,
        output_tensors=True,
    )
    model = FlashVSRStage2TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lq_proj_checkpoint=args.lq_proj_checkpoint,
        resume_stage1_checkpoint=args.resume_stage1_checkpoint,
        lq_proj_layer_num=args.lq_proj_layer_num,
        lq_proj_scale=args.lq_proj_scale,
        zero_init_lq_proj_in=args.zero_init_lq_proj_in,
        freeze_lq_proj_in=args.freeze_lq_proj_in,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        stage2_attention_mode=args.stage2_attention_mode,
        stage2_topk_ratio=args.stage2_topk_ratio,
        stage2_local_num=args.stage2_local_num,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    if accelerator.is_local_main_process:
        trainable_params = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
        print(f"Stage2 v6 attention mode: {args.stage2_attention_mode}", flush=True)
        print(f"Stage2 v6 topk_ratio: {args.stage2_topk_ratio}", flush=True)
        print(f"Trainable parameter tensors: {len(trainable_params)}", flush=True)
        print(f"Trainable parameter count: {sum(numel for _, numel in trainable_params)}", flush=True)

    validation_callback = None
    if args.validation_num_samples > 0 and accelerator.is_main_process:
        print("Preparing fixed Stage2 validation samples...", flush=True)
        validation_samples = v5.collect_fixed_validation_samples(dataset, args.validation_num_samples)
        print(f"Prepared {len(validation_samples)} fixed Stage2 validation samples.", flush=True)
        validation_callback = FlashVSRStage2ValidationCallback(
            output_path=args.output_path,
            validation_samples=validation_samples,
            num_inference_steps=args.validation_num_inference_steps,
            fps=args.validation_fps,
            seed_base=(args.global_seed if args.global_seed is not None else 20260429),
            use_wandb=args.use_wandb,
        )
    accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=v5.flashvsr_stage1_export,
        validation_callback=validation_callback,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
