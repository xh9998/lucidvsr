"""Stage3 v7-D4.4 overfit entrypoint.

This wrapper intentionally keeps the production v7-D4.4 training code intact.
It only installs deterministic data hooks that make a tiny overfit run useful
for debugging loss wiring and train/inference alignment.
"""

from __future__ import annotations

import os


def _install_fixed_sample_seed_patch() -> None:
    fixed_seed = os.environ.get("FLASHVSR_OVERFIT_FIXED_SAMPLE_SEED")
    if fixed_seed in (None, ""):
        return

    from wanvideo.data.flashvsr.datasets import streaming_dataset

    seed = int(fixed_seed)

    def _fixed_next_sample_seed(self, rng):  # noqa: ARG001
        return seed

    streaming_dataset.FlashVSRStreamingDataset._next_sample_seed = _fixed_next_sample_seed
    print(f"[stage3_overfit] fixed degradation sample_seed={seed}", flush=True)


def _install_gt_sharpen_patch() -> None:
    """Apply Stage1-USMGT-style GT sharpening only for isolated OF probes."""
    enabled = os.environ.get("FLASHVSR_OVERFIT_GT_SHARPEN", "").lower() in ("1", "true", "yes", "y")
    if not enabled:
        return

    from wanvideo.data.flashvsr.datasets import streaming_dataset
    from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53_usmgt import ConsistentClipGTSharpen

    backend = os.environ.get("FLASHVSR_OVERFIT_GT_SHARPEN_BACKEND", "torch")
    device = os.environ.get("FLASHVSR_OVERFIT_GT_SHARPEN_DEVICE", "auto")
    sharpener = ConsistentClipGTSharpen(backend=backend, device=device)

    original_process_video_bytes = streaming_dataset.FlashVSRStreamingDataset._process_video_bytes

    def _process_video_bytes_with_sharpen(self, video_bytes, sample_id, rng):
        frames = self._extract_frames(video_bytes)
        if frames is None:
            return None
        clip = self._select_clip(frames, rng=rng)
        if clip is None:
            return None
        sample_seed = self._next_sample_seed(rng)
        gt_clip = sharpener.sharpen_batch(clip)
        lq_video = self._build_lq_clip(gt_clip, rng=rng, sample_seed=sample_seed)
        return self._maybe_convert_output(
            {
                "video": gt_clip,
                "lq_video": lq_video,
                "sample_id": sample_id,
                "source_type": "video",
                "sample_seed": sample_seed,
            }
        )

    streaming_dataset.FlashVSRStreamingDataset._process_video_bytes = _process_video_bytes_with_sharpen
    print(
        "[stage3_overfit] enabled Stage1-USMGT-style GT sharpening "
        f"backend={backend} device={device} patched_from={original_process_video_bytes.__name__}",
        flush=True,
    )


def main() -> None:
    _install_fixed_sample_seed_patch()
    _install_gt_sharpen_patch()
    from wanvideo.model_training.flashvsr.train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora import main as stage3_main

    print("[stage3_overfit] using isolated train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py", flush=True)
    stage3_main()


if __name__ == "__main__":
    main()
