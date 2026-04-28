from typing import Any, Dict, List, Sequence

import torch


def _pad_temporal_tensor(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    if tensor.shape[0] == target_frames:
        return tensor
    if tensor.shape[0] > target_frames:
        return tensor[:target_frames]
    pad_shape = (target_frames - tensor.shape[0],) + tuple(tensor.shape[1:])
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=0)


def _normalize_segment_lengths(sample: Dict[str, Any], sequence_length: int) -> List[int]:
    segment_lengths = sample.get("segment_lengths")
    if segment_lengths is None:
        return [sequence_length]
    normalized = [int(length) for length in segment_lengths if int(length) > 0]
    if sum(normalized) != sequence_length:
        raise ValueError(
            f"segment_lengths={normalized} do not sum to sequence_length={sequence_length} "
            f"for sample_id={sample.get('sample_id')}"
        )
    return normalized


def collate_image_video_joint_v5(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    V5 joint image/video collate.

    This collate keeps the raw per-sample temporal layout, pads only at the
    batch level, and forwards enough metadata for:
    - segment-aware VAE encoding
    - segment-aware LQ projection
    - segment-aware 3D patch embedding

    For grouped image samples, `video.shape[0]` is the grouped image count
    (equal to the latent-frame count for the configured `num_frames`) and
    `segment_lengths` is `[1, 1, ..., 1]`.
    """

    if not batch:
        raise ValueError("Empty batch")
    if not all(torch.is_tensor(sample["video"]) and torch.is_tensor(sample["lq_video"]) for sample in batch):
        raise ValueError("collate_image_video_joint_v5 expects tensor outputs from the dataset")

    sequence_lengths = [int(sample["video"].shape[0]) for sample in batch]
    max_frames = max(sequence_lengths)
    videos = torch.stack([_pad_temporal_tensor(sample["video"], max_frames) for sample in batch], dim=0)
    lq_videos = torch.stack([_pad_temporal_tensor(sample["lq_video"], max_frames) for sample in batch], dim=0)
    segment_lengths = [_normalize_segment_lengths(sample, sequence_length) for sample, sequence_length in zip(batch, sequence_lengths)]

    output: Dict[str, Any] = {
        "video": videos,
        "lq_video": lq_videos,
        "raw_video_list": [sample["video"] for sample in batch],
        "raw_lq_video_list": [sample["lq_video"] for sample in batch],
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        "segment_lengths": segment_lengths,
    }

    if "sample_seed" in batch[0] and torch.is_tensor(batch[0]["sample_seed"]):
        sample_seeds = [sample["sample_seed"] for sample in batch]
        first_shape = tuple(sample_seeds[0].shape)
        if all(tuple(seed.shape) == first_shape for seed in sample_seeds):
            output["sample_seed"] = torch.stack(sample_seeds, dim=0)
        else:
            # Mixed video/image_group batches can carry scalar and vector seeds.
            # Keep them as a list instead of forcing an invalid stack.
            output["sample_seed"] = sample_seeds

    passthrough_keys: Sequence[str] = (
        "sample_id",
        "media_path",
        "tar_member_path",
        "source_dataset",
        "caption_text",
        "source_type",
        "sample_kind",
        "target_num_frames",
    )
    for key in passthrough_keys:
        if key in batch[0]:
            output[key] = [sample.get(key) for sample in batch]
    if "image_group_size" in batch[0]:
        output["image_group_size"] = torch.tensor(
            [int(sample.get("image_group_size", 0)) for sample in batch],
            dtype=torch.long,
        )
    return output
