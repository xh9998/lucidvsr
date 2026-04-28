from typing import Any, Dict, List, Optional, Sequence

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


def collate_image_video_joint_v1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Experimental joint image/video collate for paper-style f=1 image handling.

    This keeps image samples as true single-frame sequences and pads only at the
    batch level. It also emits per-sample segment lengths so an experimental
    model path can construct block-diagonal attention masks later.
    """

    if not batch:
        raise ValueError("Empty batch")
    if not all(torch.is_tensor(sample["video"]) and torch.is_tensor(sample["lq_video"]) for sample in batch):
        raise ValueError("collate_image_video_joint_v1 expects tensor outputs from the dataset")

    sequence_lengths = [int(sample["video"].shape[0]) for sample in batch]
    max_frames = max(sequence_lengths)
    videos = torch.stack([_pad_temporal_tensor(sample["video"], max_frames) for sample in batch], dim=0)
    lq_videos = torch.stack([_pad_temporal_tensor(sample["lq_video"], max_frames) for sample in batch], dim=0)
    segment_lengths = [_normalize_segment_lengths(sample, sequence_length) for sample, sequence_length in zip(batch, sequence_lengths)]

    output: Dict[str, Any] = {
        "video": videos,
        "lq_video": lq_videos,
        "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
        "segment_lengths": segment_lengths,
    }

    if "sample_seed" in batch[0] and torch.is_tensor(batch[0]["sample_seed"]):
        output["sample_seed"] = torch.stack([sample["sample_seed"] for sample in batch], dim=0)

    passthrough_keys: Sequence[str] = (
        "sample_id",
        "media_path",
        "tar_member_path",
        "source_dataset",
        "caption_text",
        "source_type",
    )
    for key in passthrough_keys:
        if key in batch[0]:
            output[key] = [sample.get(key) for sample in batch]
    return output
