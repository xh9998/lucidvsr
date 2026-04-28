__all__ = ["FlashVSRStreamingDataset", "FlashVSRTarStreamingDatasetV3"]


def __getattr__(name):
    if name == "FlashVSRStreamingDataset":
        from .streaming_dataset import FlashVSRStreamingDataset

        return FlashVSRStreamingDataset
    if name == "FlashVSRTarStreamingDatasetV3":
        from .tar_streaming_dataset_v3 import FlashVSRTarStreamingDatasetV3

        return FlashVSRTarStreamingDatasetV3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
