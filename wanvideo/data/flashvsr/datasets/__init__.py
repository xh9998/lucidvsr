__all__ = ["FlashVSRStreamingDataset"]


def __getattr__(name):
    if name == "FlashVSRStreamingDataset":
        from .streaming_dataset import FlashVSRStreamingDataset

        return FlashVSRStreamingDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
