from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wanvideo.model_training.flashvsr.train_flashvsr_stage1_v5_3_lora as base_train
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v532_yubari_frames import (
    FlashVSRTarStreamingDatasetV532YubariFrames,
)


# Keep the main v5.3 / v5.3.1 implementation untouched.
# This wrapper redirects only this dedicated experimental training entry.
base_train.FlashVSRTarStreamingDatasetV53 = FlashVSRTarStreamingDatasetV532YubariFrames


if __name__ == "__main__":
    base_train.main()
