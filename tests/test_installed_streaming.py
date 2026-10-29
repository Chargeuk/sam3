"""Preserve the installed desktop streaming contract while tracking SAM3 main."""
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from sam3.model.io_utils import VideoFrameDataset, StreamingVideoFrameLoader
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
import sam3


def test_numpy2_pil_frames_keep_installed_half_normalization():
    frames = [Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)) for value in (0, 255)]
    dataset = VideoFrameDataset(frames, image_size=4)
    assert len(dataset) == 2
    assert dataset[0].dtype == torch.float16
    torch.testing.assert_close(dataset[0], torch.full((3, 4, 4), -1.0, dtype=torch.float16))
    torch.testing.assert_close(dataset[1], torch.ones((3, 4, 4), dtype=torch.float16))


def test_streaming_options_and_packaged_vocabulary_remain_available():
    parameters = inspect.signature(Sam3VideoInferenceWithInstanceInteractivity.init_state).parameters
    assert {"use_streaming", "streaming_cache_size", "streaming_num_workers"} <= set(parameters)
    assert (Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="installed streaming VRAM cache requires CUDA")
def test_streaming_cache_remains_bounded_and_clearable():
    frames = [Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)) for value in (0, 255, 128)]
    loader = StreamingVideoFrameLoader(frames, image_size=4, device=torch.device("cuda"), cache_size=1, num_workers=0)
    try:
        for index in (0, 2, 1):
            assert loader[index].device.type == "cuda"
        loader.stop_prefetch()
        assert len(loader._gpu_cache) <= 1
        loader.clear_cache()
        assert not loader._gpu_cache
    finally:
        loader.stop_prefetch()
        loader.clear_cache()
