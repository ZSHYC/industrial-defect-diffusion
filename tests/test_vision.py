import importlib.util

import numpy as np
import pytest
from PIL import Image


torch_available = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torchvision is required for vision module imports")


def test_binary_mask_from_probability() -> None:
    from industrial_defect.vision import binary_mask_from_probability

    prob = np.array([[0.1, 0.5], [0.51, 0.9]], dtype=np.float32)
    mask = binary_mask_from_probability(prob, threshold=0.5)
    assert mask.tolist() == [[0, 1], [1, 1]]


def test_save_overlay_preserves_output_size(tmp_path) -> None:
    from industrial_defect.vision import save_overlay

    image_path = tmp_path / "input.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (20, 12), (120, 120, 120)).save(image_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    save_overlay(image_path, mask, output_path, output_size=(20, 12))

    assert output_path.exists()
    assert Image.open(output_path).size == (20, 12)
