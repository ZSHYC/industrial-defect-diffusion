import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch and torchvision are required for model tests")


def test_light_unet_forward_shape() -> None:
    import torch

    from industrial_defect.models import build_light_unet

    model = build_light_unet("cpu")
    output = model(torch.zeros(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 1, 64, 64)


def test_light_unet_checkpoint_roundtrip(tmp_path) -> None:
    import torch

    from industrial_defect.models import build_light_unet, load_light_unet_checkpoint

    model = build_light_unet("cpu")
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    loaded = load_light_unet_checkpoint(checkpoint_path, "cpu")

    assert set(loaded.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
