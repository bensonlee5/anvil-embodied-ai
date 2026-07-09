"""Regression coverage for Anvil's custom validation and test-loss hooks."""

import torch

from anvil_trainer.patches import _normalize_uint8_camera_images


def test_custom_eval_normalizes_only_uint8_camera_images():
    batch = {
        "observation.images.waist": torch.tensor([0, 255], dtype=torch.uint8),
        "observation.state": torch.tensor([1, 2], dtype=torch.uint8),
    }

    result = _normalize_uint8_camera_images(batch, ("observation.images.waist",))

    assert result is batch
    assert batch["observation.images.waist"].dtype == torch.float32
    assert torch.equal(batch["observation.images.waist"], torch.tensor([0.0, 1.0]))
    assert batch["observation.state"].dtype == torch.uint8
