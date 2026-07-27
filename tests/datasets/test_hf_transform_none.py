"""hf_transform_to_torch must convert None scalars (arrow nulls in nullable
float columns) to NaN tensors — passing None through crashes torch's
default_collate in the DataLoader."""

import math

import torch

from lerobot.datasets.io_utils import hf_transform_to_torch


def test_none_scalar_becomes_nan_tensor():
    batch = {"ik.latency_ms": [None, 1.5, None], "action": [[0.0, 1.0]], "task": ["press it"]}
    out = hf_transform_to_torch(batch)
    assert isinstance(out["ik.latency_ms"][0], torch.Tensor)
    assert math.isnan(out["ik.latency_ms"][0].item())
    assert out["ik.latency_ms"][1].item() == 1.5
    assert math.isnan(out["ik.latency_ms"][2].item())
    assert torch.equal(out["action"][0], torch.tensor([0.0, 1.0]))
    assert out["task"] == ["press it"]  # LANGUAGE_COLUMNS stay strings


def test_all_none_column_becomes_nan_tensors():
    out = hf_transform_to_torch({"teleop.frame_age_ms": [None, None]})
    assert all(math.isnan(t.item()) for t in out["teleop.frame_age_ms"])
