# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Muon: orthogonalized momentum for matrix weights, aux AdamW for the rest.

What is worth pinning is not "the loss goes down" alone but the three properties
that make Muon Muon — and the two integration seams where a silent mistake would
train something else entirely:

  * Newton-Schulz really orthogonalizes: singular values land in a band around 1
    (the quintic iteration deliberately does NOT converge them exactly to 1).
  * Membership by shape: matrices go to Muon, biases/gains to AdamW, conv kernels
    flattened rather than dropped.
  * ACT's param groups survive: a group carrying its own "lr" (the pretrained
    backbone at optimizer_lr_backbone) must keep that lr on BOTH halves — a
    pretrained ResNet at the full Muon rate is the failure this protects.
  * The registry knows "muon", so --optimizer.type=muon resolves.
  * TrainPipelineConfig.validate() honours an explicit optimizer instead of
    silently replacing it with the policy preset (the same bug class the
    scheduler override fixed).

CPU-only; the largest tensor is 64x48.
"""

import pytest
import torch

from lerobot.optim.optimizers import (
    MuonConfig,
    OptimizerConfig,
    _MuonWithAuxAdamW,
    _zeropower_via_newtonschulz5,
)


def test_newton_schulz_orthogonalizes_into_a_band():
    g = torch.randn(32, 48, dtype=torch.float32)
    o = _zeropower_via_newtonschulz5(g, steps=5)
    s = torch.linalg.svdvals(o.float())
    # input singular values are spread over ~[0.1, 10]; the iteration must pull
    # them into a band around 1 (bf16 + quintic => a band, not exactly 1)
    assert s.max() < 1.6 and s.min() > 0.3, f"singular values {s.min():.2f}..{s.max():.2f}"
    assert o.shape == g.shape


def test_newton_schulz_handles_tall_matrices_by_transposing():
    g = torch.randn(48, 16)
    o = _zeropower_via_newtonschulz5(g, steps=5)
    assert o.shape == g.shape
    s = torch.linalg.svdvals(o.float())
    assert s.max() < 1.6 and s.min() > 0.3


def test_newton_schulz_rejects_non_2d():
    with pytest.raises(ValueError):
        _zeropower_via_newtonschulz5(torch.randn(4, 4, 4), steps=5)


def _param(*shape):
    p = torch.nn.Parameter(torch.randn(*shape))
    p.grad = torch.randn(*shape)
    return p


def test_membership_is_decided_by_shape():
    w2d, w4d, bias = _param(16, 8), _param(8, 4, 3, 3), _param(16)
    opt = MuonConfig().build([w2d, w4d, bias])
    assert isinstance(opt, _MuonWithAuxAdamW)
    by_flag = {g["use_muon"]: g["params"] for g in opt.param_groups}
    # identity, not `in`: `in` triggers elementwise tensor comparison
    ids = {k: {id(p) for p in v} for k, v in by_flag.items()}
    assert id(w2d) in ids[True] and id(w4d) in ids[True]  # conv joins Muon (flattened)
    assert id(bias) in ids[False]
    opt.step()
    # muon state = momentum buffer; adamw state = exp_avg — proves each param was
    # stepped by the intended algorithm, not just sorted into a group
    assert "momentum_buffer" in opt.state[w2d] and "momentum_buffer" in opt.state[w4d]
    assert "exp_avg" in opt.state[bias]


def test_act_style_group_lr_is_preserved_for_both_halves():
    """ACT's get_optim_params yields a backbone group with its own lr
    (--policy.optimizer_lr_backbone). Losing it would put a pretrained ResNet on
    the full Muon rate — the knob this config promises to keep."""
    main_w, backbone_w, backbone_bias = _param(16, 8), _param(8, 8), _param(8)
    opt = MuonConfig(lr=0.02, adamw_lr=3e-4).build([
        {"params": [main_w]},
        {"params": [backbone_w, backbone_bias], "lr": 1e-5},
    ])
    lrs = {(g["use_muon"], id(g["params"][0])): g["lr"] for g in opt.param_groups}
    assert lrs[(True, id(main_w))] == 0.02          # config default where no group lr
    assert lrs[(True, id(backbone_w))] == 1e-5      # group lr kept on the Muon half
    assert lrs[(False, id(backbone_bias))] == 1e-5  # ...and on the AdamW half


def test_muon_reduces_loss_on_a_small_regression():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    x, y = torch.randn(256, 16), torch.randn(256, 4)
    opt = MuonConfig(lr=0.02, adamw_lr=3e-4).build(list(net.parameters()))
    first = last = None
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.6, f"loss barely moved: {first:.4f} -> {last:.4f}"


def test_registry_resolves_muon():
    assert "muon" in dict(OptimizerConfig.get_known_choices())


def test_validate_honours_an_explicit_optimizer():
    """Mirror of the scheduler-override test: without the train.py patch,
    --optimizer.type=muon silently trains with the policy's preset AdamW."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    muon = MuonConfig()
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=ACTConfig(push_to_hub=False),
        optimizer=muon,
    )
    cfg.validate()
    assert cfg.optimizer is muon, "the policy preset replaced an explicit optimizer"

    # and with nothing configured the preset still applies (older runs unchanged)
    cfg2 = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=ACTConfig(push_to_hub=False),
    )
    cfg2.validate()
    assert cfg2.optimizer is not None and cfg2.optimizer.type == "adamw"
