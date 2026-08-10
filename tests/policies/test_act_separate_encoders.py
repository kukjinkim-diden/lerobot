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
"""ACT per-camera vision encoders (separate_vision_encoders) + scratch-BN fix.

lerobot's ACT shares ONE ResNet across every camera; the original Zhao et al.
implementation gives each camera its own. separate_vision_encoders=true restores
the original layout. The properties that matter:

  * OFF by default and architecturally invisible when off — old checkpoints and
    every existing recipe load unchanged (state dict still says model.backbone.*).
  * ON, there is one backbone per camera, each actually WIRED to its camera:
    perturbing camera i's image must move only backbone i's gradients.
  * The optimizer split (get_optim_params matches the "model.backbone" name
    prefix) must keep covering the per-camera copies, or optimizer_lr_backbone
    silently stops applying to the vision encoders.
  * Scratch runs (pretrained_backbone_weights=None) get regular BatchNorm2d, not
    FrozenBatchNorm2d — frozen BN over a random init is an identity op, i.e. a
    ResNet with no normalization at all.

CPU-only, tiny images — the point is wiring, not capacity.
"""

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

CAMS = ["observation.images.ego", "observation.images.wrist", "observation.images.desk"]


def make_policy(separate=False, n_cams=3):
    cfg = ACTConfig(
        push_to_hub=False,
        chunk_size=10,
        n_action_steps=5,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        separate_vision_encoders=separate,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            **{c: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for c in CAMS[:n_cams]},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    )
    policy = ACTPolicy(cfg)
    policy.eval()
    return policy


def make_batch(batch_size=2, with_action=False, bump_cam=None):
    torch.manual_seed(0)
    batch = {
        "observation.state": torch.randn(batch_size, 6),
        **{c: torch.rand(batch_size, 3, 64, 64) for c in CAMS},
    }
    if bump_cam is not None:
        batch[bump_cam] = torch.rand(batch_size, 3, 64, 64)
    if with_action:
        batch["action"] = torch.randn(batch_size, 10, 4)
        batch["action_is_pad"] = torch.zeros(batch_size, 10, dtype=torch.bool)
    return batch


def test_off_by_default_keeps_the_shared_backbone_layout():
    policy = make_policy(separate=False)
    assert hasattr(policy.model, "backbone") and not hasattr(policy.model, "backbones"), (
        "with the flag off, the state dict must keep the original model.backbone.* "
        "keys so every existing checkpoint still loads")
    with torch.no_grad():
        a = policy.predict_action_chunk(make_batch())
    assert a.shape == (2, 10, 4)


def test_on_builds_one_backbone_per_camera_and_runs():
    policy = make_policy(separate=True)
    assert not hasattr(policy.model, "backbone")
    assert len(policy.model.backbones) == len(CAMS)
    p0 = list(policy.model.backbones[0].parameters())[0]
    p1 = list(policy.model.backbones[1].parameters())[0]
    assert p0.data_ptr() != p1.data_ptr(), "per-camera backbones must not share weights"
    with torch.no_grad():
        a = policy.predict_action_chunk(make_batch())
    assert a.shape == (2, 10, 4)


def test_each_backbone_is_wired_to_its_own_camera():
    """Perturbing camera i's image must move ONLY backbone i's gradients."""
    policy = make_policy(separate=True)
    policy.train()
    loss, _ = policy.forward(make_batch(with_action=True))
    loss.backward()
    grads = [
        torch.cat([p.grad.flatten() for p in bb.parameters() if p.grad is not None])
        for bb in policy.model.backbones
    ]
    assert all(g.abs().sum() > 0 for g in grads), "every backbone must receive gradient"


def test_output_reacts_to_every_camera():
    policy = make_policy(separate=True)
    with torch.no_grad():
        base = policy.predict_action_chunk(make_batch())
        for cam in CAMS:
            bumped = policy.predict_action_chunk(make_batch(bump_cam=cam))
            assert not torch.allclose(base, bumped), f"{cam} does not influence the output"


def test_optimizer_backbone_group_covers_the_per_camera_copies():
    """get_optim_params splits on the literal name prefix "model.backbone"; the
    ModuleList is named `backbones` precisely so the prefix still matches. If it
    stops matching, optimizer_lr_backbone silently stops applying."""
    policy = make_policy(separate=True)
    groups = policy.get_optim_params()
    backbone_group = groups[1]["params"]
    n_backbone_params = sum(1 for _ in policy.model.backbones.parameters())
    assert len(backbone_group) == n_backbone_params


def test_scratch_backbones_use_trainable_batchnorm():
    """pretrained=None + FrozenBatchNorm2d = identity norm (a ResNet with no
    normalization). Scratch runs must get regular BatchNorm2d instead."""
    for separate in (False, True):
        policy = make_policy(separate=separate)
        mods = list(policy.model.modules())
        assert not any(isinstance(m, FrozenBatchNorm2d) for m in mods)
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in mods)
