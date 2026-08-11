#!/usr/bin/env python

# Copyright 2025 DIDEN Robotics.
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
"""Unit tests for multi_task_dit objective="streaming_flow" (SFP-S on the DiT)."""

import pytest
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    MultiTaskDiTPolicy,
    StreamingFlowObjective,
)
from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
    make_multi_task_dit_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.random_utils import set_seed

STATE_DIM = 10
ACTION_DIM = 6
HORIZON = 16
N_ACTION_STEPS = 8


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(17)


def create_config(**overrides) -> MultiTaskDiTConfig:
    kwargs = dict(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            f"{OBS_IMAGES}.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        objective="streaming_flow",
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        # Small model for fast tests.
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        sfp_integration_steps_per_action=2,
    )
    kwargs.update(overrides)
    config = MultiTaskDiTConfig(**kwargs)
    config.validate_features()
    return config


def make_policy_and_preprocessor(config: MultiTaskDiTConfig):
    """Policy + preprocessor that tokenizes the raw 'task' strings (identity norm)."""
    policy = MultiTaskDiTPolicy(config)
    policy.to(config.device)
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, _ = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)
    return policy, preprocessor


def create_train_batch(batch_size: int = 2) -> dict[str, Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 2, STATE_DIM),
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, 2, 3, 224, 224),
        ACTION: torch.randn(batch_size, HORIZON, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, HORIZON, dtype=torch.bool),
        "task": ["press the red button"] * batch_size,
    }


def create_observation_batch(batch_size: int = 2) -> dict:
    return {
        OBS_STATE: torch.randn(batch_size, STATE_DIM),
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, 3, 224, 224),
        "task": ["press the red button"] * batch_size,
    }


def test_config_validation():
    with pytest.raises(ValueError):
        create_config(sfp_sigma_0=0.5, sfp_sigma_1=0.1)  # sigma_0 > sigma_1
    with pytest.raises(ValueError):
        create_config(n_obs_steps=1)  # streaming_flow requires n_obs_steps=2
    with pytest.raises(ValueError):
        create_config(n_action_steps=HORIZON)  # > horizon - n_obs_steps + 1
    with pytest.raises(ValueError):
        create_config(integration_method="dopri5")
    with pytest.raises(ValueError):
        create_config(sfp_a0_from_state_indices=tuple(range(ACTION_DIM + 1)))  # wrong length
    with pytest.raises(ValueError):
        create_config(sfp_a0_from_state_indices=(0, 1, 2, 3, 4, STATE_DIM))  # index out of range
    cfg = create_config(sfp_a0_from_state_indices=tuple(range(ACTION_DIM)))
    assert cfg.is_streaming_flow and not cfg.is_diffusion and not cfg.is_flow_matching


def test_policy_uses_streaming_flow_objective():
    policy = MultiTaskDiTPolicy(create_config())
    assert isinstance(policy.objective, StreamingFlowObjective)


def test_loss_is_finite_and_backprops():
    policy, pre = make_policy_and_preprocessor(create_config())
    policy.train()
    loss, _ = policy.forward(pre(create_train_batch()))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.noise_predictor.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_loss_padding_mask():
    policy, pre = make_policy_and_preprocessor(create_config(do_mask_loss_for_padding=True))
    batch = create_train_batch()
    batch["action_is_pad"][:, -4:] = True
    loss, _ = policy.forward(pre(batch))
    assert torch.isfinite(loss)


def test_select_action_shape_queue_and_a0_cache():
    policy, pre = make_policy_and_preprocessor(create_config())
    policy.eval()
    obs = create_observation_batch()
    action = policy.select_action(pre(dict(obs)))
    assert action.shape == (2, ACTION_DIM)
    assert len(policy._queues[ACTION]) == N_ACTION_STEPS - 1
    assert policy._last_executed_action is not None
    assert policy._last_executed_action.shape == (2, ACTION_DIM)


def test_a0_bootstrap_and_reset():
    idxs = (5, 0, 1, 2, 3, 4)
    policy = MultiTaskDiTPolicy(create_config(sfp_a0_from_state_indices=idxs))
    batch = {OBS_STATE: torch.randn(3, 2, STATE_DIM)}
    a0 = policy._flow_start_point(batch)
    torch.testing.assert_close(a0, batch[OBS_STATE][:, -1, list(idxs)])
    # Without indices: zeros fallback.
    policy2 = MultiTaskDiTPolicy(create_config())
    a0 = policy2._flow_start_point(batch)
    assert torch.all(a0 == 0) and a0.shape == (3, ACTION_DIM)
    # Cache takes precedence once set; reset() clears it.
    cached = torch.randn(3, ACTION_DIM)
    policy2._last_executed_action = cached
    torch.testing.assert_close(policy2._flow_start_point(batch), cached)
    policy2.reset()
    assert policy2._last_executed_action is None


def test_chunk_continuity():
    """The a(0) cache after a chunk must be the chunk's last returned action."""
    policy, pre = make_policy_and_preprocessor(create_config())
    policy.eval()
    obs = create_observation_batch(batch_size=1)
    chunk_actions = [policy.select_action(pre(dict(obs))) for _ in range(N_ACTION_STEPS)]
    torch.testing.assert_close(policy._last_executed_action, chunk_actions[-1])


def test_sample_actions_shape_and_euler():
    for method in ("euler", "rk4"):
        policy = MultiTaskDiTPolicy(create_config(integration_method=method))
        policy.eval()
        conditioning = torch.randn(2, policy.observation_encoder.conditioning_dim)
        a0 = torch.randn(2, ACTION_DIM)
        actions = policy.objective.sample_actions(policy.noise_predictor, a0, conditioning)
        assert actions.shape == (2, N_ACTION_STEPS, ACTION_DIM)
        assert torch.isfinite(actions).all()


def test_velocity_targets_match_reference_math():
    """Flow construction must match the SFP-S reference (sfps.py) formulas."""
    horizon, sigma_0, sigma_1 = HORIZON, 0.1, 0.3
    sigma_r = (sigma_1**2 - sigma_0**2) ** 0.5
    actions = torch.randn(1, horizon, ACTION_DIM)
    t = torch.tensor([3.5 / (horizon - 1)])  # midway in segment 3->4
    pos = t * (horizon - 1)
    idx = pos.floor().long().clamp(max=horizon - 2)
    frac = (pos - idx.to(pos.dtype)).unsqueeze(-1)
    a_lo, a_hi = actions[0, idx], actions[0, idx + 1]
    xi = a_lo + frac * (a_hi - a_lo)
    dxi = (a_hi - a_lo) * (horizon - 1)
    assert idx.item() == 3 and abs(frac.item() - 0.5) < 1e-5
    torch.testing.assert_close(xi, 0.5 * (actions[0, 3] + actions[0, 4]).unsqueeze(0))
    z0 = torch.randn(1, ACTION_DIM)
    va = dxi + sigma_r * z0
    vz = xi + t.unsqueeze(-1) * dxi - (1 - sigma_1) * z0
    assert va.shape == vz.shape == (1, ACTION_DIM)


def test_diffusion_objective_still_default():
    """Adding streaming_flow must not change the default objective."""
    cfg = create_config(objective="diffusion")
    assert cfg.is_diffusion
    policy, pre = make_policy_and_preprocessor(cfg)
    loss, _ = policy.forward(pre(create_train_batch()))
    assert torch.isfinite(loss)
