#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

# ruff: noqa: E402

"""multi_task_dit task-index conditioning (conditioning_mode="task_index").

Language conditioning (CLIP text encoder over the per-frame instruction) is the
default and is unaffected by these tests. conditioning_mode="task_index" swaps
the text-encoder slot in the conditioning vector for a learned
nn.Embedding(n_tasks, hidden_dim) looked up from batch["task_index"] — for
closed, small task sets where testing whether the policy actually parses
language is not the point yet.

The properties that matter, mirroring the ACT n_task_embeddings precedent:
  * "language" mode is architecturally unaffected — no task_embed module, and a
    stray task_index in the batch changes nothing.
  * "task_index" mode actually conditions the output: same observation,
    different task_index -> different actions/conditioning vector.
  * Loud failures at the two seams: a missing task_index (KeyError) and an
    out-of-range index (ValueError) — a CUDA embedding overflow is a
    device-side assert with no message otherwise.
  * Gradient flows into exactly the looked-up embedding row.

To run locally:
    python -m pytest tests/policies/multi_task_dit/test_multi_task_dit_task_conditioning.py -v
"""

import os

import pytest
import torch

pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local transformers installation and is not meant for CI",
)

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def make_config(conditioning_mode: str = "language", n_tasks: int | None = None) -> MultiTaskDiTConfig:
    """No visual features: keeps task_index tests fast and network-free (no CLIP
    vision encoder to instantiate; in task_index mode there is no CLIP text
    encoder either)."""
    config = MultiTaskDiTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        conditioning_mode=conditioning_mode,
        n_tasks=n_tasks,
    )
    config.validate_features()
    return config


def make_batch(
    task_index: int | list[int] | None = None,
    batch_size: int = 2,
    n_obs_steps: int = 2,
    state_dim: int = 6,
    with_action: bool = False,
    horizon: int = 8,
    action_dim: int = 4,
):
    torch.manual_seed(0)
    batch = {OBS_STATE: torch.randn(batch_size, n_obs_steps, state_dim)}
    if with_action:
        batch[ACTION] = torch.randn(batch_size, horizon, action_dim)
    if task_index is not None:
        indices = task_index if isinstance(task_index, list) else [task_index] * batch_size
        batch["task_index"] = torch.tensor(indices)
    return batch


def test_language_mode_has_no_task_embed_and_ignores_task_index():
    config = make_config(conditioning_mode="language")
    policy = MultiTaskDiTPolicy(config=config)
    policy.eval()

    assert policy.observation_encoder.task_embed is None, (
        "no embedding may exist in language mode — its parameters would change "
        "the optimizer's param count and the checkpoint layout"
    )

    with torch.no_grad():
        out_without = policy.observation_encoder.encode(make_batch(task_index=None))
        out_with = policy.observation_encoder.encode(make_batch(task_index=3))
    assert torch.allclose(out_without, out_with), (
        "in language mode, a stray task_index in the batch must change nothing"
    )


def test_task_index_mode_has_no_text_encoder():
    config = make_config(conditioning_mode="task_index", n_tasks=5)
    policy = MultiTaskDiTPolicy(config=config)
    assert policy.observation_encoder.text_encoder is None
    assert policy.observation_encoder.task_embed is not None
    assert policy.observation_encoder.task_embed.num_embeddings == 5


def test_task_index_conditions_the_output():
    config = make_config(conditioning_mode="task_index", n_tasks=5)
    policy = MultiTaskDiTPolicy(config=config)
    policy.eval()

    with torch.no_grad():
        out0 = policy.observation_encoder.encode(make_batch(task_index=0))
        out0_again = policy.observation_encoder.encode(make_batch(task_index=0))
        out3 = policy.observation_encoder.encode(make_batch(task_index=3))

    assert torch.allclose(out0, out0_again), "eval-mode encode must be deterministic"
    assert not torch.allclose(out0, out3), (
        "different task_index must produce a different conditioning vector — "
        "otherwise the embedding is dead weight and the model cannot tell tasks apart"
    )


def test_missing_task_index_fails_loudly():
    config = make_config(conditioning_mode="task_index", n_tasks=5)
    policy = MultiTaskDiTPolicy(config=config)
    with pytest.raises(KeyError, match="task_index"):
        policy.observation_encoder.encode(make_batch(task_index=None))


def test_out_of_range_task_index_fails_with_the_numbers():
    config = make_config(conditioning_mode="task_index", n_tasks=5)
    policy = MultiTaskDiTPolicy(config=config)
    with pytest.raises(ValueError, match="out of range"):
        policy.observation_encoder.encode(make_batch(task_index=5))


def test_training_forward_carries_gradient_into_looked_up_row_only():
    """The DiT blocks zero-init their adaLN modulation's last layer (standard
    "adaLN-zero" stability trick — see DiffusionTransformer._initialize_weights),
    so on the very first forward/backward the conditioning vector's gradient
    (language, vision, state, task_index — all of it) is exactly zero: d(output)/
    d(input) for that layer is its weight matrix, which is 0 at init. A few warm-up
    optimizer steps move those weights off zero so this test actually exercises the
    task_index wiring instead of the shared zero-init artifact.
    """
    config = make_config(conditioning_mode="task_index", n_tasks=5)
    policy = MultiTaskDiTPolicy(config=config)
    policy.train()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = policy.forward(make_batch(task_index=1, with_action=True))
        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    loss, _ = policy.forward(make_batch(task_index=2, with_action=True))
    loss.backward()

    grad = policy.observation_encoder.task_embed.weight.grad
    assert grad is not None
    assert grad[2].abs().sum() > 0, "the looked-up row must receive gradient"
    assert grad[0].abs().sum() == 0, "rows of other tasks must not"
    assert grad[4].abs().sum() == 0, "rows of other tasks must not"


def test_conditioning_mode_validation():
    with pytest.raises(ValueError, match="conditioning_mode"):
        make_config(conditioning_mode="bogus")
    with pytest.raises(ValueError, match="n_tasks"):
        make_config(conditioning_mode="task_index", n_tasks=None)
