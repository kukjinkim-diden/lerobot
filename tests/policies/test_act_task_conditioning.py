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
"""ACT multi-task conditioning via a task-index embedding token.

Vanilla ACT ignores the task entirely — modeling_act.py contained zero uses of
batch["task_index"] — so a model trained on the aggregated 10-button dataset has
no way to know WHICH button to press: all five are visible, and the instruction
exists only in the task field it discards. n_task_embeddings adds one learned
token to the transformer encoder, looked up from task_index.

The properties that matter:

  * OFF by default and architecturally invisible when off — old checkpoints and
    every existing recipe load and run unchanged, and a batch that happens to
    carry task_index is still ignored.
  * ON, the token actually CONDITIONS the output: same observation, different
    task_index -> different actions. Without this the embedding could silently
    train to zero influence and multi-task success would read as a data problem.
  * Loud failures at the two seams: a missing task_index at inference (the
    caller forgot the mapping) and an out-of-range index (on CUDA an embedding
    overflow is a device-side assert with no message).

CPU-only, tiny images (2 x 64x64) — the point is wiring, not capacity.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def make_policy(n_tasks=None):
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
        n_task_embeddings=n_tasks,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    )
    policy = ACTPolicy(cfg)
    policy.eval()
    return policy


def make_batch(task_index=None, batch_size=2, with_action=False):
    torch.manual_seed(0)
    batch = {
        "observation.state": torch.randn(batch_size, 6),
        "observation.images.cam": torch.rand(batch_size, 3, 64, 64),
    }
    if with_action:
        batch["action"] = torch.randn(batch_size, 10, 4)
        batch["action_is_pad"] = torch.zeros(batch_size, 10, dtype=torch.bool)
    if task_index is not None:
        batch["task_index"] = torch.tensor([task_index] * batch_size)
    return batch


def test_off_by_default_and_task_index_is_ignored():
    policy = make_policy(n_tasks=None)
    assert not hasattr(policy.model, "encoder_task_embed"), (
        "no embedding may exist when conditioning is off — its parameters would "
        "change the optimizer's param count and the checkpoint layout")
    with torch.no_grad():
        a_without = policy.predict_action_chunk(make_batch())
        a_with = policy.predict_action_chunk(make_batch(task_index=3))
    assert torch.allclose(a_without, a_with), (
        "with conditioning off, a stray task_index in the batch must change nothing")


def test_task_index_conditions_the_output():
    policy = make_policy(n_tasks=10)
    with torch.no_grad():
        a0 = policy.predict_action_chunk(make_batch(task_index=0))
        a0_again = policy.predict_action_chunk(make_batch(task_index=0))
        a7 = policy.predict_action_chunk(make_batch(task_index=7))
    assert torch.allclose(a0, a0_again), "eval-mode forward must be deterministic"
    assert not torch.allclose(a0, a7), (
        "different task_index must produce different actions — otherwise the "
        "token is dead weight and the model still cannot tell tasks apart")


def test_missing_task_index_fails_loudly():
    policy = make_policy(n_tasks=10)
    with pytest.raises(KeyError, match="task_index"):
        with torch.no_grad():
            policy.predict_action_chunk(make_batch())


def test_out_of_range_index_fails_with_the_numbers():
    policy = make_policy(n_tasks=10)
    with pytest.raises(ValueError, match="out of range"):
        with torch.no_grad():
            policy.predict_action_chunk(make_batch(task_index=10))


def test_training_forward_carries_the_gradient_into_the_embedding():
    """The embedding must actually take part in the loss, or it trains to nothing."""
    policy = make_policy(n_tasks=10)
    policy.train()
    loss, _ = policy.forward(make_batch(task_index=2, with_action=True))
    loss.backward()
    g = policy.model.encoder_task_embed.weight.grad
    assert g is not None
    assert g[2].abs().sum() > 0, "the looked-up row must receive gradient"
    assert g[5].abs().sum() == 0, "rows of other tasks must not"
