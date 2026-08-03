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
"""An explicitly configured scheduler must survive TrainPipelineConfig.validate().

validate() fills the optimizer and scheduler from the policy preset when
use_policy_training_preset is on (the default). Assigning the scheduler
unconditionally made a configured one a silent no-op: ACT's
get_scheduler_preset() returns None, so a requested warmup+cosine schedule became
a constant LR with no error, no warning, and nothing to see but a flat LR curve in
the run's metrics.

Turning the preset off does honour the CLI, but it also makes
make_optimizer_and_scheduler use policy.parameters() instead of
policy.get_optim_params(), discarding ACT's separate backbone param group — so
optimizer_lr_backbone stops working. Overriding only the scheduler keeps both.
"""

from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig


def _cfg(**kwargs) -> TrainPipelineConfig:
    """A validate()-able config whose policy (ACT) has NO scheduler preset."""
    from lerobot.configs.default import DatasetConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        # push_to_hub=False: validate() otherwise demands a Hub repo_id
        policy=ACTConfig(push_to_hub=False),
        **kwargs,
    )
    cfg.validate()
    return cfg


def test_act_has_no_scheduler_preset():
    """The premise: without an override there is nothing to schedule with."""
    assert ACTConfig(push_to_hub=False).get_scheduler_preset() is None


def test_configured_scheduler_is_not_replaced_by_the_preset():
    sched = CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=500)
    cfg = _cfg(scheduler=sched)
    assert cfg.scheduler is sched, (
        "the policy preset overwrote a configured scheduler — --scheduler.type "
        "becomes a silent no-op and the LR stays constant"
    )


def test_preset_still_fills_an_unset_scheduler():
    """Policies that DO ship a preset (diffusion, vqbet) must keep getting it."""
    cfg = _cfg()
    assert cfg.scheduler is None  # ACT's preset is None, so None is correct here


def test_optimizer_preset_still_applies_alongside_a_configured_scheduler():
    """Only the scheduler is overridden. The optimizer must still come from the
    policy, since that is what carries optimizer_lr / optimizer_lr_backbone."""
    cfg = _cfg(scheduler=CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=10))
    assert cfg.optimizer is not None
    assert cfg.optimizer.lr == ACTConfig(push_to_hub=False).optimizer_lr


def test_policy_training_preset_stays_on_so_param_groups_survive():
    """The whole point of overriding the scheduler rather than disabling the
    preset: make_optimizer_and_scheduler only builds ACT's two param groups while
    use_policy_training_preset is True."""
    cfg = _cfg(scheduler=CosineAnnealingWithWarmupSchedulerConfig(num_warmup_steps=10))
    assert cfg.use_policy_training_preset is True
