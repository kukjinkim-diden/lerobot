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
"""The eval loop must report the loss COMPONENTS, not only the total.

`eval_loss` is policy.forward()'s total. For ACT that is
`l1_loss + kl_weight * kld_loss`, so it cannot be compared across runs that change
`kl_weight` or `use_vae`: halving the KL coefficient shrinks the total on its own
and ranks that run as generalising better when its predictions are unchanged. The
components were already computed by the policy and thrown away
(`loss, _ = policy.forward(...)`).

The two failure modes worth pinning are both about the key set NOT being fixed:
with `use_vae=false` ACT reports no `kld_loss` at all, and a loss dict may carry
non-numeric diagnostics that would crash a naive sum.
"""

import pytest

from lerobot.scripts.lerobot_train import accumulate_loss_components, mean_loss_components


def test_components_are_averaged_over_batches_not_summed():
    sums: dict[str, float] = {}
    accumulate_loss_components(sums, {"l1_loss": 0.04, "kld_loss": 0.002})
    accumulate_loss_components(sums, {"l1_loss": 0.06, "kld_loss": 0.004})
    out = mean_loss_components(sums, 2)
    assert out == pytest.approx({"eval_l1_loss": 0.05, "eval_kld_loss": 0.003})


def test_missing_kld_is_fine_when_the_vae_is_off():
    """use_vae=false: ACT's loss dict has l1_loss only, and the whole point of the
    sweep is comparing that run against a VAE one."""
    sums: dict[str, float] = {}
    accumulate_loss_components(sums, {"l1_loss": 0.05})
    accumulate_loss_components(sums, {"l1_loss": 0.07})
    assert mean_loss_components(sums, 2) == pytest.approx({"eval_l1_loss": 0.06})


def test_non_numeric_entries_are_skipped():
    sums: dict[str, float] = {}
    accumulate_loss_components(sums, {"l1_loss": 0.05, "note": "diagnostic", "ok": True})
    assert mean_loss_components(sums, 1) == pytest.approx({"eval_l1_loss": 0.05})


def test_no_batches_does_not_divide_by_zero():
    assert mean_loss_components({}, 0) == {}
    assert mean_loss_components({"l1_loss": 0.1}, 0) == pytest.approx({"eval_l1_loss": 0.1})


def test_absent_loss_dict_is_tolerated():
    sums: dict[str, float] = {"l1_loss": 0.05}
    accumulate_loss_components(sums, None)
    assert sums == pytest.approx({"l1_loss": 0.05})


def test_l1_loss_separates_runs_that_a_total_would_confound():
    """The reason this exists: two runs with identical predictions but different
    kl_weight get different totals and the SAME l1_loss."""
    l1, kld = 0.040, 0.002
    total_kl10 = l1 + 10.0 * kld
    total_kl01 = l1 + 0.1 * kld
    assert total_kl10 != pytest.approx(total_kl01)      # totals disagree...

    sums_a: dict[str, float] = {}
    accumulate_loss_components(sums_a, {"l1_loss": l1, "kld_loss": kld})
    sums_b: dict[str, float] = {}
    accumulate_loss_components(sums_b, {"l1_loss": l1, "kld_loss": kld})
    assert (mean_loss_components(sums_a, 1)["eval_l1_loss"]
            == pytest.approx(mean_loss_components(sums_b, 1)["eval_l1_loss"]))
