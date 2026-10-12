# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Fixed routing integration with sigmoid-group EP adapters."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.expert_parallel.recipes import deepseekv3_ep_compute_fn
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS

from tests.common.mark_utils import arg_mark


class TestFixedSigmoidRouter(unittest.TestCase):
    """Verify rank assignment, group bypass, normalization, scaling and gradients."""

    @staticmethod
    def _module() -> SimpleNamespace:
        """Build a biased group router with known contiguous EP ownership."""
        gate = torch.nn.Linear(3, 8, bias=False)
        gate.e_score_correction_bias = torch.tensor([100., 100., 0., 0., 0., 0., 0., 0.])
        return SimpleNamespace(
            gate=gate, experts=SimpleNamespace(local_expert_count=2), top_k=2,
            n_group=2, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.5,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fixed_choices_override_groups_and_regather_weights(self) -> None:
        """Feature: fixed MoE routing.
        Description: The fixed route can cross groups but must use its own unbiased weights.
        Expectation: Fixed IDs, normalized weights and gradients match unbiased scores.
        """
        module = self._module()
        states = torch.randn(5, 3, requires_grad=True)
        indices, weights = MOE_ROUTER_ADAPTERS["deepseekv3_fixed"](module, states)
        expected_indices = torch.tensor([[0, 2], [4, 6], [1, 3], [5, 7], [0, 2]])
        scores = module.gate(states).float().sigmoid().gather(1, expected_indices)
        reference = scores / (scores.sum(-1, keepdim=True) + 1e-20) * 2.5
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, reference)
        gradients = torch.autograd.grad(weights.square().sum(), (states, module.gate.weight), retain_graph=True)
        expected = torch.autograd.grad(reference.square().sum(), (states, module.gate.weight))
        for gradient, target in zip(gradients, expected):
            torch.testing.assert_close(gradient, target)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unscaled_unnormalized_fixed_weights(self) -> None:
        """Feature: fixed MoE routing.
        Description: Turning normalization off keeps the selected raw sigmoid scores.
        Expectation: Weights equal the selected sigmoid scores times the configured scale.
        """
        module = self._module()
        module.norm_topk_prob = False
        states = torch.randn(3, 3)
        indices, weights = MOE_ROUTER_ADAPTERS["deepseekv3_fixed"](module, states)
        torch.testing.assert_close(weights, module.gate(states).float().sigmoid().gather(1, indices) * 2.5)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_normal_router_keeps_bias_and_group_selection(self) -> None:
        """Feature: fixed MoE routing.
        Description: Disabling the fixed adapter retains the model's group-limited selection.
        Expectation: Learned selection obeys the correction bias and group restriction.
        """
        module = self._module()
        states = torch.randn(3, 3)
        indices, weights = MOE_ROUTER_ADAPTERS["deepseekv3"](module, states)
        self.assertTrue(bool((indices < 2).all()))
        scores = module.gate(states).float().sigmoid().gather(1, indices)
        torch.testing.assert_close(weights, scores / (scores.sum(-1, keepdim=True) + 1e-20) * 2.5)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_factory_defaults_off_and_selects_fixed_adapter(self) -> None:
        """Feature: fixed MoE routing.
        Description: Both existing spellings resolve to the corrected fixed adapter.
        Expectation: The factory selects the requested adapter and both aliases agree.
        """
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch(
                "hyper_parallel.distributed.expert_parallel.recipes.build_ep_compute"
            ) as build:
                deepseekv3_ep_compute_fn(
                    module=self._module(), mesh=None, tp_mesh=None, cp_mesh=None,
                    ep_mesh=None, fix_router=enabled,
                )
                expected = MOE_ROUTER_ADAPTERS["deepseekv3_fixed" if enabled else "deepseekv3"]
                self.assertIs(build.call_args.kwargs["router_fn"], expected)
        self.assertIs(MOE_ROUTER_ADAPTERS["deepseek_v3_fixed"], MOE_ROUTER_ADAPTERS["deepseekv3_fixed"])
