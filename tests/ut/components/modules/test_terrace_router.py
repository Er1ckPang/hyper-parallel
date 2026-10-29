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
"""Adversarial routing contracts and gradient checks for T-Route."""

import unittest

import torch

from hyper_parallel.components.modules.moe import MoE, TokenChoiceTopKRouter
from hyper_parallel.components.modules.terrace_router import (
    TerraceRouteConfig, TerraceRouter, terrace_route, terrace_select,
)
from tests.common.mark_utils import arg_mark


class TestTerraceRouter(unittest.TestCase):
    """Verify exact quotas independently of statistical load balance."""

    def setUp(self) -> None:
        """Prepare an adversarial distribution concentrated on the first group."""
        self.config = TerraceRouteConfig(16, 4, 4, 2)
        self.affinity = torch.tensor([
            [.99, .98, .97, .96, .8, .7, .6, .5, .4, .3, .2, .1, .09, .08, .07, .06],
        ]).repeat(8, 1)
        self.bias = torch.zeros(16)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_adversarial_fanout_and_quota(self) -> None:
        """Feature: T-Route fanout and per-group quotas.

        Description: Concentrate affinities on the first expert group.
        Expectation: Each token selects two experts in each of two groups.
        """
        indices, gates, groups = terrace_route(self.affinity, self.bias, self.config)
        self.assertEqual(indices[0].tolist(), [0, 1, 4, 5])
        self.assertEqual(groups[0].tolist(), [0, 1])
        histogram = torch.zeros(8, 4).scatter_add_(1, indices // 4, torch.ones(8, 4))
        torch.testing.assert_close(histogram, torch.tensor([[2., 2., 0., 0.]]).repeat(8, 1))
        torch.testing.assert_close(gates.sum(-1), torch.ones(8))
        self.assertGreater(histogram.sum(0).std().item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_precomputed_selection_preserves_caller_gate_policy(self) -> None:
        """Feature: Precomputed routing selection.

        Description: Select IDs without performing gate normalization.
        Expectation: Selected IDs and gathered affinities preserve caller values.
        """
        indices, groups = terrace_select(self.affinity, self.config)
        self.assertEqual(indices[0].tolist(), [0, 1, 4, 5])
        self.assertEqual(groups[0].tolist(), [0, 1])
        self.assertEqual(self.affinity.gather(1, indices)[0].tolist(),
                         self.affinity[0, [0, 1, 4, 5]].tolist())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_native_global_router_accepts_precomputed_affinities(self) -> None:
        """Feature: Native global top-k selection.

        Description: Supply precomputed affinities and selection-only bias.
        Expectation: Bias changes selected IDs without changing gathered values.
        """
        router = TokenChoiceTopKRouter(1, 16, 4)
        bias = torch.zeros(16)
        bias[12:] = 2
        values, indices, counts = router.select_experts(self.affinity, bias)
        self.assertEqual(set(indices[0].tolist()), {12, 13, 14, 15})
        torch.testing.assert_close(values, self.affinity.gather(1, indices))
        self.assertEqual(counts.sum().item(), 32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_four_modes_are_live(self) -> None:
        """Feature: Routing ablation modes.

        Description: Use skewed affinities, bias and group-score pooling.
        Expectation: Quota, limited-group and unrestricted selections differ.
        """
        selections = {}
        for mode in ("full", "group_limited", "quota_only", "global_topk"):
            indices, _, _ = terrace_route(self.affinity, self.bias, self.config, mode)
            selections[mode] = tuple(sorted(indices[0].tolist()))
        self.assertEqual(selections["full"], (0, 1, 4, 5))
        self.assertEqual(selections["quota_only"], (0, 4, 8, 12))
        self.assertEqual(selections["group_limited"], (0, 1, 2, 3))
        bias = self.bias.clone()
        bias[12:] = 1
        global_ids, _, _ = terrace_route(self.affinity, bias, self.config, "global_topk")
        limited_ids, _, _ = terrace_route(self.affinity, bias, self.config, "group_limited")
        # Both modes may coincide on some inputs; changing group-score pooling
        # exposes their different candidate sets without relying on randomness.
        self.assertEqual(global_ids.shape, limited_ids.shape)
        cfg = TerraceRouteConfig(16, 4, 4, 1, group_score_pool=4)
        uneven = torch.tensor([[.99, .01, .02, .03, .8, .79, .78, .77, .5, .4, .3, .2, .1, .09, .08, .07]])
        global_ids, _, _ = terrace_route(uneven, self.bias, cfg, "global_topk")
        limited_ids, _, _ = terrace_route(uneven, self.bias, cfg, "group_limited")
        self.assertNotEqual(set(global_ids[0].tolist()), set(limited_ids[0].tolist()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_bias_changes_selection_but_not_gate_values(self) -> None:
        """Feature: Selection-only balancing bias.

        Description: Favor the final two expert groups through external bias.
        Expectation: Gate values normalize unbiased affinities at selected IDs.
        """
        bias = self.bias.clone()
        bias[8:] = 2
        indices, gates, groups = terrace_route(self.affinity, bias, self.config)
        self.assertEqual(set(groups[0].tolist()), {2, 3})
        expected = self.affinity.gather(1, indices)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(gates, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_selecting_all_groups_is_strictly_balanced(self) -> None:
        """Feature: All-group quota balance.

        Description: Select every group under adversarial affinities.
        Expectation: Every group receives exactly the same token count.
        """
        config = TerraceRouteConfig(16, 4, 8, 4)
        indices, _, _ = terrace_route(self.affinity, self.bias, config)
        group_load = torch.bincount((indices // 4).reshape(-1), minlength=4)
        torch.testing.assert_close(group_load, torch.full((4,), 16, dtype=torch.int64))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_forward_leaves_external_bias_unchanged(self) -> None:
        """Feature: Recomputed routing forwards.

        Description: Route identical token states twice with external bias.
        Expectation: Bias remains unchanged and both routing results match.
        """
        router = TerraceRouter(8, self.config)
        bias = torch.arange(16).float() * .03
        before = bias.clone()
        states = torch.randn(7, 8)
        first = router(states, bias)
        second = router(states, bias)
        torch.testing.assert_close(bias, before, rtol=0, atol=0)
        for actual, expected in zip(first, second):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_selected_gates_have_correct_gradient(self) -> None:
        """Feature: Selected gate gradients.

        Description: Compare autograd derivatives with finite differences.
        Expectation: Gradcheck passes through unbiased gate normalization.
        """
        affinity = self.affinity[:1].double().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: terrace_route(value, self.bias.double(), self.config)[1],
            (affinity,),
        ))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_geometry_and_modes(self) -> None:
        """Feature: Routing boundary validation.

        Description: Supply invalid quotas, group counts and mode names.
        Expectation: Invalid configurations raise ValueError before dispatch.
        """
        for args in ((0, 4, 4, 2), (16, 0, 4, 2), (16, 4, 4, 0), (16, 4, 4, 5),
                     (15, 4, 4, 2), (16, 4, 3, 2), (16, 4, 10, 2), (True, 1, 1, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TerraceRouteConfig(*args)
        with self.assertRaises(ValueError):
            terrace_route(self.affinity, self.bias, self.config, "Full")
        with self.assertRaises(ValueError):
            TerraceRouter(8, TerraceRouteConfig(16, 4, 2, 2), "quota_only")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_empty_and_zero_affinities(self) -> None:
        """Feature: Empty batches and affinity underflow.

        Description: Route empty inputs in every mode and all-zero affinities.
        Expectation: Output shapes remain valid and gates contain no NaNs.
        """
        for mode in ("full", "group_limited", "quota_only", "global_topk"):
            with self.subTest(mode=mode):
                indices, gates, _ = terrace_route(torch.empty(0, 16), self.bias, self.config, mode)
                self.assertEqual(indices.shape, (0, 4))
                self.assertEqual(gates.numel(), 0)
        _, gates, _ = terrace_route(torch.zeros_like(self.affinity), self.bias, self.config)
        self.assertTrue(torch.isfinite(gates).all())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos", "cpu_windows"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_hyper_moe_router_replacement(self) -> None:
        """Feature: TerraceRouter integration with the existing MoE owner.

        Description: Replace the router and run a complete forward and backward.
        Expectation: Shapes, finite input gradients and router gradients agree.
        """
        torch.manual_seed(7)
        moe = MoE(dim=8, hidden_dim=16, num_experts=16, top_k=4)
        moe.router = TerraceRouter(8, self.config)
        states = torch.randn(1, 7, 8, requires_grad=True)
        output = moe(states)
        output.square().mean().backward()
        self.assertEqual(output.shape, states.shape)
        self.assertTrue(torch.isfinite(states.grad).all())
        self.assertGreater(moe.router.gate.weight.grad.abs().sum().item(), 0)
