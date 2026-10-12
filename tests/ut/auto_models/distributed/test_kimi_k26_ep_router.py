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
"""Balanced-router contract of the Kimi-K2.6 EP archetype factory.

``fix_router=True`` is a benchmark-only switch: it must give every destination
rank exactly the same slot count (that is what stops a skewed real router from
dominating step time) while keeping the real gate weights differentiable. These
tests pin the arithmetic and the switch wiring without any distributed setup.
"""

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch  # pylint: disable=wrong-import-position

from tests.common.mark_utils import arg_mark  # pylint: disable=wrong-import-position

from hyper_parallel.components.functional.fix_router import fixed_expert_indices  # pylint: disable=wrong-import-position
from hyper_parallel.distributed.expert_parallel import (  # pylint: disable=wrong-import-position
    routing as routing_module,
)
from hyper_parallel.models.kimi_k26.adapter.distributed import (  # pylint: disable=wrong-import-position
    ep_compute,
)


# These stub modules never run a forward pass; they only expose the attributes
# the balanced router reads, so the abstract ``forward`` hook is left out.
class _FakeExperts(torch.nn.Module):  # pylint: disable=abstract-method
    """Stand-in exposing the attributes the balanced router reads."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Store the expert layout the router reads back."""
        super().__init__()
        self.local_expert_count = local_expert_count
        self.num_experts = num_experts


# Minimal attribute carrier for the router; no forward pass is ever executed.
class _FakeMoE(torch.nn.Module):  # pylint: disable=abstract-method
    """Minimal MoE block carrying an ``experts`` child."""

    def __init__(self, local_expert_count: int, num_experts: int) -> None:
        """Expose an ``experts`` child with the given layout."""
        super().__init__()
        self.experts = _FakeExperts(local_expert_count, num_experts)


class TestBalancedExpertIndices(unittest.TestCase):
    """The round-robin assignment itself."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_spreads_slots_evenly_over_destination_ranks(self) -> None:
        """Feature: fixed MoE routing.
        Description: Every destination rank gets the same slot count; experts differ by <=1.
        Expectation: Each destination receives equal load and expert spread is at most one.
        """
        tokens, per_token, total, local = 10, 2, 8, 2
        indices = fixed_expert_indices(
            tokens, per_token, total, local_expert_count=local, device=torch.device("cpu"))

        self.assertEqual(tuple(indices.shape), (tokens, per_token))
        self.assertEqual(indices.dtype, torch.int64)
        self.assertTrue(bool((indices >= 0).all() and (indices < total).all()))

        destinations = torch.div(indices.reshape(-1), local, rounding_mode="floor")
        per_destination = torch.bincount(destinations, minlength=total // local)
        self.assertEqual(len(set(per_destination.tolist())), 1,
                         f"destination load is not uniform: {per_destination.tolist()}")

        per_expert = torch.bincount(indices.reshape(-1), minlength=total)
        self.assertLessEqual(int(per_expert.max() - per_expert.min()), 1,
                             f"expert load spread too wide: {per_expert.tolist()}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_indivisible_expert_layout(self) -> None:
        """Feature: fixed MoE routing.
        Description: A global expert count that does not split evenly fails fast.
        Expectation: An indivisible layout raises ValueError.
        """
        with self.assertRaises(ValueError):
            fixed_expert_indices(4, 1, 6, local_expert_count=4, device=torch.device("cpu"))


class TestBalancedRouter(unittest.TestCase):
    """The router wrapper around the real sigmoid-group adapter."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gathers_real_gate_weights_for_replacement_destinations(self) -> None:
        """Feature: fixed MoE routing.
        Description: Weights and gradients follow the experts selected by the fixed assignment.
        Expectation: Weights and gradients match the experts dispatched by the fixed plan.
        """
        tokens, per_token, total, local = 6, 2, 8, 2
        module = _FakeMoE(local, total)
        module.gate = torch.nn.Linear(3, total, bias=False)
        module.top_k = per_token
        module.norm_topk_prob = True
        module.routed_scaling_factor = 2.5
        states = torch.randn(tokens, 3, requires_grad=True)
        indices, weights = ep_compute._balanced_router(module, states)  # pylint: disable=protected-access
        expected_indices = torch.tensor([[0, 2], [4, 6], [1, 3], [5, 7], [0, 2], [4, 6]])
        scores = module.gate(states).float().sigmoid().gather(1, expected_indices)
        reference = scores / (scores.sum(-1, keepdim=True) + 1e-20) * 2.5
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, reference)
        actual = torch.autograd.grad(weights.square().sum(), (states, module.gate.weight), retain_graph=True)
        expected = torch.autograd.grad(reference.square().sum(), (states, module.gate.weight))
        for gradient, target in zip(actual, expected):
            torch.testing.assert_close(gradient, target)


class TestFactorySwitch(unittest.TestCase):
    """``fix_router`` selects the router the archetype is built with."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fix_router_selects_the_balanced_router(self) -> None:
        """Feature: fixed MoE routing.
        Description: On by switch, off by default; the stub captures what was passed.
        Expectation: The switch selects the balanced adapter and the default selects the real adapter.
        """
        captured = {}
        original = ep_compute.build_ep_compute

        # The leading underscores mark the positional slots as intentionally
        # unused (``ignored-argument-names``); the stub only needs the signature.
        def _stub(_module, _ep_mesh, *, router_fn, **_kwargs):  # pylint: disable=invalid-name
            """Record the router the factory selected."""
            captured["router_fn"] = router_fn
            return lambda *args, **kw: None

        ep_compute.build_ep_compute = _stub
        try:
            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=None, fix_router=True)
            self.assertIs(captured["router_fn"], ep_compute._balanced_router)  # pylint: disable=protected-access

            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None,
                ep_mesh=None)
            self.assertIs(captured["router_fn"],
                          routing_module.MOE_ROUTER_ADAPTERS["deepseekv3"])
        finally:
            ep_compute.build_ep_compute = original


class TestOverlapAndCapacitySwitches(unittest.TestCase):
    """``overlap_shared_expert`` / ``capacity_factor`` reach the EP factory.

    The 2-SN reference numbers are measured WITH the shared-expert overlap, and the
    family archetype is the entry point the recipe uses, so the archetype must forward
    both switches; before this was covered, the recipe silently built a non-overlapped
    plan (measured: +2.28 GB peak, +22% step time at GBS 256).
    """

    @staticmethod
    def _capture(**kwargs):
        """Build the archetype with ``build_ep_compute`` stubbed out, return its kwargs."""
        captured = {}
        original = ep_compute.build_ep_compute

        # The leading underscores mark the positional slots as intentionally
        # unused (``ignored-argument-names``); the stub only records the kwargs.
        def _stub(_module, _ep_mesh, **stub_kwargs):  # pylint: disable=invalid-name
            """Record the keyword arguments the archetype passed."""
            captured.clear()
            captured.update(stub_kwargs)
            return lambda *args, **kw: None

        ep_compute.build_ep_compute = _stub
        try:
            ep_compute.kimi_k26_ep_compute_fn(
                module=_FakeMoE(2, 8), mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=None,
                **kwargs)
        finally:
            ep_compute.build_ep_compute = original
        return captured

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_both_switches_default_off_and_are_forwarded(self) -> None:
        """Feature: fixed MoE routing.
        Description: Off by default; on when asked for, as the recipe asks for the overlap.
        Expectation: Overlap and capacity settings are forwarded without changing their defaults.
        """
        default = self._capture()
        self.assertIsNone(default["overlap_fn"])
        self.assertIsNone(default["capacity_factor"])

        asked = self._capture(overlap_shared_expert=True, capacity_factor=2.0)
        self.assertTrue(callable(asked["overlap_fn"]))
        self.assertEqual(asked["capacity_factor"], 2.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_combine_uses_the_overlapped_branch_when_it_is_given_one(self) -> None:
        """Feature: fixed MoE routing.
        Description: The overlap is real only if ``combine`` consumes the precomputed branch.
        Expectation: Combine consumes a supplied shared branch and otherwise computes it.
        """
        captured = self._capture(overlap_shared_expert=True)
        hidden = torch.randn(3, 4)
        shared = torch.randn(3, 4)
        routed = torch.randn(3, 4)

        class _Block:
            """Stand-in whose shared branch is recognisable in the output."""

            @staticmethod
            def shared_experts(states: torch.Tensor) -> torch.Tensor:
                """Return the marker branch offset by the states."""
                return shared + states

        computed = captured["combine"](_Block, hidden, routed)
        # Grouped the way the archetype groups it: the two forms differ only in float
        # addition order, which is not the contract under test.
        self.assertTrue(torch.allclose(computed, routed + (shared + hidden)))
        overlapped = captured["combine"](_Block, hidden, routed, shared)
        self.assertTrue(torch.equal(overlapped, routed + shared))


if __name__ == "__main__":
    unittest.main()
