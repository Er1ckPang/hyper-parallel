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
"""Fixed-router integration with the local MoE building blocks."""

import unittest

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules.moe import MoE, TokenChoiceTopKRouter

from tests.common.mark_utils import arg_mark


class TestTokenChoiceTopKRouter(unittest.TestCase):
    """Check fixed selection, unchanged normal selection, and end-to-end gradients."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fixed_router_uses_unbiased_scores_and_actual_counts(self) -> None:
        """Feature: fixed MoE routing.
        Description: Bias and group masks cannot leak into fixed combine weights.
        Expectation: Weights match unbiased fixed scores and counts match the new IDs.
        """
        for score_func in ("sigmoid", "softmax"):
            with self.subTest(score_func=score_func):
                router = TokenChoiceTopKRouter(
                    3, 4, 2, score_func=score_func, num_expert_groups=2,
                    num_limited_groups=1, route_scale=2.5, fix_router=True,
                )
                states = torch.randn(3, 3)
                bias = torch.tensor([100., 0., 0., 0.])
                weights, indices, counts = router(states, bias)
                expected_indices = torch.tensor([[0, 1], [2, 3], [0, 1]])
                logits = router.gate(states).float()
                scores = logits.sigmoid() if score_func == "sigmoid" else logits.softmax(-1)
                torch.testing.assert_close(indices, expected_indices)
                torch.testing.assert_close(weights, (scores * 2.5).gather(1, expected_indices))
                torch.testing.assert_close(counts, torch.tensor([2, 2, 1, 1]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_and_explicit_false_preserve_normal_routing(self) -> None:
        """Feature: fixed MoE routing.
        Description: Disabled routing matches a direct biased top-k computation.
        Expectation: Default and disabled outputs match the learned top-k reference.
        """
        router = TokenChoiceTopKRouter(3, 4, 2)
        states = torch.randn(5, 3, requires_grad=True)
        bias = torch.tensor([0., 1., 0., -1.])
        weights, indices, counts = router(states, bias)
        scores = router.gate(states).float().sigmoid()
        expected_indices = (scores + bias).topk(2, dim=-1).indices
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, scores.gather(1, expected_indices))
        torch.testing.assert_close(counts, torch.bincount(expected_indices.flatten(), minlength=4))
        router.fix_router = False
        torch.testing.assert_close(router(states, bias)[0], weights)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_variable_and_empty_batches_are_deterministic(self) -> None:
        """Feature: fixed MoE routing.
        Description: Assignment restarts every call, including after an empty microbatch.
        Expectation: The same local token count gives the same assignment on every call.
        """
        router = TokenChoiceTopKRouter(3, 4, 2, fix_router=True)
        for tokens in (3, 0, 1, 3):
            weights, indices, counts = router(torch.randn(tokens, 3))
            self.assertEqual(tuple(weights.shape), (tokens, 2))
            self.assertEqual(int(counts.sum()), tokens * 2)
            if tokens == 3:
                torch.testing.assert_close(indices, torch.tensor([[0, 1], [2, 3], [0, 1]]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_moe_forward_backward_matches_explicit_expert_sum(self) -> None:
        """Feature: fixed MoE routing.
        Description: Fixed routing reaches MoE through router_kwargs and trains the used weights.
        Expectation: Outputs, gate gradients and expert gradients match explicit experts.
        """
        for score_before in (False, True):
            with self.subTest(score_before=score_before):
                moe = MoE(3, 5, 4, top_k=2, score_before_experts=score_before, router_kwargs={"fix_router": True})
                states = torch.randn(1, 3, 3, requires_grad=True)
                output = moe(states)
                flat_states = states.reshape(3, 3)
                scores = moe.router.gate(flat_states).float().sigmoid()
                expected_rows = []
                for token, expert_ids in enumerate(((0, 1), (2, 3), (0, 1))):
                    row = torch.zeros(3)
                    for expert in expert_ids:
                        expert_input = flat_states[token]
                        if score_before:
                            expert_input = expert_input * scores[token, expert]
                        hidden = expert_input @ moe.experts.w1[expert].T
                        up = expert_input @ moe.experts.w3[expert].T
                        expert_output = (torch.nn.functional.silu(hidden) * up) @ moe.experts.w2[expert].T
                        row = row + (expert_output if score_before else expert_output * scores[token, expert])
                    expected_rows.append(row)
                expected = torch.stack(expected_rows).reshape_as(output)
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(moe.tokens_per_expert, torch.tensor([2., 2., 1., 1.]))
                parameters = (states, moe.router.gate.weight, moe.experts.w1, moe.experts.w2, moe.experts.w3)
                actual_grads = torch.autograd.grad(output.square().sum(), parameters, retain_graph=True)
                expected_grads = torch.autograd.grad(expected.square().sum(), parameters)
                for actual, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, reference)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_checkpoint_recomputation_preserves_gate_gradient(self) -> None:
        """Feature: fixed MoE routing.
        Description: Recomputation needs no cached route or host-side state.
        Expectation: Checkpointed weights and gate gradients match eager execution.
        """
        router = TokenChoiceTopKRouter(3, 4, 2, fix_router=True)
        states = torch.randn(3, 3, requires_grad=True)
        plain = router(states)[0]
        recomputed = checkpoint(lambda value: router(value)[0], states, use_reentrant=False)
        torch.testing.assert_close(recomputed, plain)
        expected = torch.autograd.grad(plain.sum(), (states, router.gate.weight), retain_graph=True)
        actual = torch.autograd.grad(recomputed.sum(), (states, router.gate.weight))
        for gradient, reference in zip(actual, expected):
            torch.testing.assert_close(gradient, reference)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_invalid_switch_and_topk(self) -> None:
        """Feature: fixed MoE routing.
        Description: Configuration mistakes fail at router construction.
        Expectation: Invalid switch types and excessive top-k raise ValueError.
        """
        with self.assertRaises(ValueError):
            TokenChoiceTopKRouter(3, 4, 2, fix_router="false")
        with self.assertRaises(ValueError):
            TokenChoiceTopKRouter(3, 4, 5)
