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
"""CPU contracts for fixed expert assignment and differentiable score gathering."""

import unittest

import torch

from hyper_parallel.components.functional import fixed_expert_indices, fixed_topk_routing

from tests.common.mark_utils import arg_mark


class TestFixedRouter(unittest.TestCase):
    """Check assignment properties against independent load and gradient expectations."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_mindformers_expert_order(self) -> None:
        """Feature: fixed MoE routing.
        Description: The default ordering follows the pynative expert round robin.
        Expectation: Indices match the explicit MindFormers round-robin example.
        """
        expected = torch.tensor([[0, 1], [2, 3], [0, 1]])
        actual = fixed_expert_indices(3, 2, 4, device="cpu")
        torch.testing.assert_close(actual, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rank_order_compatibility(self) -> None:
        """Feature: fixed MoE routing.
        Description: The EP variant retains interleaved contiguous expert ownership.
        Expectation: Indices preserve the historical EP destination-first order.
        """
        expected = torch.tensor([[0, 2], [4, 6], [1, 3], [5, 7], [0, 2]])
        actual = fixed_expert_indices(5, 2, 8, device="cpu", local_expert_count=2)
        torch.testing.assert_close(actual, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_load_bounds_and_distinct_experts(self) -> None:
        """Feature: fixed MoE routing.
        Description: Remainders, small batches, full top-k and empty batches stay valid.
        Expectation: Expert and rank loads differ by at most one with unique per-token experts.
        """
        for experts, local in ((1, 1), (6, 1), (6, 2), (8, 4), (8, 8)):
            for top_k in (1, experts):
                for tokens in (0, 1, 3, 13):
                    with self.subTest(experts=experts, local=local, top_k=top_k, tokens=tokens):
                        indices = fixed_expert_indices(
                            tokens, top_k, experts, device="cpu", local_expert_count=local,
                        )
                        self.assertEqual(tuple(indices.shape), (tokens, top_k))
                        expert_load = torch.bincount(indices.flatten(), minlength=experts)
                        self.assertLessEqual(int(expert_load.max() - expert_load.min()), 1)
                        rank_load = torch.bincount(indices.flatten() // local, minlength=experts // local)
                        self.assertLessEqual(int(rank_load.max() - rank_load.min()), 1)
                        ordered = indices.sort(dim=-1).values
                        self.assertTrue(bool((ordered[:, 1:] != ordered[:, :-1]).all()))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_gathers_weights_and_gradients_for_replacement_experts(self) -> None:
        """Feature: fixed MoE routing.
        Description: A score's gradient is attached to its routed expert, not the old top-k.
        Expectation: Weights and score gradients match the fixed expert IDs.
        """
        scores = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
        original = torch.full((3, 2), 3, dtype=torch.int32)
        indices, weights = fixed_topk_routing(scores, original)
        torch.testing.assert_close(weights, torch.tensor([[0., 1.], [6., 7.], [8., 9.]], dtype=torch.float64))
        weights.sum().backward()
        torch.testing.assert_close(
            scores.grad, torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.], [1., 1., 0., 0.]], dtype=torch.float64),
        )
        self.assertEqual(indices.dtype, torch.int64)
        torch.testing.assert_close(original, torch.full((3, 2), 3, dtype=torch.int32))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_preserves_weight_dtype_and_empty_autograd(self) -> None:
        """Feature: fixed MoE routing.
        Description: The helper neither casts scores nor detaches empty score tensors.
        Expectation: Empty tensors retain score dtype and produce empty gradients.
        """
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                scores = torch.empty(0, 4, dtype=dtype, requires_grad=True)
                indices, weights = fixed_topk_routing(scores, torch.empty(0, 2, dtype=torch.int64))
                self.assertEqual(weights.dtype, dtype)
                self.assertEqual(tuple(indices.shape), (0, 2))
                weights.sum().backward()
                self.assertEqual(tuple(scores.grad.shape), (0, 4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_invalid_expert_layout(self) -> None:
        """Feature: fixed MoE routing.
        Description: Invalid top-k, token and local counts fail with descriptive errors.
        Expectation: Invalid layout counts raise ValueError.
        """
        for tokens, top_k, experts, local in ((-1, 1, 4, 1), (1, 0, 4, 1), (1, 5, 4, 1),
                                             (1, 1, 0, 1), (1, 1, 4, 0), (1, 1, 6, 4)):
            with self.subTest(shape=(tokens, top_k, experts, local)), self.assertRaises(ValueError):
                fixed_expert_indices(tokens, top_k, experts, device="cpu", local_expert_count=local)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_invalid_score_contract(self) -> None:
        """Feature: fixed MoE routing.
        Description: Shape, dtype and device mismatches cannot reach a gather kernel.
        Expectation: Invalid shapes, dtypes and devices raise ValueError.
        """
        invalid = (
            (torch.ones(2, 4, 1), torch.zeros(2, 2, dtype=torch.int64)),
            (torch.ones(2, 4), torch.zeros(3, 2, dtype=torch.int64)),
            (torch.ones(2, 4, dtype=torch.int64), torch.zeros(2, 2, dtype=torch.int64)),
            (torch.ones(2, 4), torch.zeros(2, 2)),
            (torch.ones(2, 4), torch.empty(2, 2, device="meta", dtype=torch.int64)),
        )
        for scores, indices in invalid:
            with self.subTest(scores=scores.shape, indices=indices.shape), self.assertRaises(ValueError):
                fixed_topk_routing(scores, indices)
