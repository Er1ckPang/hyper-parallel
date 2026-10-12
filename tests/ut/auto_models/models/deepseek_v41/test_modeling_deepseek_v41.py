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
"""DeepSeek-V4.1 learned-router fixed assignment contracts."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter

from tests.common.mark_utils import arg_mark


class TestDeepseekV41FixedRouter(unittest.TestCase):
    """Check each scoring function against explicit gathered-score references."""

    @staticmethod
    def _config(**overrides) -> SimpleNamespace:
        """Make a minimal text/image router configuration."""
        values = {
            "hidden_size": 3, "num_local_experts": 4, "num_experts_per_tok": 2,
            "scoring_func": "sigmoid", "routed_scaling_factor": 2.5, "v41_vision_enabled": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fixed_router_regathers_all_score_functions(self) -> None:
        """Feature: fixed MoE routing.
        Description: Text/image correction biases do not affect the fixed combine weights.
        Expectation: All scoring functions produce the specified fixed weights and gradients.
        """
        for scoring in ("sigmoid", "softmax", "sqrtsoftplus"):
            with self.subTest(scoring=scoring):
                router = DeepseekV41TopKRouter(self._config(
                    scoring_func=scoring, moe_router_force_expert_balance=True,
                ))
                torch.nn.init.normal_(router.weight)
                with torch.no_grad():
                    router.bias.copy_(torch.tensor([100., 100., 0., 0.]))
                    router.bias_vl.copy_(torch.tensor([0., 0., 100., 100.]))
                states = torch.randn(1, 3, 3, requires_grad=True)
                logits, weights, indices = router(states, image_mask=torch.tensor([[False, True, False]]))
                expected_indices = torch.tensor([[0, 1], [2, 3], [0, 1]])
                if scoring == "sigmoid":
                    scores = logits.sigmoid()
                elif scoring == "softmax":
                    scores = logits.softmax(-1)
                else:
                    scores = torch.nn.functional.softplus(logits).sqrt()  # pylint: disable=not-callable
                selected = scores.gather(1, expected_indices)
                reference = selected / (selected.sum(-1, keepdim=True) + 1e-20) * 2.5
                torch.testing.assert_close(indices, expected_indices)
                torch.testing.assert_close(weights, reference)
                actual = torch.autograd.grad(weights.square().sum(), (states, router.weight), retain_graph=True)
                expected = torch.autograd.grad(reference.square().sum(), (states, router.weight))
                for gradient, target in zip(actual, expected):
                    torch.testing.assert_close(gradient, target)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_router_uses_text_and_image_biases(self) -> None:
        """Feature: fixed MoE routing.
        Description: The new switch defaults off, preserving learned modality routing.
        Expectation: The disabled router selects experts according to modality biases.
        """
        router = DeepseekV41TopKRouter(self._config())
        with torch.no_grad():
            router.weight.zero_()
            router.bias.copy_(torch.tensor([100., 100., 0., 0.]))
            router.bias_vl.copy_(torch.tensor([0., 0., 100., 100.]))
        _, _, indices = router(torch.zeros(1, 2, 3), image_mask=torch.tensor([[False, True]]))
        torch.testing.assert_close(indices.sort(-1).values, torch.tensor([[0, 1], [2, 3]]))
        self.assertFalse(router.fix_router)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_string_switch(self) -> None:
        """Feature: fixed MoE routing.
        Description: A string false must never accidentally enable fixed routing.
        Expectation: A string switch raises ValueError.
        """
        with self.assertRaises(ValueError):
            DeepseekV41TopKRouter(self._config(moe_router_force_expert_balance="false"))
