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
"""TerraceMoE group-quota routing, compatible with HyperParallel MoE experts.

Algorithm attribution: TerraceMoE ``terrace/routing.py`` (Apache-2.0).
Expert groups are contiguous logical expert-ID ranges. Physical placement must
map those ranges to the intended communication domains separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

__all__ = ["TerraceRouteConfig", "TerraceRouter", "terrace_route", "terrace_select"]

_MODES = ("full", "group_limited", "quota_only", "global_topk")


@dataclass(frozen=True)
class TerraceRouteConfig:
    """Geometry of contiguous expert groups and per-token selection.

    Args:
        num_experts: Total expert count.
        num_groups: Number of equally sized expert groups.
        top_k: Number of distinct experts selected per token.
        top_groups: Groups selected in full and group_limited modes.
        group_score_pool: Number of best biased affinities averaged per group.
    """

    num_experts: int
    num_groups: int
    top_k: int
    top_groups: int
    group_score_pool: int = 2

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_experts % self.num_groups:
            raise ValueError("num_experts must be divisible by num_groups")
        if self.top_groups > self.num_groups:
            raise ValueError("top_groups cannot exceed num_groups")
        if self.top_k % self.top_groups:
            raise ValueError("top_k must be divisible by top_groups")
        if self.top_k // self.top_groups > self.group_size:
            raise ValueError("per-group quota exceeds group size")

    @property
    def group_size(self) -> int:
        """Number of experts in each contiguous group."""
        return self.num_experts // self.num_groups


def terrace_route(
    affinity: torch.Tensor,
    expert_bias: torch.Tensor,
    config: TerraceRouteConfig,
    mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select experts with TerraceMoE semantics, without mutating state.

    Args:
        affinity: Positive affinities shaped ``[tokens, num_experts]``.
        expert_bias: Selection-only balancing bias shaped ``[num_experts]``.
        config: Expert-group geometry.
        mode: full, group_limited, quota_only, or global_topk.

    Returns:
        Expert IDs, normalized unbiased gates, and group IDs. In full mode,
        each token selects exactly top_groups groups and top_k/top_groups
        experts per group. Group loads are statistical, except when all groups
        are selected. Tie order follows topk, as in the source algorithm.
        All-zero affinities produce zero gates rather than NaNs.

    Raises:
        ValueError: If mode, geometry, or tensor metadata is invalid.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown routing mode {mode!r}; expected one of {_MODES}")
    if affinity.ndim != 2 or affinity.shape[1] != config.num_experts:
        raise ValueError("affinity must have shape [tokens, num_experts]")
    if not affinity.is_floating_point():
        raise ValueError("affinity must be floating point")
    if expert_bias.shape != (config.num_experts,) or expert_bias.device != affinity.device:
        raise ValueError("expert_bias must have shape [num_experts] and match affinity device")
    if mode == "quota_only" and config.top_k % config.num_groups:
        raise ValueError("quota_only requires top_k divisible by num_groups")
    indices, groups = terrace_select(affinity + expert_bias, config, mode)
    gates = affinity.gather(1, indices)
    gates = gates / gates.sum(-1, keepdim=True).clamp_min(torch.finfo(gates.dtype).tiny)
    return indices, gates, groups


def terrace_select(
    selection: torch.Tensor, config: TerraceRouteConfig, mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select expert and group IDs from already bias-corrected scores.

    Args:
        selection: Selection scores with shape ``[tokens, num_experts]``.
        config: Group geometry; set group_score_pool to the source convention.
        mode: Routing ablation mode.

    Returns:
        Expert IDs and group IDs. No gate normalization or state update is
        performed, so callers can preserve their existing precision and loss.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown routing mode {mode!r}")
    if selection.ndim != 2 or selection.shape[1] != config.num_experts:
        raise ValueError("selection must have shape [tokens, num_experts]")
    if mode == "quota_only" and config.top_k % config.num_groups:
        raise ValueError("quota_only requires top_k divisible by num_groups")
    tokens = selection.shape[0]
    width = config.group_size
    if mode == "global_topk":
        indices = selection.topk(config.top_k, dim=-1).indices
        groups = indices // width
    elif mode == "quota_only":
        local = selection.reshape(tokens, config.num_groups, width).topk(
            config.top_k // config.num_groups, dim=-1).indices
        groups = torch.arange(config.num_groups, device=selection.device).expand(tokens, -1)
        indices = (groups.unsqueeze(-1) * width + local).reshape(tokens, config.top_k)
    else:
        grouped = selection.reshape(tokens, config.num_groups, width)
        group_scores = grouped.topk(min(config.group_score_pool, width), dim=-1).values.mean(-1)
        groups = group_scores.topk(config.top_groups, dim=-1).indices
        candidates = grouped.gather(1, groups.unsqueeze(-1).expand(-1, -1, width))
        if mode == "full":
            local = candidates.topk(config.top_k // config.top_groups, dim=-1).indices
            indices = (groups.unsqueeze(-1) * width + local).reshape(tokens, config.top_k)
        else:
            local = candidates.reshape(tokens, config.top_groups * width).topk(config.top_k, dim=-1).indices
            base = groups.unsqueeze(-1).expand(-1, -1, width).reshape(tokens, config.top_groups * width)
            indices = base.gather(1, local) * width + local % width
    return indices, groups


class TerraceRouter(nn.Module):
    """T-Route router implementing the HyperParallel MoE router interface.

    Args:
        dim: Input hidden size.
        config: Expert-group geometry.
        mode: Routing ablation mode.

    Note:
        Returns ``(scores, indices, counts)``, like TokenChoiceTopKRouter.
        Supply the MoE owner's expert_bias; update it through the existing
        MoE load-balancing lifecycle after globally aggregating counts. Forward
        does not update bias, so activation recomputation is side-effect free.
        FP32 sigmoid and normalization follow HyperParallel's router precision.
    """

    def __init__(self, dim: int, config: TerraceRouteConfig, mode: str = "full") -> None:
        """Validate group geometry and initialize the affinity projection."""
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if mode not in _MODES:
            raise ValueError(f"unknown routing mode {mode!r}")
        if mode == "quota_only" and config.top_k % config.num_groups:
            raise ValueError("quota_only requires top_k divisible by num_groups")
        self.config = config
        self.mode = mode
        self.gate = nn.Linear(dim, config.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=dim ** -0.5)

    def forward(
        self, hidden: torch.Tensor, expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized gates, expert IDs, and the exact local histogram.

        Args:
            hidden: Token states shaped ``[tokens, dim]``.
            expert_bias: Optional selection-only bias; defaults to zero.
        """
        affinity = torch.sigmoid(self.gate(hidden).float())
        if expert_bias is None:
            expert_bias = affinity.new_zeros(self.config.num_experts)
        indices, gates, _ = terrace_route(affinity, expert_bias, self.config, self.mode)
        counts = torch.bincount(indices.reshape(-1), minlength=self.config.num_experts)
        return gates, indices, counts
