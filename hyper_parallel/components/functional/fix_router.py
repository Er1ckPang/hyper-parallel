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
"""Deterministic expert assignments for MoE debugging and profiling."""

from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import


def fixed_expert_indices(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    *,
    device: torch.device | str,
    local_expert_count: int = 1,
) -> torch.Tensor:
    """Build a balanced assignment without reading tensor data on the host.

    Args:
        num_tokens: Number of local tokens, including zero.
        top_k: Distinct experts assigned to each token.
        num_experts: Global routed expert count.
        device: Device on which to build the indices.
        local_expert_count: Experts per EP destination. With the default of
            one, slot ``i`` selects ``i % num_experts``, matching MindFormers.
            A larger value interleaves destination ranks before local experts.

    Returns:
        Int64 indices of shape ``[num_tokens, top_k]``. Expert loads differ
        by at most one. Destination loads are equal when the slot count is
        divisible by the destination count; otherwise they differ by one.

    Raises:
        ValueError: If token, expert, top-k, or local expert counts are invalid.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be nonnegative, got {num_tokens}")
    if num_experts <= 0 or not 1 <= top_k <= num_experts:
        raise ValueError(f"require num_experts > 0 and 1 <= top_k <= num_experts, got {num_experts}, {top_k}")
    if local_expert_count <= 0 or num_experts % local_expert_count != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by a positive local_expert_count "
            f"({local_expert_count})"
        )
    destinations = num_experts // local_expert_count
    slots = torch.arange(num_tokens * top_k, dtype=torch.int64, device=device)
    indices = (slots % destinations) * local_expert_count + (slots // destinations) % local_expert_count
    return indices.reshape(num_tokens, top_k)


def fixed_topk_routing(
    scores: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    local_expert_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace expert choices and gather the corresponding unbiased scores.

    Apply this after normal top-k selection and before model-specific score
    normalization and scaling. The gate and top-k stay in the compute path;
    gradients flow through scores for the experts actually selected.
    This deliberately changes the model's routing and is for profiling only.

    Args:
        scores: Floating-point, unbiased scores of shape ``[T, E]``.
        selected_experts: Integer choices of shape ``[T, K]`` on the same
            device, used as the layout template for the replacement.
        local_expert_count: Experts per EP destination; one uses expert-order
            round robin, as in MindFormers pynative.

    Returns:
        ``(indices, weights)`` of shape ``[T, K]``. Indices are int64 and
        weights keep the score dtype, device, and autograd connection.

    Raises:
        ValueError: If input shapes, devices, dtypes, or expert counts differ
            from the contract.
    """
    if scores.ndim != 2 or selected_experts.ndim != 2 or scores.shape[0] != selected_experts.shape[0]:
        raise ValueError("scores [T, E] and selected_experts [T, K] must have the same token count")
    if not scores.is_floating_point() or selected_experts.dtype not in (torch.int32, torch.int64):
        raise ValueError("scores must be floating point and selected_experts must be int32 or int64")
    if scores.device != selected_experts.device:
        raise ValueError("scores and selected_experts must be on the same device")
    indices = fixed_expert_indices(
        scores.shape[0], selected_experts.shape[1], scores.shape[1],
        device=selected_experts.device, local_expert_count=local_expert_count,
    )
    # Keep the selection's layout rather than letting a standalone arange
    # discard it before gather, as in the MindFormers pynative implementation.
    indices = selected_experts.to(torch.int64) * 0 + indices
    return indices, scores.gather(1, indices)
