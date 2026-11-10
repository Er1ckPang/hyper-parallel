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
"""Xing4 attention contracts for the shared MLA CP runtime."""

from __future__ import annotations

# pylint: disable=forbidden-backend-import

from functools import wraps
from typing import Any, Literal

import torch
from torch import nn

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLACPStrategy, apply_mla_rope
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper
from hyper_parallel.models.xing_4.modeling_xing4_0 import fusion_attention_with_logit_maxima


def _rotary(tensor: torch.Tensor, positions: tuple[torch.Tensor, torch.Tensor] | None,
            *, interleaved: bool = False) -> torch.Tensor:
    """Preserve official FP32 interleaved RoPE for the runtime's BSND tensors."""
    if positions is None:
        raise ValueError("Xing4 CP requires local RoPE position embeddings")
    return apply_mla_rope(tensor.float(), positions, interleaved=interleaved).to(tensor.dtype)


@inner_wrapper
def xing4_mla_cp_wrapper(target_module: nn.Module, mesh: Any, tp_mesh: Any,
                         cp_mesh: Any, ep_mesh: Any, *, strategy: MLACPStrategy = "expanded_ulysses",
                         overlap: bool = False, ulysses_degree: int | None = None,
                         kv_exchange: Literal["all_gather", "all_to_all_v"] = "all_gather") -> list:
    """Select the shared MLA CP plan through a ``when: cp`` forward rule.

    Ulysses is the default and composes with TP when the local head count is
    divisible by CP. At TP1, ``ulysses_degree=1`` selects the shared KV AllGather
    plan. The enclosing training model accepts only implicit causal sequences;
    its HF-generated local mask must not replace the CP runtime's global mask.

    Args:
        target_module: Optimized Xing4 MLA attention.
        mesh: Full framework mesh.
        tp_mesh: Tensor parallel mesh, or None.
        cp_mesh: Active context parallel mesh.
        ep_mesh: Expert parallel mesh, or None.
        strategy: Expanded QKV or expanded Q with latent KV exchange.
        overlap: Overlap input communication with independent projections.
        ulysses_degree: Head redistribution degree; None uses the CP size.
        kv_exchange: Full KV collection or selective causal-prefix distribution.

    Returns:
        Atomic forward rewrites from the shared MLA runtime.
    """
    requests = mla_cp_wrapper(
        target_module, mesh, tp_mesh, cp_mesh, ep_mesh, strategy=strategy, overlap=overlap,
        ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
        rotary_fn=_rotary, attention_interface=fusion_attention_with_logit_maxima,
    )
    original = requests[0].forward

    @wraps(original)
    def forward(hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Use the CP runtime's global causal mask instead of the HF local-shard mask."""
        del attention_mask
        return original(hidden_states, position_embeddings=position_embeddings, attention_mask=None, **kwargs)

    requests[0] = _ForwardRewriteRequest(target_module, forward, requests[0].companion_attrs)
    return requests
