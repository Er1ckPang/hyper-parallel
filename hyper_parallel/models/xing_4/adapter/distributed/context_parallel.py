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

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules.attention.multi_latent_attention import MultiLatentAttention
from hyper_parallel.models.replacement import module_replacement


@module_replacement
def build_xing4_mla_cp(*, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
                       strategy: str = "expanded_ulysses", overlap: bool = False,
                       ulysses_degree: int | None = None, kv_exchange: str = "all_gather") -> nn.Module:
    """Select the shared MLA CP member and preserve Xing4's RoPE and QK clipping."""
    if context["cp_size"] == 1:
        return module
    attention = MultiLatentAttention(
        module=module, module_fqn=module_fqn, context=context, strategy=strategy,
        overlap=overlap, ulysses_degree=ulysses_degree, kv_exchange=kv_exchange,
    )
    attention.rotary_fn = module.apply_position
    attention.record_logit_maxima = module.record_logit_maxima
    attention.max_logits = module.max_logits
    return attention
