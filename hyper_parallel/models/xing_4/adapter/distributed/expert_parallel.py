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
"""Bind Xing4 routing to the shared expert dispatcher and benchmark overrides."""

from collections.abc import Callable
from typing import Any

import torch

from hyper_parallel.distributed.expert_parallel.experts import SHARED_MERGE_MODES
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.expert_parallel.routing import resolve_moe_router
from hyper_parallel.distributed.recipe_spec import local_compute

__all__ = ["xing4_ep_compute_fn"]


@local_compute
def xing4_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                        ep_mesh: Any, fix_router: bool = False) -> Callable:
    """Bind local or EP experts through one forward rule.

    The model owns sequence statistics and hash-token alignment independently
    of this rule. EP requires it for expert dispatch; local experts use it only
    for the optional ``fix_router`` override.

    Args:
        module: Xing4 MoE layer with grouped experts.
        mesh: Full framework mesh.
        tp_mesh: Tensor parallel mesh, or None.
        cp_mesh: Context parallel mesh, or None.
        ep_mesh: Expert parallel mesh, or None for local experts.
        fix_router: Benchmark-only balanced destinations.

    Returns:
        The local-tensor forward that the rewriter binds to the module.
    """
    del mesh, tp_mesh, cp_mesh
    route = type(module).route
    executor = None if ep_mesh is None else build_ep_compute(
        module, ep_mesh, router_fn=route, archetype_key="xing4",
        expected_attrs=["gate", "experts", "shared_experts"], combine=SHARED_MERGE_MODES["none"],
        use_grouped_gemm=True, fix_router=fix_router)
    if ep_mesh is None:
        route = resolve_moe_router(route, fix_router)

    def forward(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute routed and shared experts through the common model path."""
        return module.dispatch(hidden_states, route, executor)

    return forward
