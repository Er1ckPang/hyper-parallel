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
"""Bind Xing4 Engram to the sequence and table meshes selected by the recipe."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, local_compute


@local_compute
def xing4_engram_compute_fn(
    *, module: EngramModule, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Bind the shared Engram sequence alignment and EP row lookup."""
    del mesh
    if not isinstance(module, EngramModule):
        raise TypeError("Xing4 parallel Engram requires the EngramModule replacement")
    ep_rank = 0 if ep_mesh is None else ep_mesh.get_local_rank("ep")
    ep_size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    if module.padded_num_embeddings % ep_size:
        raise ValueError("Xing4 Engram padded table rows must be divisible by the EP degree")
    coordinates = {
        "ep_rank": ep_rank,
        "ep_size": ep_size,
        "ep_group": None if ep_mesh is None else ep_mesh.get_group("ep"),
        "cp_rank": 0 if cp_mesh is None else cp_mesh.get_local_rank(),
        "cp_size": 1 if cp_mesh is None else cp_mesh.size(),
        "cp_group": None if cp_mesh is None else cp_mesh.get_group(),
        "tp_rank": 0 if tp_mesh is None else tp_mesh.get_local_rank(),
        "tp_size": 1 if tp_mesh is None else tp_mesh.size(),
    }

    def forward(
        module: EngramModule, hidden_states: torch.Tensor, input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None, token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Engram with the local token IDs and residual streams."""
        return module.parallel_forward(hidden_states, input_ids, segment_starts, token_mask, **coordinates)

    return forward


def with_engram_parallelism(setup: DistributedSetup) -> DistributedSetup:
    """Declare Engram placement and communication for a parallel Xing4 build."""
    mesh = setup.mesh_context
    overrides = dict(setup.plan_overrides or {})
    key = "model.layers.*.engram"
    spec = overrides.get(key, ModuleShardingSpec())
    replicas = {"tp": Replicate()}
    params = {
        "embed.weight": {"ep": Shard(0)} if mesh.ep_size > 1 else replicas,
        "wkv.weight": replicas,
        "q_weight": replicas,
        "k_weight": replicas,
    }
    overrides[key] = replace(
        spec, params={**params, **(spec.params or {})},
        region_dispatch=False if spec.region_dispatch is None else spec.region_dispatch,
        local_compute_fn=spec.local_compute_fn or xing4_engram_compute_fn,
    )
    return replace(setup, plan_overrides=overrides)
