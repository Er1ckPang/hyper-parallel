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
"""Use native Hyper routing, EP and grouped experts inside the source L-0 model.

The surrounding MindSpeed model/trainer is shared to hold MLA, initialization,
auxiliary losses, balancing, distributed optimizer and data order constant.
Expert parameters keep their original names, storage and optimizer ownership.
"""

from functools import wraps
from types import MethodType
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch_npu
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.transformer.moe import moe_layer, moe_utils

from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.components.modules.moe import TokenChoiceTopKRouter
from hyper_parallel.components.modules.terrace_router import TerraceRouteConfig, terrace_select
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.expert_parallel.expert_parallel import AllToAllTokenDispatcher


def hyper_moe_forward(layer: torch.nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
    """Run dynamic routing and native Hyper EP without replacing parameters.

    Args:
        layer: Source MoE layer retaining its expert parameters and router.
        hidden: Input token states in the source model's layout.

    Returns:
        MoE output states and the absent output bias.
    """
    if parallel_state.get_expert_tensor_parallel_world_size() != 1:
        raise ValueError("The L-0 adapter requires expert TP=1")
    if layer.config.add_bias_linear:
        raise ValueError("The L-0 adapter requires bias-free experts")
    if not hasattr(layer, "hyper_ep_mesh"):
        layer.hyper_ep_mesh = DeviceMesh.from_group(
            parallel_state.get_expert_model_parallel_group(), "npu", mesh_dim_names=("ep",),
        )
    probs, routing_map = layer.router(hidden)
    states = hidden.reshape(-1, hidden.shape[-1])
    indices = routing_map.to(torch.int32).topk(layer.config.moe_router_topk, dim=-1).indices
    gates = probs.gather(1, indices).to(states.dtype).contiguous()
    counts = routing_map.sum(0)
    packed, mapping = torch_npu.npu_moe_token_permute(states, indices)
    # NPU permute returns the inverse permutation: original slot -> packed
    # row. Keep probabilities paired with exactly the same expert-token rows.
    packed_gates = torch.zeros_like(gates.reshape(-1)).scatter(0, mapping.long(), gates.reshape(-1))
    local_input, local_counts, local_gates, context = AllToAllTokenDispatcher.dispatch(
        layer.experts, (packed, counts, packed_gates), layer.hyper_ep_mesh,
    )
    weight1 = layer.experts.weight1.view(layer.num_local_experts, states.shape[-1], -1)
    weight2 = layer.experts.weight2.view(layer.num_local_experts, -1, states.shape[-1])
    local_output = npu_grouped_swiglu(
        local_input, weight1.transpose(1, 2), weight2.transpose(1, 2), local_counts, routing_weights=local_gates,
    )
    handle = AllToAllTokenDispatcher.combine_start(local_output, layer.hyper_ep_mesh, context)
    shared = layer.shared_experts(hidden) if layer.shared_experts is not None else None
    combined = AllToAllTokenDispatcher.combine_wait(handle)
    result = torch_npu.npu_moe_token_unpermute(
        combined, mapping, probs=torch.ones_like(gates, dtype=torch.float32),
    ).reshape_as(hidden)
    if shared is not None:
        result = result + shared
    return result, None


def mega_moe_forward(layer: torch.nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
    """Run the same live router through parameterless native MegaMoE.

    Args:
        layer: Source MoE layer containing the native MegaMoE executor.
        hidden: Input token states in the source model's layout.

    Returns:
        MoE output states and the absent output bias.
    """
    probs, routing_map = layer.router(hidden)
    states = hidden.reshape(-1, hidden.shape[-1])
    indices = routing_map.to(torch.int32).topk(layer.config.moe_router_topk, dim=-1).indices
    gates = probs.gather(1, indices).to(states.dtype).contiguous()
    weight1 = layer.experts.weight1.view(layer.num_local_experts, states.shape[-1], -1)
    weight2 = layer.experts.weight2.view(layer.num_local_experts, -1, states.shape[-1])
    output = layer.mega_executor(states, indices, gates, tokens_per_expert=routing_map.sum(0),
                                 expert_weights=(weight1, weight2)).reshape_as(hidden)
    if layer.shared_experts is not None:
        output = output + layer.shared_experts(hidden)
    return output, None


def install_hyper_adapter(arm: str) -> dict[str, Any]:
    """Install an explicit Hyper arm after MindSpeed and source routing hooks.

    Args:
        arm: ``hyper_terrace`` or ``hyper_global``. The caller separately loads
            the original Terrace hooks for its reference arm.

    Returns:
        Live counters, checked by the worker after training.
    """
    if arm not in ("hyper_terrace", "hyper_global", "mega_terrace", "mega_global"):
        raise ValueError(f"Unknown Hyper arm: {arm}")
    counters = {"routing_calls": 0, "moe_calls": 0, "layers": 0}
    native_routers = {}

    def _select(scores, topk, num_tokens, num_experts, num_groups, group_topk):
        if scores.shape != (num_tokens, num_experts):
            raise ValueError("Unexpected live selection-score shape")
        counters["routing_calls"] += 1
        if arm.endswith("global"):
            if (num_groups, group_topk) != (1, 1):
                raise ValueError("Native global top-k must receive one unrestricted group")
            key = (num_experts, topk)
            if key not in native_routers:
                with torch.random.fork_rng(devices=[]):
                    native_routers[key] = TokenChoiceTopKRouter(1, num_experts, topk)
            values, indices, _ = native_routers[key].select_experts(scores)
            return values, indices
        config = TerraceRouteConfig(
            num_experts, num_groups, topk, group_topk, group_score_pool=topk // group_topk,
        )
        indices, _ = terrace_select(scores, config)
        return scores.gather(1, indices), indices

    moe_utils.group_limited_topk = _select
    original_init = moe_layer.MoELayer.__init__

    @wraps(original_init)
    def _initialize(layer, *args, **kwargs):
        original_init(layer, *args, **kwargs)
        counters["layers"] += 1
        if arm.startswith("mega"):
            # Native MegaMoE is optional for the plain Hyper EP arm.
            from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=import-outside-toplevel
            layer.mega_executor = MegaMoeExperts(
                local_num_tokens=get_args().seq_length * get_args().micro_batch_size,
                hidden_size=layer.config.hidden_size, intermediate_size=layer.config.moe_ffn_hidden_size,
                num_experts=layer.config.num_moe_experts, top_k=layer.config.moe_router_topk,
                ep_size=parallel_state.get_expert_model_parallel_world_size(),
                ep_group=parallel_state.get_expert_model_parallel_group(), create_parameters=False,
                initial_capacity_factor=float(parallel_state.get_expert_model_parallel_world_size()),
            )

        def _forward(instance, hidden):
            counters["moe_calls"] += 1
            if arm.startswith("mega"):
                return mega_moe_forward(instance, hidden)
            return hyper_moe_forward(instance, hidden)

        layer.forward = MethodType(_forward, layer)

    moe_layer.MoELayer.__init__ = _initialize
    return counters
