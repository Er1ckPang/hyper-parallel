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
"""Register the Xing4 model and its adapter.

Attention context parallelism follows the CP mesh. Expert parallelism uses
``plan_overrides``; the optimizer is selected by the ``optimizer`` target.
Set ``model._target_`` to
``hyper_parallel.models.xing_4.modeling_xing4_0.build_model`` to bind CP sequence
objectives and TP loss-parallel AUX scaling, without recipe ``method_replacements``.
The model builder also binds TP/CP statistics for every backbone and MTP router.
``xing4_ep_compute_fn`` selects EP dispatch or local ``fix_router`` independently.
See ``examples/training_demo/xing4/train_single.yaml`` for the plan; Dense MLPs must not match the MoE rule.
With TP, prefer ``loss_parallel: true`` to avoid full-vocabulary logits gathers.
"""

from collections.abc import Mapping
from functools import partial
from importlib import import_module
from typing import Any

from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsModule
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.adapter_spec import ModelAdapterSpec, RecomputePolicy
from hyper_parallel.models.registry import register_custom_model, register_model_adapter
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.models.xing_4.engram_config import engram_enabled


def _replacements(model: Any, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Select mesh-aware MLA and routed-expert members at their module boundaries."""
    requests = []
    if context["cp_size"] > 1:
        # The training classes are loaded only after model construction.
        from hyper_parallel.models.xing_4.adapter.distributed.context_parallel import (  # pylint: disable=C0415
            build_xing4_mla_cp,
        )
        from hyper_parallel.models.xing_4.modeling_xing4_0 import Xing4MLAAttention  # pylint: disable=C0415

        requests.append(ModuleReplacementRequest(
            match=("*.self_attn",), target=build_xing4_mla_cp, module_type=Xing4MLAAttention,
        ))
    if engram_enabled(model.config) and any(isinstance(module, GroupedExpertsModule) for module in model.modules()):
        requests.append(ModuleReplacementRequest(
            match=("*.mlp.experts",), target=GroupedExperts, module_type=GroupedExpertsModule,
        ))
    return tuple(requests)


def build_recompute_policy() -> RecomputePolicy:
    """Expose whole decoder and MTP blocks; layer selection remains a recipe choice."""
    return RecomputePolicy(region_patterns=("model.layers.*", "mtp.layers.*"))


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Use the declared MTP depths instead of inferring nested decoder units."""
    return () if model.mtp is None else ("mtp",)


def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Wrap each MTP depth and separate EP-owned Engram tables from dense projections."""
    mtp_units = (() if model.mtp is None else
                 tuple(f"mtp.layers.{name}" for name, _ in model.mtp.layers.named_children()))
    engram_units = tuple(
        fqn for fqn, _ in model.named_modules() if fqn.endswith((".engram.embed", ".engram.wkv"))
    )
    return engram_units + mtp_units


def get_fsdp_execution_order(model: Any, module_fqns: tuple[str, ...]) -> tuple[str, ...]:
    """Place Engram children next to the backbone layer that executes them."""
    del model
    selected = set(module_fqns)
    child_suffixes = ("engram.embed", "engram.wkv", "mlp.experts", "mtp_block.mlp.experts")
    child_units = {
        f"{parent}.{suffix}"
        for parent in module_fqns for suffix in child_suffixes
    } & selected
    order = []
    for fqn in module_fqns:
        if fqn in child_units:
            continue
        order.append(fqn)
        for suffix in child_suffixes:
            child = f"{fqn}.{suffix}"
            if child in selected:
                order.append(child)
    order.extend(fqn for fqn in module_fqns if fqn not in order)
    return tuple(order)


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Declare Xing4 MLA roles before the generic naming rules.

    The mHC, stream-collapse head and MTP fusion parameters are declared
    replicated by the recipe's ``plan_overrides``.

    Returns:
        Name patterns and their parameter roles.
    """
    return [
        (["linear_qkv", "engram"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
    ]


register_custom_model(
    "Xing4ForCausalLM",
    "hyper_parallel.models.xing_4.modeling_xing4_0",
    "Xing4ForCausalLM",
)

XING4_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Xing4ForCausalLM",
    model_type="xing4_0",
    replacements=_replacements,
    expert_parallel=partial(import_module, "hyper_parallel.models.xing_4.adapter.distributed.expert_parallel"),
    recompute=build_recompute_policy,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_execution_order=get_fsdp_execution_order,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
)
register_model_adapter(XING4_ADAPTER_SPEC)

__all__ = ["XING4_ADAPTER_SPEC"]
