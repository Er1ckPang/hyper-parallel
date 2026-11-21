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

from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.adapter_spec import ModelAdapterSpec, RecomputePolicy
from hyper_parallel.models.registry import register_custom_model, register_model_adapter
from hyper_parallel.models.replacement import ModuleReplacementRequest


def _replacements(model: Any, context: Mapping[str, Any]) -> tuple[ModuleReplacementRequest, ...]:
    """Apply the shared CP implementation to MLA in both the backbone and MTP."""
    del model
    if context["cp_size"] == 1:
        return ()
    # Model registration precedes construction of the NPU training classes.
    from hyper_parallel.models.xing_4.adapter.distributed.context_parallel import build_xing4_mla_cp
    from hyper_parallel.models.xing_4.modeling_xing4_0 import Xing4MLAAttention

    return (ModuleReplacementRequest(match=("*.self_attn",), target=build_xing4_mla_cp,
                                     module_type=Xing4MLAAttention),)


def build_recompute_policy() -> RecomputePolicy:
    """Expose whole decoder and MTP blocks; layer selection remains a recipe choice."""
    return RecomputePolicy(region_patterns=("model.layers.*", "mtp.layers.*"))


def get_fsdp_excluded_subtrees(model: Any) -> tuple[str, ...]:
    """Use the declared MTP depths instead of inferring nested decoder units."""
    return () if model.mtp is None else ("mtp",)


def get_fsdp_wrap_modules(model: Any) -> tuple[str, ...]:
    """Keep each MTP depth's parameters under the module that calls its decoder."""
    return (() if model.mtp is None else
            tuple(f"mtp.layers.{name}" for name, _ in model.mtp.layers.named_children()))


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Declare Xing4 MLA roles before the generic naming rules.

    The mHC, stream-collapse head and MTP fusion parameters are declared
    replicated by the recipe's ``plan_overrides``.

    Returns:
        Name patterns and their parameter roles.
    """
    return [
        (["linear_qkv"], ParamRole.REPLICATED),
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
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
)
register_model_adapter(XING4_ADAPTER_SPEC)

__all__ = ["XING4_ADAPTER_SPEC"]
