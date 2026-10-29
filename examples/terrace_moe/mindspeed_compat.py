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
"""Common source-stack compatibility fixes applied identically to every arm."""

import sys

from megatron.core.transformer.moe import moe_utils
from mindspeed.lite.ops.triton.sort_chunks_by_idx import (
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
)


def repair_fused_sort_registration() -> list[str]:
    """Connect the installed Ascend Triton sorter to early-imported aliases.

    The source hook loads moe_utils before the MindSpeed feature manager.
    That manager patches the extension module, while core_v0.12.1's cached
    aliases remain unset. Use the same installed fused implementation directly
    without a CUDA Transformer Engine dependency or a slower sorting fallback.
    """
    original = moe_utils.sort_chunks_by_idxs

    def _sort_chunks(inputs, split_sizes, sorted_indices, probs=None, fused=False):
        if not fused:
            return original(inputs, split_sizes, sorted_indices, probs=probs, fused=False)
        if probs is None:
            return moe_sort_chunks_by_index(inputs, split_sizes, sorted_indices), None
        return moe_sort_chunks_by_index_with_probs(inputs, probs, split_sizes, sorted_indices)

    changed = []
    for name, module in list(sys.modules.items()):
        if name.startswith(("megatron.", "mindspeed.")) and getattr(
                module, "sort_chunks_by_idxs", None) is original:
            module.sort_chunks_by_idxs = _sort_chunks
            changed.append(name)
    return changed
