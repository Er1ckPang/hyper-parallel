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
"""Compare native routing IDs, gates and derivatives against installed Terrace."""

import argparse
import json
from pathlib import Path

import torch  # pylint: disable=forbidden-backend-import
from terrace.routing import TRouteConfig, t_route

from hyper_parallel.components.modules.terrace_router import TerraceRouteConfig, terrace_route


def main() -> None:
    """Run independent source comparisons without accelerator dependencies."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1197)
    rows = []
    for experts, groups, top_k, top_groups in ((64, 16, 4, 2), (16, 4, 4, 2),
                                               (128, 8, 8, 4), (24, 4, 6, 2)):
        for mode in ("full", "group_limited", "global_topk", "quota_only"):
            if mode == "quota_only" and top_k % groups:
                continue
            affinity = torch.rand(127, experts, dtype=torch.float64) + .01
            bias = torch.randn(experts, dtype=torch.float64) * .1
            source_affinity = affinity.clone().requires_grad_()
            hyper_affinity = affinity.clone().requires_grad_()
            source = t_route(source_affinity, bias, TRouteConfig(experts, groups, top_k, top_groups), mode)
            hyper = terrace_route(hyper_affinity, bias,
                                  TerraceRouteConfig(experts, groups, top_k, top_groups), mode)
            torch.testing.assert_close(hyper[0], source[0], rtol=0, atol=0)
            torch.testing.assert_close(hyper[1], source[1], rtol=0, atol=0)
            coefficient = torch.randn_like(source[1])
            (source[1] * coefficient).sum().backward()
            (hyper[1] * coefficient).sum().backward()
            torch.testing.assert_close(hyper_affinity.grad, source_affinity.grad, rtol=0, atol=0)
            rows.append({"experts": experts, "groups": groups, "top_k": top_k,
                         "top_groups": top_groups, "mode": mode, "exact_ids": True,
                         "max_gate_difference": float((hyper[1] - source[1]).abs().max()),
                         "max_gradient_difference": float(
                             (hyper_affinity.grad - source_affinity.grad).abs().max())})
    args.output.write_text(json.dumps({"status": "PASS", "cases": rows}, indent=2) + "\n")
    print(f"PASS: {len(rows)} source routing / gate / gradient comparisons")


if __name__ == "__main__":
    main()
