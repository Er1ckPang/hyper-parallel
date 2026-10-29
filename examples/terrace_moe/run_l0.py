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
"""Build or execute a deterministic 8-die source-L-0 / Hyper comparison arm.

Defaults to printing the plan. This launcher deliberately imports no framework.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--msllm-root", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--mindspeed-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("terrace", "hyper_terrace", "hyper_global",
                                         "mega_terrace", "mega_global"), required=True)
    parser.add_argument("--model", choices=("l0", "l1"), default="l0")
    parser.add_argument("--dispatcher", choices=("vendor", "ta2a"), default="vendor")
    parser.add_argument("--rpn", type=int)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--port", type=int, default=29711)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--nic")
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--gbs", type=int, default=8)
    parser.add_argument("--device-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--mock-smoke", action="store_true", help="At most two debugging steps; excluded from results")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def _build_plan(args: argparse.Namespace) -> dict:
    if args.dispatcher == "ta2a" and args.arm != "terrace":
        raise ValueError("The T-A2A dispatcher is a source reference, not a native Hyper/Mega dispatcher")
    spec = importlib.util.spec_from_file_location(
        "terrace_reproduce_launcher", args.source_root / "train/reproduce/launch.py",
    )
    source_launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_launcher)
    source_argv = [
        "--model", args.model, "--profile", "performance", "--router",
        "global_topk" if args.arm.endswith("global") else "full",
        "--steps", str(args.steps), "--gbs", str(args.gbs), "--mbs", "1", "--recompute", "none",
        "--msllm-root", str(args.msllm_root), "--megatron-root", str(args.megatron_root),
        "--mindspeed-root", str(args.mindspeed_root), "--output", str(args.output),
        "--master-port", str(args.port), "--seed", "1",
        "--nnodes", str(args.nnodes), "--nproc-per-node", str(args.nproc_per_node),
        "--node-rank", str(args.node_rank), "--master-addr", args.master_addr, "--ep", str(args.ep),
        "--dispatcher", args.dispatcher,
    ]
    if args.rpn is not None:
        source_argv.extend(["--rpn", str(args.rpn)])
    if args.nic:
        source_argv.extend(["--nic", args.nic])
    if args.mock_smoke:
        if args.steps > 2:
            raise ValueError("Mock data is restricted to at most two debugging steps")
    else:
        source_argv.extend([
            "--data-path", str((args.data_root / "fineweb_edu_subset_text_document").resolve()),
            "--tokenizer", str((args.data_root / "tokenizer-qwen25").resolve()),
        ])
    source_args = source_launcher.parser().parse_args(source_argv)
    plan = source_launcher.build_plan(source_args)
    worker = Path(__file__).with_name("l0_worker.py").resolve()
    command = plan["command"]
    source_worker = next(index for index, value in enumerate(command) if value.endswith("/train/reproduce/worker.py"))
    command[source_worker] = str(worker)
    # core_v0.12.1 enables NaN checking by default and only exposes its
    # negative flag. Preserve and validate the behavior, not the old spelling.
    command.remove("--check-for-nan-in-loss-and-grad")
    command.extend(["--moe-router-score-function", "sigmoid", "--moe-router-dtype", "fp32",
                    "--npu-deterministic", "--log-params-norm"])
    plan["expected_args"].update(moe_router_score_function="sigmoid", moe_router_dtype="fp32",
                                 npu_deterministic=True, log_params_norm=True, check_for_nan_in_loss_and_grad=True)
    plan["arm"] = args.arm
    plan["data_identity"] = ({"mock": True, "files": {}} if args.mock_smoke else
                             json.loads((args.data_root / "identity.json").read_text()))
    plan["environment"].update({
        "ASCEND_RT_VISIBLE_DEVICES": args.device_ids, "HCCL_DETERMINISTIC": "True",
        "CLOSE_MATMUL_K_SHIFT": "1", "PYTHONHASHSEED": "1", "OMP_NUM_THREADS": "8",
        "HCCL_EXEC_TIMEOUT": "600", "HCCL_CONNECT_TIMEOUT": "180",
        "HCCL_IF_BASE_PORT": "61000", "TOKENIZERS_PARALLELISM": "false",
        "L0_ARM": args.arm, "TERRACE_SOURCE": str(args.source_root.resolve()),
        "L0_OUTPUT": str(args.output.resolve()),
        "PYTHONPATH": os.pathsep.join(str(path.resolve()) for path in (
            args.msllm_root, args.megatron_root, args.mindspeed_root, args.source_root)),
    })
    return plan


def _validate_inputs(args: argparse.Namespace, plan: dict) -> Path:
    package = importlib.util.find_spec("hyper_parallel")
    expected_root = Path(__file__).resolve().parents[2]
    if package is None or not Path(package.origin).resolve().is_relative_to(expected_root):
        raise RuntimeError("Editable HyperParallel installation must point to this checkout")
    for name, identity in plan["data_identity"]["files"].items():
        if not name.endswith((".bin", ".idx")):
            continue
        data_path = args.data_root / name
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if data_path.stat().st_size != identity["bytes"] or digest != identity["sha256"]:
            raise RuntimeError(f"Input data fingerprint mismatch: {data_path}")
    return expected_root


def _execute_plan(args: argparse.Namespace, plan: dict) -> None:
    expected_root = _validate_inputs(args, plan)
    command = plan["command"]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output.resolve() / "manifest.json"
    plan["experiment_source_sha256"] = {
        str(path.relative_to(expected_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    manifest.write_text(json.dumps(plan, indent=2) + "\n")
    (args.output / "command.sh").write_text(shlex.join(command) + "\n")
    environment = os.environ.copy()
    environment.update(plan["environment"])
    environment["TM_REPRO_MANIFEST"] = str(manifest)
    with (args.output / "train.log").open("w") as log:
        with subprocess.Popen(command, cwd=plan["cwd"], env=environment,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
            (args.output / "pid.json").write_text(json.dumps({
                "pid": process.pid, "pgid": process.pid, "command": command, "cwd": plan["cwd"],
            }, indent=2) + "\n")
            code = process.wait()
    (args.output / "exit.json").write_text(json.dumps({"returncode": code}) + "\n")
    if code:
        raise subprocess.CalledProcessError(code, command)
    local_ranks = range(args.node_rank * args.nproc_per_node, (args.node_rank + 1) * args.nproc_per_node)
    runtime = [json.loads((args.output / f"runtime-rank{rank}.json").read_text()) for rank in local_ranks]
    if any(row["status"] != "PASS_RUNTIME" or row["steps"] != args.steps for row in runtime):
        raise RuntimeError("Missing or incomplete per-rank runtime validation")
    print(f"PASS arm={args.arm} steps={args.steps} local_ranks={args.nproc_per_node}", flush=True)


def main() -> None:
    """Derive a common model plan and optionally launch its measured arm."""
    args = _parse_args()
    plan = _build_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.run:
        _execute_plan(args, plan)


if __name__ == "__main__":
    main()
