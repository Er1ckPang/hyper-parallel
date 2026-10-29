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
"""Instrument the exact source L-0 training entrypoint for paired experiments."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch_npu


def fingerprint(models: list[torch.nn.Module], output: Path, rank: int, stage: str = "initial") -> None:
    """Hash every parameter byte, including this rank's expert shard.

    Args:
        models: Local model partitions containing the parameters to hash.
        output: Directory for the per-rank parameter fingerprints.
        rank: Global rank identifying the local parameter shard.
        stage: Label distinguishing initial and final fingerprints.
    """
    state = {}
    for model in models:
        for name, parameter in model.named_parameters():
            raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
            state[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                           "sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
    (output / f"{stage}-rank{rank}.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def install_data_trace(models: list[torch.nn.Module], output: Path, rank: int, state: dict) -> None:
    """Hash actual model tokens and labels at every measured optimizer step.

    Args:
        models: Local model partitions whose inputs must be observed.
        output: Directory for the per-rank input trace.
        rank: Global rank identifying the input stream.
        state: Training state containing the step counter and trace handle.
    """
    state["trace_file"] = (output / f"batch-hashes-rank{rank}.jsonl").open("w", encoding="utf-8")

    def _capture(_module, inputs, kwargs):
        tokens = inputs[0] if inputs else kwargs.get("input_ids")
        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError("The paired L-0 experiment must observe actual token inputs")
        row = {"step": state["step"] + 1}
        for name, value in (("tokens", tokens), ("labels", kwargs.get("labels"))):
            if value is not None:
                raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                row[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
        state["trace_file"].write(json.dumps(row) + "\n")

    for model in models:
        model.register_forward_pre_hook(_capture, with_kwargs=True)


def parameter_probes(models: list[torch.nn.Module], output: Path, rank: int, step: int) -> None:
    """Save fixed parameter and full-precision gradient samples for diagnostics.

    Args:
        models: Local model partitions containing parameters and gradients.
        output: Directory for the diagnostic probe archive.
        rank: Global rank identifying the local parameter shard.
        step: Completed optimizer step used to label the archive.
    """
    probes = {}
    for model in models:
        for name, parameter in model.named_parameters():
            size = parameter.numel()
            count = min(96, size)
            indices = torch.arange(count, device=parameter.device, dtype=torch.int64) * (size - 1) // max(count - 1, 1)
            probes[name + ".parameter"] = parameter.detach().reshape(-1)[indices].float().cpu().numpy()
            grad = getattr(parameter, "main_grad", parameter.grad)
            if grad is not None:
                probes[name + ".gradient"] = grad.detach().reshape(-1)[indices].float().cpu().numpy()
    np.savez(output / f"probes-step{step}-rank{rank}.npz", **probes)


def instrument_training(output: Path, rank: int, plan: dict) -> dict:
    """Record optimizer-step wall time and unrounded framework metrics.

    Args:
        output: Directory for timing, metrics, hashes and probe artifacts.
        rank: Global rank participating in the measured training run.
        plan: Frozen experiment plan containing expected runtime arguments.

    Returns:
        Mutable counters and resources shared by the installed training hooks.
    """
    # MindSpeed is an optional dependency specific to this example.
    from mindspeed_llm.training import training  # pylint: disable=import-outside-toplevel

    original_step = training.train_step
    original_log = training.training_log
    state = {"step": 0, "seconds": 0.0, "executors": []}

    def _step(*args, **kwargs):
        models = args[2]
        if state["step"] == 0:
            install_data_trace(models, output, rank, state)
            executors = [module.mega_executor for model in models for module in model.modules()
                         if hasattr(module, "mega_executor")]
            if executors:
                # Optional native MegaMoE is only loaded by a Mega arm.
                from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=import-outside-toplevel
                MegaMoeExperts.share_execution_resources(executors)
                state["executors"] = executors
            fingerprint(models, output, rank)
        torch.npu.synchronize()
        started = time.perf_counter()
        result = original_step(*args, **kwargs)
        torch.npu.synchronize()
        seconds = torch.tensor(time.perf_counter() - started, device="npu", dtype=torch.float32)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        state["seconds"] = seconds.item()
        state["step"] += 1
        if state["step"] in (1, plan["expected_args"]["train_iters"]):
            parameter_probes(models, output, rank, state["step"])
        if state["step"] == plan["expected_args"]["train_iters"]:
            fingerprint(models, output, rank, stage="final")
        return result

    def _log(*args, **kwargs):
        result = original_log(*args, **kwargs)
        if rank == 0:
            row = {
                "step": int(args[4]), "step_seconds": state["seconds"],
                "tokens_per_second": plan["expected_args"]["global_batch_size"] * 4096 / state["seconds"],
                "learning_rate": args[2], "skipped": int(args[7]),
                "grad_norm": float(args[8]) if args[8] is not None else None,
                "params_norm": float(args[9]) if args[9] is not None else None,
                "losses": {key: float(value) for key, value in args[0].items()},
            }
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        return result

    training.train_step = _step
    training.training_log = _log
    return state


def _validate_dispatcher(plan: dict, ta2a: dict) -> None:
    """Require the selected communication path to remain active.

    Args:
        plan: Frozen experiment plan containing the selected dispatcher.
        ta2a: Source dispatcher activity and fallback counters.
    """
    if plan["dispatcher"] == "ta2a":
        if ta2a["fallback"] or ta2a["active"] < 1 or ta2a["fastpath"] != ta2a["active"]:
            raise RuntimeError(f"Source T-A2A quota path did not remain active: {ta2a}")
    elif ta2a["active"] or ta2a["fallback"]:
        raise RuntimeError(f"A vendor/Hyper arm unexpectedly entered T-A2A: {ta2a}")


def main() -> None:
    """Load the source hook explicitly, then select a measured implementation."""
    plan = json.loads(Path(os.environ["TM_REPRO_MANIFEST"]).read_text(encoding="utf-8"))
    output = Path(os.environ["L0_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    source = Path(os.environ["TERRACE_SOURCE"])
    spec = importlib.util.spec_from_file_location(
        "usercustomize", source / "train/mindspeed/terrace_patches/usercustomize.py",
    )
    hook = importlib.util.module_from_spec(spec)
    sys.modules["usercustomize"] = hook
    spec.loader.exec_module(hook)
    # Load the optional source trainer before replacing its registered methods.
    from mindspeed_llm import megatron_adaptor  # pylint: disable=unused-import,import-outside-toplevel
    from megatron.core.transformer.moe import moe_utils  # pylint: disable=import-outside-toplevel
    from megatron.training import get_args  # pylint: disable=import-outside-toplevel
    from mindspeed_compat import repair_fused_sort_registration  # pylint: disable=import-outside-toplevel

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.use_deterministic_algorithms(True)
    compatibility_modules = repair_fused_sort_registration()
    counters = None
    if os.environ["L0_ARM"] != "terrace":
        # Hyper's example adapter needs the optional MindSpeed installation.
        from mindspeed_adapter import install_hyper_adapter  # pylint: disable=import-outside-toplevel
        counters = install_hyper_adapter(os.environ["L0_ARM"])
    training_state = instrument_training(output, rank, plan)
    sys.argv[0] = str(Path(plan["cwd"]) / "pretrain_gpt.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
    for executor in training_state["executors"]:
        executor.close()
    training_state["trace_file"].close()
    args = get_args()
    actual = {key: getattr(args, key, None) for key in plan["expected_args"]}
    mismatches = {key: (value, actual[key]) for key, value in plan["expected_args"].items()
                  if value != actual[key]}
    if mismatches:
        raise RuntimeError(f"Runtime configuration mismatch: {mismatches}")
    ta2a = dict(hook._TA2A_STATS)  # pylint: disable=protected-access
    _validate_dispatcher(plan, ta2a)
    expected_layers = plan["expected_args"]["num_layers"] - plan["expected_args"]["first_k_dense_replace"]
    if counters is not None and (counters["layers"] != expected_layers or counters["moe_calls"] < expected_layers
                                 or counters["routing_calls"] < expected_layers):
        raise RuntimeError(f"Hyper path was not live in all L-0 layers: {counters}")
    observed = {
        "status": "PASS_RUNTIME", "rank": rank, "steps": training_state["step"],
        "actual_args": actual, "hyper_counters": counters,
        "ta2a_counters": ta2a,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
        "routing_precision": "fp32 projection/sigmoid/normalization; bf16 transport probabilities",
        "group_topk_implementation": str(moe_utils.group_limited_topk),
        "fused_sort_compatibility_modules": compatibility_modules,
    }
    (output / f"runtime-rank{rank}.json").write_text(json.dumps(observed, indent=2) + "\n")
    print(f"PASS_RUNTIME rank={rank} steps={training_state['step']}", flush=True)


if __name__ == "__main__":
    main()
