# TerraceMoE routing and source-network validation

Native routing has no dependency on the TerraceMoE repository:

```python
from hyper_parallel.components.modules import TerraceRouteConfig, TerraceRouter
from hyper_parallel.components.modules.moe import MoE

moe = MoE(dim=1024, hidden_dim=512, num_experts=64, top_k=4,
          score_before_experts=False, use_grouped_mm=True)
moe.router = TerraceRouter(1024, TerraceRouteConfig(64, 16, 4, 2))
```

`TerraceRouter` returns normalized unbiased weights, selected expert IDs and local
expert counts. Selection uses the caller's balancing bias. It never updates bias
inside forward; the owning MoE updates it after accumulating global loads.
`terrace_select` accepts precomputed selection scores when a trainer owns scoring,
normalization, auxiliary losses and bias updates.

## Same-network experiment

The L-0/L-1 experiment uses the original MindSpeed network and trainer, preserving
MLA, QK-Norm, YaRN, shared experts, initialization, data and distributed AdamW.
It replaces expert selection and execution with native Hyper components. Install
the source TerraceMoE repository and matching MindSpeed/Megatron dependencies
separately, and install this checkout editable.

```bash
python -m pip install --no-deps --no-build-isolation -e . -e /path/to/TerraceMoE
python examples/terrace_moe/check_reference.py --output routing-parity.json
source /usr/local/Ascend/cann/set_env.sh
```

For MegaMoE, build and activate this checkout's private native payload before
starting Python:

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910_93 --jobs 16
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

Build a plan first; add `--run` to execute. Output directories must be new.

```bash
python examples/terrace_moe/run_l0.py \
  --source-root /path/to/TerraceMoE --msllm-root /path/to/MindSpeed-LLM \
  --megatron-root /path/to/Megatron-LM --mindspeed-root /path/to/MindSpeed \
  --data-root /path/to/frozen-data --arm hyper_terrace --steps 100 \
  --output /path/to/results/hyper-terrace
```

Available arms are `terrace`, `hyper_terrace`, `mega_terrace`, `hyper_global` and
`mega_global`. Default execution is 8 local dies, EP8, MBS1, GBS8, sequence 4096,
seed 1 and deterministic NPU/HCCL operation. `--model l1` and explicit multi-node
arguments prepare larger jobs; these require independent topology validation.
`--mock-smoke` permits at most two debugging steps and must not supply quality results.

The data directory contains `fineweb_edu_subset_text_document.bin/.idx`,
`tokenizer-qwen25/`, and `identity.json` with immutable source identity and file
hashes. The launcher verifies the indexed data and editable import location before
launch. The worker saves initial/final parameter hashes, actual input hashes,
unrounded per-step loss/norm/time metrics and per-rank runtime evidence.

## Numerical convention and scope

The source fused permutation rounds routing weights to BF16 and applies them
before the expert down projection. The plain Hyper adapter preserves this through
`npu_grouped_swiglu(..., routing_weights=...)`. Existing users omitting this optional
argument retain their original behavior.

MegaMoE applies weights after expert outputs. With bias-free experts these are
algebraically equivalent, while BF16 rounding can produce different training
trajectories. Compare Mega against both the source and an output-weighted oracle;
do not use plain EP's numerical result as proof for Mega.

Each optimizer-step timer includes forward, backward, gradient synchronization,
input loading and optimizer, and records the maximum across ranks. Parameter-hash
diagnostics and framework logging are outside that timer. Use separate warm-up
steps and independent runs for throughput; profiling must be off.

T-A2A's multi-level communication protocol is not fused into Hyper EP or MegaMoE
by this adapter. Native Hyper routing and within-domain EP/Mega measurements do
not establish cross-domain T-A2A speedups.
