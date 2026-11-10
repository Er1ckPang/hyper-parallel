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
"""Convert a saved MindFormers training checkpoint to native HP DCP offline."""

import argparse
import ast
import hashlib
import importlib
import json
import logging
import re
import shutil
import tempfile
import warnings
from copy import deepcopy
from dataclasses import replace
from functools import partial
from itertools import combinations
from math import prod
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open

from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer, initialize_optimizer_state
from hyper_parallel.components.checkpoint.conversion_ops import ConcatenateWithSections, Split
from hyper_parallel.components.checkpoint.load_groups import build_load_groups, convert_group, first_tensor
from hyper_parallel.components.checkpoint.weight_conversion import ConversionOps, WeightConverter, WeightRenaming
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader, FileSystemWriter
from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata, Metadata, MetadataIndex, TensorProperties, TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner, StandardSavePlanner
from hyper_parallel.core.distributed_checkpoint.storage import StorageInfo
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.data.batching import FixedBatchDataLoader, build_dataloader, calculate_num_micro_batches
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.parser import load_training_config

LOGGER = logging.getLogger(__name__)
_SLOTS = {
    "momentum_buffer": "muon_m.", "exp_avg": "adam_m.",
    "exp_avg_sq": "adam_v.", "max_exp_avg_sq": "max_exp_avg_sq.",
}
_MAIN = "_mixed_precision_optimizer.fp32_from_fp16_params."

_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64, "F16": torch.float16,
    "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
}


def _check_chunks(name: str, shape: tuple, chunks: list[ChunkStorageMetadata]) -> None:
    # The load planner copies intersections; it does not detect holes in a source checkpoint.
    for chunk in chunks:
        if (len(chunk.offsets) != len(shape) or len(chunk.sizes) != len(shape)
                or any(start < 0 or size < 0 or start + size > total
                       for start, size, total in zip(chunk.offsets, chunk.sizes, shape))):
            raise ValueError(f"Invalid source shard extent for {name}: {chunk}")
    overlap = any(all(max(a, b) < min(a + sa, b + sb)
                      for a, b, sa, sb in zip(left.offsets, right.offsets, left.sizes, right.sizes))
                  for left, right in combinations(chunks, 2))
    if overlap or sum(prod(chunk.sizes) for chunk in chunks) != prod(shape):
        raise ValueError(f"Source shards overlap or do not cover {name}, shape={shape}")


class MindFormersCheckpoint:
    """Expose full logical tensors without loading the model or importing MindSpore.

    Args:
        directory: A completed MF checkpoint iteration directory.
        metadata_path: Optional MF metadata.json exported with the source layout.
            Multi-rank checkpoints need offsets, not just TP/EP/PP degrees.

    Native metadata selects one stored replica for each rectangular shard.
    Without metadata, only files declaring a single storage rank are accepted.
    """

    def __init__(self, directory: str | Path, metadata_path: str | Path | None = None) -> None:
        self.directory = Path(directory)
        self.headers: dict[str, dict[str, tuple[tuple, torch.dtype]]] = {}
        for path in sorted(self.directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as reader:
                self.headers[path.name] = {
                    name: (tuple(reader.get_slice(name).get_shape()), _DTYPES[reader.get_slice(name).get_dtype()])
                    for name in reader.keys()
                }
        if not self.headers:
            raise ValueError(f"No MF safetensors in {self.directory}")
        path = Path(metadata_path) if metadata_path else self.directory / "metadata.json"
        if metadata_path is not None and not path.is_file():
            raise FileNotFoundError(path)
        self.metadata = self._from_metadata(json.loads(path.read_text())) if path.exists() else self._unsharded()
        common = self.directory / "common.json"
        self.common = json.loads(common.read_text()) if common.exists() else {}
        self.reader = FileSystemReader(self.directory)
        self.reader.configure_reader(self.metadata, is_coordinator=True, rank=0)

    def _from_metadata(self, source: dict[str, Any]) -> Metadata:
        storage = {ast.literal_eval(key): replicas for key, replicas in source["storage_data"].items()}
        tensors, locations = {}, {}
        for name, item in source["state_dict_metadata"].items():
            shape = tuple(item["global_shape"])
            fragments = tuple(item.get("axis_fragmentations", ()))
            # MF Layout checkpoints encode a flattened slice ordinal; other
            # exports may already carry per-axis element coordinates.
            ordinal = (len(fragments) == len(shape) and all(
                len(part["global_offset"]) == 1 and 0 <= part["global_offset"][0] < prod(fragments)
                for part in item["chunk"]))
            if ordinal and any(total % parts for total, parts in zip(shape, fragments)):
                raise ValueError(f"MF slice ordinals require an evenly partitioned global shape: {name}")
            chunks, dtypes = [], set()
            for part in item["chunk"]:
                saved_offset, size = tuple(part["global_offset"]), tuple(part["local_shape"])
                replicas = storage[(name, saved_offset)]
                offset = (tuple((saved_offset[0] // prod(fragments[axis + 1:]) % parts) * (total // parts)
                                for axis, (total, parts) in enumerate(zip(shape, fragments)))
                          if ordinal else saved_offset)
                file_name = next((replica["file_name"] for replica in replicas
                                  if replica["file_name"] in self.headers), None)
                if file_name is None:
                    raise FileNotFoundError(f"Missing source shard for {name} at {offset}")
                actual_shape, dtype = self.headers[file_name][name]
                if actual_shape != size:
                    raise ValueError(f"MF metadata and stored shape differ for {name} in {file_name}")
                chunk = ChunkStorageMetadata(offsets=offset, sizes=size)
                if chunk not in chunks:
                    chunks.append(chunk)
                dtypes.add(dtype)
                locations[MetadataIndex(name, offset)] = StorageInfo(file_name, 0, -1, tensor_key=name)
            _check_chunks(name, shape, chunks)
            if len(dtypes) != 1:
                raise ValueError(f"Source shards have different dtypes for {name}")
            tensors[name] = TensorStorageMetadata(TensorProperties(str(dtypes.pop())), shape, chunks)
        return Metadata(state_dict_metadata=tensors, storage_data=locations)

    def _unsharded(self) -> Metadata:
        tensors, locations = {}, {}
        for file_name, entries in self.headers.items():
            # MF's filename records the storage-rank count even if only a subset was copied.
            count = Path(file_name).stem.rsplit("-", 1)[-1]
            if count.isdecimal() and int(count) != 1:
                raise ValueError("A sharded MF checkpoint requires metadata.json with the source tensor layouts")
            for name, (shape, dtype) in entries.items():
                if name in tensors:
                    raise ValueError(f"Repeated tensor {name}; supply the MF shard metadata")
                offset = (0,) * len(shape)
                tensors[name] = TensorStorageMetadata(
                    TensorProperties(str(dtype)), shape, [ChunkStorageMetadata(offset, shape)])
                locations[MetadataIndex(name, offset)] = StorageInfo(file_name, 0, -1, tensor_key=name)
        return Metadata(state_dict_metadata=tensors, storage_data=locations)

    def keys(self) -> Any:
        """Return logical source names, including optimizer tensors."""
        return self.metadata.state_dict_metadata.keys()

    def empty(self, name: str, *, device: str = "meta") -> torch.Tensor:
        """Allocate a tensor with the stored global shape and dtype."""
        entry = self.metadata.state_dict_metadata[name]
        dtype = getattr(torch, entry.properties.dtype.removeprefix("torch."))
        return torch.empty(entry.size, dtype=dtype, device=device)

    def read(self, name: str) -> torch.Tensor:
        """Reconstruct one full tensor on CPU using the existing DCP load planner."""
        state = {name: self.empty(name, device="cpu")}
        planner = StandardLoadPlanner()
        planner.configure_planner(state, self.metadata, rank=0, is_coordinator=True, use_collectives=False)
        local = self.reader.optimize_local_plan(planner.build_local_plan())
        plan = self.reader.optimize_global_plan(planner.build_global_plan([local]))
        self.reader.execute_read(planner.finalize_plan(plan), planner)
        return state[name]


class DeinterleaveGateUp(ConversionOps):
    """Convert MF PyNative dense/shared FC1 rows to contiguous gate/up halves."""

    def convert(self, input_dict: dict, source_patterns: list[str], target_patterns: list[str],
                **kwargs: Any) -> dict[str, torch.Tensor]:
        """Apply the same coordinate permutation to weights and optimizer slots."""
        del kwargs
        tensor = first_tensor(input_dict[source_patterns[0]])
        return {source_patterns[0] if len(target_patterns) > 1 else target_patterns[0]:
                torch.cat((tensor[0::2], tensor[1::2]), dim=0)}


def _build_target(config: Any) -> tuple[Any, list, dict[str, float]]:
    """Build a full meta model and MF-to-HP transforms for offline conversion.

    Args:
        config: HP training configuration. Only model construction and module
            replacements are applied here; target meshes are applied at load time.

    Returns:
        Model, framework weight transforms, and frozen source constants omitted
        by fused modules. MF PyNative experts already use the grouped 3-D layout.
    """
    # Only target construction needs the model's optional runtime dependencies.
    implementation = importlib.import_module("hyper_parallel.models.xing_4.modeling_xing4_0")
    model = implementation.Xing4ForCausalLM.from_config(
        implementation.create_config(config.model.to_dict()["config"]))
    plan = compile_module_replacements(model, entries_to_module_replacements(config.plan_overrides))
    model, _ = apply_module_replacements(model, plan, weights_mapping=[], capture_checkpoint_metadata=False)
    targets = model.state_dict()
    rules, constants = [], {}

    def rename(source: str, target: str) -> None:
        rules.append(WeightRenaming(source_patterns=f"^{re.escape(source)}$", target_patterns=target))

    for source, target in (("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
                           ("decoder.final_layernorm.weight", "model.norm.weight"),
                           ("output_layer.weight", "lm_head.weight")):
        rename(source, target)
    layers = [(f"decoder.layers.{i}", f"model.layers.{i}") for i in range(len(model.model.layers))]
    for i in range(len(model.mtp.layers) if model.mtp is not None else 0):
        prefix = f"mtp.layers.{i}"
        layers.append((f"{prefix}.transformer_layer", f"{prefix}.mtp_block"))
        for field in ("eh_proj", "enorm", "hnorm", "final_layernorm"):
            target = "post_norm" if field == "final_layernorm" else field
            rename(f"{prefix}.{field}.weight", f"{prefix}.{target}.weight")
    fields = {
        "input_layernorm.weight": "input_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "self_attention.k_layernorm.weight": "self_attn.kv_a_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_a_layernorm.weight",
        "self_attention.linear_qkv.weight": "self_attn.linear_qkv.weight",
        "self_attention.linear_qb.weight": "self_attn.q_b_proj.weight",
        "self_attention.linear_kvb.weight": "self_attn.kv_b_proj.weight",
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.tid2eid": "mlp.gate.tid2eid",
        "mlp.expert_bias": "mlp.gate.e_score_correction_bias",
        "mlp.tokens_per_expert": "mlp.gate._token_counts",
        "mlp.experts.weight1": "mlp.experts.gate_up_proj",
        "mlp.experts.weight2": "mlp.experts.down_proj",
    }
    for source, target in layers:
        for old, new in fields.items():
            rename(f"{source}.{old}", f"{target}.{new}")
        for part in ("mlp", "mlp.shared_experts"):
            fused = f"{target}.{part}.linear_fc1.weight"
            gate = f"{target}.{part}.gate_proj.weight"
            if fused not in targets and gate not in targets:
                continue
            outputs = [fused] if fused in targets else [gate, f"{target}.{part}.up_proj.weight"]
            operations = [DeinterleaveGateUp()]
            if fused not in targets:
                operations.append(Split((targets[gate].shape[0], targets[gate].shape[0])))
            rules.append(WeightConverter(
                source_patterns=f"^{re.escape(source + '.' + part + '.linear_fc1.weight')}$",
                target_patterns=outputs, operations=operations))
            rename(f"{source}.{part}.linear_fc2.weight",
                   f"{target}.{part}.{'linear_fc2' if fused in targets else 'down_proj'}.weight")
        for part in ("attn_hc", "ffn_hc"):
            if f"{target}.{part}.hc_base" not in targets:
                continue
            rename(f"{source}.{part}.bias", f"{target}.{part}.hc_base")
            rename(f"{source}.{part}.mapping_proj.weight", f"{target}.{part}.hc_fn")
            rules.append(WeightConverter(
                source_patterns=[f"^{re.escape(source + '.' + part + '.alpha_' + suffix)}$"
                                 for suffix in ("pre", "post", "res")],
                target_patterns=f"{target}.{part}.hc_scale",
                operations=[ConcatenateWithSections((1, 1, 1))]))
            constants[f"{source}.{part}.rms_weight"] = 1.0
    return model, rules, constants


def _build_optimizer(model: Any, config: Any, step: int) -> Any:
    """Reuse configured optimizer groups and allocate native slots on meta.

    Args:
        model: Unsharded meta model after module replacement.
        config: Target HP training configuration, including optimizer semantics.
        step: Completed optimizer updates, distinct from batches consumed.

    Returns:
        The configured optimizer with a serializable FQN state skeleton.
    """
    raw = config.optimizer.target.build(model=model).get_optimizer()
    optimizer = Float16OptimizerWithFloat16Params(raw, model) if config.optimizer.fp32_main_params else raw
    # The public helper executes a dummy update for unknown optimizers; offline conversion must not.
    if any(not isinstance(leaf, (AdamW, Muon)) for leaf in raw.chained_optimizers):
        raise ValueError("MF slot conversion currently implements HP AdamW and Muon")
    if not initialize_optimizer_state(optimizer):
        raise ValueError("The configured optimizer did not produce its checkpoint state")
    for leaf in raw.chained_optimizers:
        for group in leaf.param_groups:
            group["step"] = step
    return optimizer


def _fixed_sample_state(config: Any, consumed_samples: int, data_path: str | None,
                        mesh: MeshContext) -> tuple[dict, int]:
    """Let the configured sampler and DataLoader serialize a global sample cursor."""
    tokenizer_target = config.dataset.model_assets.tokenizer
    tokenizer = tokenizer_target.build() if tokenizer_target is not None else None
    transform = config.dataset.data_transform.build(tokenizer=tokenizer) if config.dataset.data_transform else None
    options = {"data_path": data_path} if data_path is not None else {}
    datasets = config.dataset.build(tokenizer=tokenizer, transform=transform, training_config=config.training,
                                    mesh_context=mesh, sampler_type=config.dataloader.sampler_type, **options)
    if not isinstance(datasets, tuple) or len(datasets) != 3:
        datasets = (datasets, None, None)
    collator = config.dataloader.collate_fn.build(mesh_context=mesh) if config.dataloader.collate_fn else None
    loaders, samplers = build_dataloader(
        config.dataloader, datasets=datasets, collate_fn=collator, mesh_context=mesh,
        training_config=config.training, max_seq_len=getattr(transform, "max_seq_len", None), default_seed=42)
    loader, sampler = loaders[0], samplers[0]
    if not isinstance(loader, FixedBatchDataLoader) or sampler is None:
        raise ValueError("Supply --resume-state with native HP data state for streaming or dynamic batching")
    dataset_state = getattr(loader.dataset, "state_dict", None)
    if callable(dataset_state) and dataset_state():
        raise ValueError("A stateful dataset needs its actual native --resume-state, not a reconstructed cursor")
    accumulation = calculate_num_micro_batches(
        config.training.global_batch_size, config.training.micro_batch_size, mesh.dp_size)
    epoch_samples = len(loader) // accumulation * config.training.global_batch_size
    epoch, cursor = divmod(consumed_samples, epoch_samples)
    loader.set_epoch(epoch)
    sampler.load_state_dict({**sampler.state_dict(), "consumed_samples": cursor, "epoch": epoch})
    # Use the DataLoader's public snapshot format; sampler state alone is not a loader snapshot.
    return deepcopy(loader.state_dict()), epoch


def _has_dataset_state(value: Any) -> bool:
    """Detect worker/dataset snapshots that cannot be shared across DP lanes."""
    if isinstance(value, dict):
        return any((name in ("dataset_state", "_dataset_state", "_dataset_iter_state") and item is not None)
                   or _has_dataset_state(item) for name, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_dataset_state(item) for item in value)
    return False


def _build_train_state(config: Any, optimizer: Any, common: dict, supplied: dict,
                      *, scheduler_step: int, data_path: str | None, mesh: MeshContext) -> dict:
    """Create the Trainer's extra_state, accepting supplied native state overrides.

    Stateless MF LR schedules are reconstructed from the target's equivalent
    schedule and saved LR input index. Stateful schedules, streaming/random
    datasets and compatible RNG snapshots can be supplied without data replay.
    """
    extra = dict(supplied)
    if "global_step" in extra and int(extra["global_step"]) != int(common["global_step"]):
        raise ValueError("The supplied HP resume state must belong to the MF checkpoint's training step")
    step = int(extra["global_step"] if "global_step" in extra else common["global_step"])
    extra["global_step"] = step
    horizon = config.training.lr_scheduler_iters or config.training.train_iters
    if horizon is None:
        horizon = config.training.train_samples // config.training.global_batch_size
    scheduler = config.lr_scheduler.build(optimizer=optimizer, train_iters=horizon).get_lr_scheduler()
    if "lr_scheduler" in extra:
        scheduler.load_state_dict(extra["lr_scheduler"])
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
            for _ in range(scheduler_step):
                scheduler.step()
            if scheduler_step < 0:
                # An initial standalone fused-Adam checkpoint precedes LR index zero.
                warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step")
                for leaf in scheduler:
                    leaf.step(scheduler_step)
        extra["lr_scheduler"] = scheduler.state_dict()
    if "train_dataloader" not in extra:
        state, epoch = _fixed_sample_state(config, int(common["consumed_samples"]), data_path, mesh)
        extra["train_dataloader"] = state
        extra.setdefault("epoch", epoch)
    if "epoch" not in extra:
        raise ValueError("--resume-state must include the HP zero-based epoch with its DataLoader state")
    # MindSpore RNG bytes are not a Torch RNG state. Keep HP's configured runtime seeds.
    extra.setdefault("rng_state", {})
    return extra


def _read_converted(group: Any, name: str, model: Any, dtype: torch.dtype) -> torch.Tensor:
    # Transformers consumes collected loaders when converting; retain the reusable plan.
    tensor = first_tensor(convert_group(deepcopy(group), model)[name])
    converted = tensor.to(dtype=dtype).contiguous()
    if (tensor.is_floating_point() and not converted.is_floating_point()
            and not torch.equal(tensor, converted.to(tensor))):
        raise ValueError(f"Source counter cannot be represented exactly in {dtype}: {name}")
    return converted


def _tensor_loaders(source: MindFormersCheckpoint, model: Any, rules: list, targets: dict,
                    names: dict[str, str], used: set[str]) -> dict[str, Callable]:
    def loader(name: str, *, meta: bool) -> Callable:
        physical = names[name]
        used.add(physical)
        return lambda: source.empty(physical) if meta else source.read(physical)

    meta_groups, unexpected, _ = build_load_groups(
        model, names, targets, weights_mapping=rules, make_loader=partial(loader, meta=True))
    if unexpected:
        raise ValueError(f"Source tensors have no target mapping: {unexpected}")
    groups, _, _ = build_load_groups(
        model, names, targets, weights_mapping=rules, make_loader=partial(loader, meta=False))
    result = {}
    for meta_group, group in zip(meta_groups, groups):
        for name, value in convert_group(meta_group, model).items():
            tensor = first_tensor(value)
            if (isinstance(value, list) and len(value) != 1
                    or name in result or name not in targets or tensor.shape != targets[name].shape):
                raise ValueError(f"Ambiguous or incompatible tensor mapping: {name}, shape={tuple(tensor.shape)}")
            result[name] = partial(_read_converted, group, name, model, targets[name].dtype)
    if set(result) != set(targets):
        raise ValueError(f"Checkpoint is missing target tensors: {sorted(set(targets) - set(result))}")
    return result


class ConversionPlanner(StandardSavePlanner):
    """Resolve meta checkpoint tensors lazily through weight transforms."""

    def __init__(self, loaders: dict[str, Callable]) -> None:
        super().__init__()
        self.loaders = loaders

    def get_data(self, item: Any) -> Any:
        """Read a converted tensor only when its output shard is being written."""
        loader = self.loaders.get(item.index.fqn)
        return loader() if loader is not None else super().get_data(item)


def _make_loaders(source: MindFormersCheckpoint, model: Any, optimizer: Any,
                  rules: list, constants: dict) -> tuple[dict, dict[str, Callable]]:
    model_state = model.state_dict()
    payload = {"model": model_state}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    planner = StandardSavePlanner()
    planner.configure_planner(payload, is_coordinator=True)
    used, loaders = set(), {}
    prefixes = ("main_param.", *_SLOTS.values())
    model_names = {name: name for name in source.keys() if not name.startswith(prefixes) and name != "global_step"}
    for name, value in constants.items():
        if name in model_names:
            if not torch.all(source.read(name) == value).item():
                raise ValueError(f"The fused target omits {name}, but its source value is not {value}")
            model_names.pop(name)
            used.add(name)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    # FP32 HP parameters are also their optimizer's main weights. Keep the saved
    # FP32 source, rather than widening an already-rounded BF16 compute copy.
    main_names = {}
    for name in model_names:
        main = f"main_param.{name}"
        main_names[name] = main if main in source.keys() else name
    model_loaders = _tensor_loaders(source, model, rules, model_state, model_names, used)
    main_state = {name: torch.empty_like(tensor, dtype=torch.float32) if name in trainable else tensor
                  for name, tensor in model_state.items()}
    main_loaders = _tensor_loaders(source, model, rules, main_state, main_names, used)
    for name, tensor in model_state.items():
        main = name in trainable and tensor.dtype == torch.float32
        loaders[f"model.{name}"] = (main_loaders if main else model_loaders)[name]
    if optimizer is None:
        return payload, loaders
    for slot, prefix in _SLOTS.items():
        targets = {name.removeprefix("optimizer.state.").removesuffix(f".{slot}"): tensor
                   for name, tensor in planner.state_dict.items()
                   if name.startswith("optimizer.state.") and name.endswith(f".{slot}")}
        names = {name.removeprefix(prefix): name for name in source.keys() if name.startswith(prefix)}
        converted = _tensor_loaders(source, model, rules, targets, names, used)
        loaders.update({f"optimizer.state.{name}.{slot}": load for name, load in converted.items()})
    main_targets = {name.removeprefix(f"optimizer.{_MAIN}"): tensor
                    for name, tensor in planner.state_dict.items() if name.startswith(f"optimizer.{_MAIN}")}
    for name in main_targets:
        # Main tensors use the same mapping, with their own FP32 destination dtype.
        loaders[f"optimizer.{_MAIN}{name}"] = main_loaders[name]
    if any(name in trainable and tensor.dtype != torch.float32 and name not in main_targets
           for name, tensor in model_state.items()):
        raise ValueError("Low-precision target parameters need the HP FP32-main optimizer wrapper to resume MF")
    # A missing master for a low-precision trainable source loses information;
    # dtype alone cannot reconstruct it. Buffers and frozen parameters need none.
    slot_parameters = {name.removeprefix(prefix) for prefix in _SLOTS.values()
                       for name in source.keys() if name.startswith(prefix)}
    for name in slot_parameters:
        if f"main_param.{name}" not in source.keys() and source.empty(name).dtype != torch.float32:
            raise ValueError(f"Missing FP32 main weight for the low-precision source parameter {name}")
    unused = set(source.keys()) - used - {"global_step"}
    if unused:
        raise ValueError(f"Source training tensors would be lost: {sorted(unused)}")
    return payload, loaders


def _write_checkpoint(output: Path, payload: dict, loaders: dict[str, Callable], shard_bytes: int) -> None:
    planner = ConversionPlanner(loaders)
    planner.configure_planner(payload, is_coordinator=True)
    writer = FileSystemWriter(output)
    writer.configure_writer(is_coordinator=True, rank=0, use_collectives=True)
    local = writer.optimize_local_plan(planner.build_local_plan())
    plan, metadata = planner.build_global_plan([local])
    plan = planner.finalize_plan(writer.optimize_global_plan(plan))
    results, batch, size = [], [], 0
    for item in plan.items:
        item_bytes = item.tensor_storage_size() or 0
        if batch and size + item_bytes > shard_bytes:
            results.append(writer.execute_write(replace(plan, items=batch), planner))
            writer.configure_writer(is_coordinator=True, rank=len(results), use_collectives=True)
            batch, size = [], 0
        batch.append(item)
        size += item_bytes
    if batch:
        results.append(writer.execute_write(replace(plan, items=batch), planner))
    # Storage file ordinals are not training ranks. The one global metadata file
    # describes full tensor coordinates for the runtime load planner to reshard.
    writer.finalize_checkpoint(metadata, results)


def main() -> None:
    """Convert a native MF iteration using the target HP recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", "--load-dir", dest="load_dir", type=Path, required=True,
                        help="completed native MF iteration directory")
    parser.add_argument("--hp-config", type=Path, required=True, help="HP training YAML")
    parser.add_argument("--mf-optimizer", choices=("muon", "adamw", "fused-adamw"),
                        help="source optimizer; selects its saved update-counter convention")
    parser.add_argument("--dst", "--save-dir", dest="save_dir", type=Path, required=True, help="new HP DCP directory")
    parser.add_argument("--source-metadata", type=Path, help="MF shard metadata, if stored separately")
    parser.add_argument("--data-path", help="optional override of dataset.data_path")
    parser.add_argument("--resume-state", type=Path,
                        help="optional HP extra_state .pt or checkpoint directory with per-rank extra_state files")
    parser.add_argument("--weights-only", action="store_true", help="convert model tensors without optimizer or progress")
    parser.add_argument("--shard-size-mb", type=int, default=512,
                        help="output tensor batch budget; one tensor may exceed it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = MindFormersCheckpoint(args.load_dir, args.source_metadata)
    config = load_training_config(args.hp_config)
    if not args.weights_only and args.mf_optimizer is None:
        parser.error("--mf-optimizer is required for complete training-state conversion")
    if args.shard_size_mb <= 0:
        parser.error("--shard-size-mb must be positive")
    # Standalone fused AdamW stores a zero-based kernel index. Muon, including
    # its fused AdamW fallback, stores the completed-update count directly.
    scheduler_step = None if args.weights_only else int(source.read("global_step").item())
    step = None if args.weights_only else scheduler_step + int(args.mf_optimizer == "fused-adamw")
    old_dtype = torch.get_default_dtype()
    try:
        if config.model_init_dtype is not None:
            torch.set_default_dtype(getattr(torch, config.model_init_dtype))
        with torch.device("meta"):
            model, rules, constants = _build_target(config)
            optimizer = None if args.weights_only else _build_optimizer(model, config, step)
    finally:
        torch.set_default_dtype(old_dtype)
    extra_states = []
    rank_states = args.resume_state is not None and args.resume_state.is_dir()
    if not args.weights_only:
        paths = (sorted((args.resume_state / "extra_state").glob("extra_state_rank_*.pt"))
                 if rank_states else [args.resume_state])
        if rank_states and not paths:
            raise ValueError("--resume-state directory has no native per-rank extra_state files")
        if rank_states:
            ranks = [int(path.stem.removeprefix("extra_state_rank_")) for path in paths]
            if (sorted(ranks) != list(range(len(paths)))
                    or any(path.name != f"extra_state_rank_{rank}.pt" for path, rank in zip(paths, ranks))):
                raise ValueError("Native resume-state ranks must be unique and contiguous from rank 0")
        for path in paths:
            supplied = torch.load(path, map_location="cpu", weights_only=False) if path is not None else {}
            if rank_states and "train_dataloader" not in supplied:
                raise ValueError(f"Native per-rank resume state is missing train_dataloader: {path}")
            extra_states.append(_build_train_state(
                config, optimizer, source.common, supplied, scheduler_step=scheduler_step,
                data_path=args.data_path, mesh=MeshContext()))
        if not rank_states:
            common_state = extra_states[0]
            data_state = common_state["train_dataloader"]
            if (common_state["rng_state"] or "_index_sampler_state" not in data_state
                    or _has_dataset_state(data_state)):
                raise ValueError("Rank-local RNG/data need a native per-rank --resume-state directory")
    # Scheduler construction also populates the optimizer's initial_lr fields.
    payload, loaders = _make_loaders(source, model, optimizer, rules, constants)
    if extra_states and not rank_states:
        # Runtime loaders repartition the global cursor into target DP lanes.
        payload["extra_state"] = extra_states[0]
    args.save_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.save_dir.exists():
        raise FileExistsError(args.save_dir)
    # Model plus optimizer slots/main tensors are represented by meta storage.
    check = StandardSavePlanner()
    check.configure_planner(payload, is_coordinator=True)
    total = sum(item.tensor_storage_size() or 0 for item in check.build_local_plan().items)
    if shutil.disk_usage(args.save_dir.parent).free < total:
        raise OSError(f"Checkpoint needs at least {total} bytes of free output space")
    receipt = {"source": str(args.load_dir.resolve()), "source_common": source.common,
               "config_sha256": hashlib.sha256(args.hp_config.read_bytes()).hexdigest(),
               "mf_optimizer": args.mf_optimizer,
               "weights_only": args.weights_only,
               "training_state_layout": ("weights-only" if args.weights_only else
                                         "per-rank-supplied" if rank_states else "embedded-global"),
               "global_step": extra_states[0]["global_step"] if extra_states else None,
               "optimizer_step": step, "scheduler_step": scheduler_step,
               "tensor_bytes": total}
    with tempfile.TemporaryDirectory(prefix=f".{args.save_dir.name}-", dir=args.save_dir.parent) as temporary:
        stage = Path(temporary) / "checkpoint"
        _write_checkpoint(stage, payload, loaders, args.shard_size_mb * 1024 * 1024)
        for source_path, extra in zip(paths if rank_states and not args.weights_only else [], extra_states):
            # Keep the native checkpointer's per-rank file naming in one place.
            rank = int(source_path.stem.removeprefix("extra_state_rank_"))
            path = Path(DistributedCheckpointer._extra_state_path(stage, rank))  # pylint: disable=protected-access
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(extra, path)
        (stage / "conversion.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        stage.rename(args.save_dir)
    LOGGER.info("Converted training checkpoint: %s (training step %s, optimizer updates %s)",
                args.save_dir, receipt["global_step"], step)


if __name__ == "__main__":
    main()
