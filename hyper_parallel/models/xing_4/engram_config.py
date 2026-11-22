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
"""Translate Xing4 architecture fields into a device Engram layout."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from hyper_parallel.components.modules.engram.assets import build_scaled_engram_buckets


_ARCH_FIELDS = (
    "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
    "engram_vocab_size", "engram_n_heads", "engram_head_dim",
    "engram_pad_token_id", "engram_compressed_vocab_size",
)


def merge_engram_config(values: dict[str, Any]) -> None:
    """Import only Engram fields from an architecture JSON into recipe values."""
    path = values.pop("engram_config_path", None)
    if path is None:
        return
    source = _read_json(path)
    if "vocab_size" in source and source["vocab_size"] != values.get("vocab_size", 131072):
        raise ValueError("Xing4 and Engram architecture JSON must use the same vocab_size")
    missing = [name for name in _ARCH_FIELDS if name not in source]
    if missing:
        raise ValueError(f"Missing Engram architecture fields: {missing}")
    for name in _ARCH_FIELDS:
        values.setdefault(name, source[name])


def engram_enabled(config: Any) -> bool:
    """Return whether the Xing4 config selects any Engram tables."""
    return bool(getattr(config, "engram_layer_ids", None) or getattr(config, "engram_assets_path", None))


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Engram JSON must be an object: {path}")
    return value


def _int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_token_map(mapping: list[int], vocab_size: int, compressed_size: int) -> None:
    if (len(mapping) != vocab_size or any(isinstance(value, bool) or not isinstance(value, int) for value in mapping)
            or min(mapping) < 0 or max(mapping) != compressed_size - 1
            or len(set(mapping)) != compressed_size):
        raise ValueError("Engram token_map must map every LM token into the declared compressed vocabulary")


def _hash_multipliers(layer_ids: list[int], order: int, compressed_size: int) -> list[list[int]]:
    upper = np.iinfo(np.int64).max // compressed_size // 2
    return [
        (np.random.default_rng(10007 * layer).integers(0, upper, size=order, dtype=np.int64) * 2 + 1).tolist()
        for layer in layer_ids
    ]


def _generated_assets(config: Any) -> dict[str, Any]:
    missing = [name for name in _ARCH_FIELDS if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"Missing Xing4 Engram fields: {missing}")
    compressed_size = _int(config.engram_compressed_vocab_size, "engram_compressed_vocab_size")
    if compressed_size != config.vocab_size:
        raise ValueError("Compressed Xing4 Engram requires engram_assets_path with an explicit token_map")
    # Xing4 counts hash groups from the bigram. The shared mapper takes the largest order.
    order = _int(config.engram_max_ngram_size, "engram_max_ngram_size") + 1
    layer_ids = config.engram_layer_ids
    _validate_layers(layer_ids, config.num_hidden_layers)
    heads = _int(config.engram_n_heads, "engram_n_heads")
    base = _int(config.engram_vocab_size, "engram_vocab_size", 2)
    primes, _ = build_scaled_engram_buckets(layer_ids, bucket_base=base, max_ngram_size=order, num_heads=heads)
    return {
        "layer_ids": layer_ids, "num_embeddings": config.engram_num_embeddings,
        "max_ngram_size": order, "bucket_base": base, "num_heads": heads,
        "head_dim": config.engram_head_dim, "pad_token_id": config.engram_pad_token_id,
        "token_map": list(range(config.vocab_size)),
        "multipliers": _hash_multipliers(layer_ids, order, compressed_size),
        "primes": primes,
    }


def _validate_layers(layer_ids: list[int], depth: int) -> None:
    if (not isinstance(layer_ids, (list, tuple)) or not layer_ids
            or any(isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < depth
                   for layer in layer_ids)
            or len(set(layer_ids)) != len(layer_ids)):
        raise ValueError("engram_layer_ids must identify distinct Xing4 backbone layers")


@dataclass(frozen=True)
class Xing4EngramConfig:
    """Construction facts for a shared Engram member on one Xing4 layer."""

    assets: dict[str, Any]
    layer_id: int
    hidden_size: int
    hc_mult: int
    embedding_dim: int
    eps: float
    logical_num_embeddings: int
    padded_num_embeddings: int
    clamp_value: float = 1e-6


def _validate_architecture_fields(config: Any, assets: dict[str, Any], compressed_size: int) -> None:
    asset_fields = ("layer_ids", "num_embeddings", "max_ngram_size", "bucket_base",
                    "num_heads", "head_dim", "pad_token_id", "compressed_vocab_size")
    metadata = {
        **assets, "compressed_vocab_size": compressed_size, "max_ngram_size": assets["max_ngram_size"] - 1,
    }
    for field, key in zip(_ARCH_FIELDS, asset_fields, strict=True):
        configured = getattr(config, field, None)
        if configured is not None and configured != metadata.get(key):
            raise ValueError(f"{field} does not match Xing4 Engram assets")


def resolve_engram_layers(config: Any) -> tuple[Xing4EngramConfig, ...]:
    """Validate the hash and table layouts before constructing device parameters."""
    path = getattr(config, "engram_assets_path", None)
    if not path and not engram_enabled(config):
        if any(getattr(config, name, None) is not None for name in _ARCH_FIELDS[1:]):
            raise ValueError("engram_layer_ids is required with Xing4 Engram settings")
        return ()
    assets = _read_json(path) if path else _generated_assets(config)
    if assets.get("source_model_type", "xing4_0") != "xing4_0":
        raise ValueError("Engram assets must use source_model_type xing4_0")
    layers = assets["layer_ids"]
    _validate_layers(layers, config.num_hidden_layers)
    mapping = assets["token_map"]
    compressed_size = getattr(config, "engram_compressed_vocab_size", None)
    if compressed_size is None:
        compressed_size = len(set(mapping))
    compressed_size = _int(compressed_size, "engram_compressed_vocab_size")
    _validate_token_map(mapping, config.vocab_size, compressed_size)
    if _int(assets["pad_token_id"], "engram_pad_token_id", 0) >= config.vocab_size:
        raise ValueError("engram_pad_token_id is outside the LM vocabulary")
    embedding_dim = _int(assets["head_dim"], "engram_head_dim")
    order = _int(assets["max_ngram_size"], "Engram asset max_ngram_size", 2)
    heads = _int(assets["num_heads"], "engram_n_heads")
    _validate_architecture_fields(config, assets, compressed_size)
    pad_multiple = _int(getattr(config, "engram_table_pad_multiple", 16), "engram_table_pad_multiple")
    logical = assets["num_embeddings"]
    padded = assets.get("padded_num_embeddings", logical)
    primes = assets["primes"]
    multipliers = assets["multipliers"]
    if any(len(rows) != len(layers) for rows in (primes, multipliers, logical, padded)):
        raise ValueError("Engram buckets, multipliers and table sizes must have one entry per layer")

    result = []
    for layer, buckets, factors, size, storage in zip(layers, primes, multipliers, logical, padded, strict=True):
        if len(buckets) != order - 1 or any(len(row) != heads for row in buckets) or len(factors) != order:
            raise ValueError(f"Invalid Xing4 Engram hash layout for layer {layer}")
        hash_rows = sum(_int(modulus, "Engram bucket modulus", 2) for row in buckets for modulus in row)
        if _int(size, "engram_num_embeddings") != hash_rows:
            raise ValueError(f"Xing4 layer {layer} declares {size} Engram rows but its hash buckets need {hash_rows}")
        storage = _int(storage, "padded_num_embeddings", size)
        if "padded_num_embeddings" not in assets or hasattr(config, "engram_table_pad_multiple"):
            storage = (storage + pad_multiple - 1) // pad_multiple * pad_multiple
        result.append(Xing4EngramConfig(
            assets, layer, config.hidden_size, config.hc_mult if config.enable_hyper_connections else 1,
            embedding_dim, config.rms_norm_eps, size, storage,
        ))
    return tuple(result)
