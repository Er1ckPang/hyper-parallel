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
"""Verify how the Xing4 architecture describes the shared Engram layout."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from hyper_parallel.components.modules.engram import Engram
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models.xing_4.adapter.distributed.engram_parallel import with_engram_parallelism
from hyper_parallel.models.xing_4.engram_config import merge_engram_config, resolve_engram_layers


class TestXing4EngramConfig(unittest.TestCase):
    """Check attachment layout, logical rows, and scratch hashing."""

    def setUp(self):
        """Define the attachment and the matching backbone shape."""
        self.architecture = {
            "vocab_size": 131072, "engram_layer_ids": [1, 11], "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 3, "engram_vocab_size": 16000000, "engram_n_heads": 8, "engram_head_dim": 256,
            "engram_pad_token_id": 3, "engram_compressed_vocab_size": 131072,
            "use_cache": True, "layer_types": ["kda", "dsa"],
        }
        self.model = {
            "vocab_size": 131072, "num_hidden_layers": 40, "hidden_size": 3584,
            "enable_hyper_connections": True, "rms_norm_eps": 1e-6, "hc_mult": 4,
        }

    def test_utf16_architecture_and_meta_parameters(self):
        """Import only Engram fields and preserve the attachment checkpoint rows."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "architecture.json"
            source.write_text(json.dumps(self.architecture), encoding="utf-16")
            values = dict(self.model, engram_config_path=str(source))
            merge_engram_config(values)
        self.assertNotIn("layer_types", values)
        self.assertNotIn("use_cache", values)
        config = SimpleNamespace(**values)
        layouts = resolve_engram_layers(config)
        self.assertEqual([layout.layer_id for layout in layouts], [1, 11])
        for layout, rows, padded in zip(
            layouts, (384006168, 384016682), (384006176, 384016688), strict=True,
        ):
            with torch.device("meta"):
                engram = Engram(config=layout)
            self.assertEqual(engram.embed.weight.shape, (padded, 256))
            self.assertEqual(engram.wkv.weight.shape, (17920, 6144))
            self.assertEqual(engram.q_weight.shape, (4, 3584))
            self.assertEqual(engram.checkpoint_logical_shapes(), {"embed.weight": (rows, 256)})
            self.assertEqual(engram.hash_mapping.max_ngram_size, 4)

    def test_tiny_scratch_table_and_masks(self):
        """Generate the same hashes twice and keep masked-out residual rows intact."""
        config = SimpleNamespace(**dict(
            self.model, vocab_size=32, hidden_size=4, enable_hyper_connections=False,
            engram_layer_ids=[1], engram_num_embeddings=[156], engram_max_ngram_size=3,
            engram_vocab_size=17, engram_n_heads=2, engram_head_dim=2,
            engram_pad_token_id=3, engram_compressed_vocab_size=32,
        ))
        layout = resolve_engram_layers(config)[0]
        self.assertEqual(layout.assets, resolve_engram_layers(config)[0].assets)
        engram = Engram(config=layout)
        ids = torch.tensor([[1, 2, 4, 8, 9]])
        starts = torch.tensor([[True, False, True, False, False]])
        mask = torch.tensor([[True, True, True, False, True]])
        hidden = torch.randn(1, 5, 1, 4)
        actual = engram(hidden, ids, starts, mask)
        torch.testing.assert_close(actual[:, 3], hidden[:, 3])
        actual.square().mean().backward()
        self.assertGreater(engram.embed.weight.grad.norm().item(), 0.0)

        config.engram_num_embeddings = [157]
        with self.assertRaisesRegex(ValueError, "hash buckets need 156"):
            resolve_engram_layers(config)
        config.engram_compressed_vocab_size = 31
        with self.assertRaisesRegex(ValueError, "explicit token_map"):
            resolve_engram_layers(config)


    def test_explicit_placements_for_single_rank_and_ep(self):
        """Keep every trainable Engram parameter covered at EP1 and shard only the table at EP2."""
        for ep_size, expected in ((1, {"tp": Replicate()}), (2, {"ep": Shard(0)})):
            with self.subTest(ep_size=ep_size):
                setup = DistributedSetup(mesh_context=MeshContext(ep_size=ep_size))
                result = with_engram_parallelism(setup)
                spec = result.plan_overrides["model.layers.*.engram"]
                self.assertIsNone(setup.plan_overrides)
                self.assertEqual(spec.params["embed.weight"], expected)
                for name in ("wkv.weight", "q_weight", "k_weight"):
                    self.assertEqual(spec.params[name], {"tp": Replicate()})
                self.assertFalse(spec.region_dispatch)
                self.assertTrue(callable(spec.local_compute_fn))
