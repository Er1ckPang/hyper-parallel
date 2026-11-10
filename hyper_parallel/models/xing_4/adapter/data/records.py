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
"""Prepare MF ordered records through the framework Indexed Dataset extension."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.indexed.indexed_pretrain_dataset import GPTFromMRDataset


class Xing4MFRecordDataset(GPTFromMRDataset):
    """Truncate or pad each selected document independently before label shifting.

    The parent supplies reader construction, split indices and sample count.
    Only record preparation differs from its fixed-length, unpadded contract.
    Sampling, collation and parallel batch distribution remain framework-owned.
    """

    def _finalize(self) -> None:
        """Require one extra label token and enough records in the selected split."""
        if not self.config.add_extra_token_to_sequence or self._pad_token_id < 0:
            raise ValueError("MF records require S+1 shifting and an explicit tokenizer pad ID")
        if self.num_samples > len(self.indices):
            raise ValueError(f"Requested {self.num_samples} samples from {len(self.indices)} records")
        self.indices = self.indices[:self.num_samples]

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Return shifted tokens and labels for one document, without packing."""
        sequence_id = int(self.indices[index])
        window = self.config.sequence_length + 1
        count = min(int(self.dataset.sequence_lengths[sequence_id]), window)
        text = np.full(window, self._pad_token_id, dtype=np.int64)
        text[:count] = self.dataset.get(sequence_id, length=count)
        if np.any(text < 0) or np.any(text >= self.config.tokenizer.vocab_size):
            raise ValueError(f"Record {sequence_id} contains a token outside the tokenizer vocabulary")
        # An EOD before the last input token also separates any following pad positions.
        if np.any(text[:-2] == self.config.tokenizer.eod):
            raise ValueError(f"Record {sequence_id} requires intra-record EOD attention, unsupported by Xing4")
        tokens, labels = text[:-1].copy(), text[1:].copy()
        # Preserve pad-ID supervision before mapping input padding to token zero.
        labels[labels == self._pad_token_id] = IGNORE_INDEX
        tokens[tokens == self._pad_token_id] = 0
        return {"input_ids": tokens, "labels": labels}
