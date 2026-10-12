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
"""Fixed-router propagation through the V4.1 training entry point."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.models.deepseek_v41.adapter.validation import cropped_model

from tests.common.mark_utils import arg_mark


class TestCroppedModelFixedRouter(unittest.TestCase):
    """Check public training configuration wiring without constructing a full model."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_training_builder_forwards_switch_to_configuration(self) -> None:
        """Feature: fixed MoE routing.
        Description: Both text and vision crops receive the explicitly chosen debug switch.
        Expectation: The training builder passes the switch into its configuration builder.
        """
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(
                cropped_model, "build_deepseek_v41_validation_config", return_value=SimpleNamespace(),
            ) as configure, patch.object(cropped_model.HyperAutoModelForCausalLM, "from_config") as build:
                cropped_model.build_cropped_deepseek_v41("model", "assets", fix_router=enabled)
                self.assertEqual(configure.call_args.kwargs["fix_router"], enabled)
                self.assertIs(build.call_args.args[0], configure.return_value)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_string_switch_fails_before_source_files_are_read(self) -> None:
        """Feature: fixed MoE routing.
        Description: A YAML string false must fail before any model loading takes place.
        Expectation: A string switch raises ValueError before source I/O.
        """
        with self.assertRaisesRegex(ValueError, "fix_router must be a bool"):
            cropped_model.build_deepseek_v41_validation_config("missing", "missing", fix_router="false")
