# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for to_maxtext.py"""

import unittest
import ml_dtypes
import numpy as np
import pytest

try:
  import torch
except ImportError:
  torch = None

pytestmark = [pytest.mark.decoupled_target]

from maxtext.checkpoint_conversion.to_maxtext import (
    _build_single_axis_stacked_tensor,
    _convert_tensor_to_numpy,
    _extract_conversion_args,
    resolve_scale_key,
)


class ResolveScaleKeyTest(unittest.TestCase):

  def test_exact_match(self):
    container = {"model.layers.0.mlp.gate.weight": 1}
    self.assertEqual(resolve_scale_key("model.layers.0.mlp.gate.weight", container), "model.layers.0.mlp.gate.weight")

  def test_weight_scale_to_weight_scale_inv_fallback(self):
    container = {"model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": 1}
    resolved = resolve_scale_key("model.layers.0.mlp.experts.0.gate_proj.weight_scale", container)
    self.assertEqual(resolved, "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv")

  def test_unmatched_key_returns_original(self):
    container = {}
    self.assertEqual(
        resolve_scale_key("model.layers.0.mlp.experts.0.gate_proj.weight_scale", container),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
    )


class ConvertTensorToNumpyTest(unittest.TestCase):

  def test_torch_float8_e4m3fn_to_numpy(self):
    if torch is None:
      self.skipTest("torch not available")
    t = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    arr = _convert_tensor_to_numpy(t, save_dtype="float8_e4m3fn")
    self.assertEqual(arr.dtype, ml_dtypes.float8_e4m3fn)
    self.assertEqual(arr.shape, (3,))

  def test_torch_float8_e4m3fn_to_float32(self):
    if torch is None:
      self.skipTest("torch not available")
    t = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    arr = _convert_tensor_to_numpy(t, save_dtype="float32")
    self.assertEqual(arr.dtype, np.float32)

  def test_torch_bfloat16_to_numpy(self):
    if torch is None:
      self.skipTest("torch not available")
    t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    arr = _convert_tensor_to_numpy(t, save_dtype="bfloat16")
    self.assertEqual(arr.dtype, ml_dtypes.bfloat16)

  def test_numpy_passthrough(self):
    arr_in = np.array([1.0, 2.0], dtype=np.float32)
    arr_out = _convert_tensor_to_numpy(arr_in, save_dtype="float8_e4m3fn")
    self.assertTrue(np.array_equal(arr_in, arr_out))


class ExtractConversionArgsTest(unittest.TestCase):

  def test_extract_mixed_flags(self):
    args = [
        "base.yml",
        "model_name=qwen3.5-35b-a3b-fp8",
        "save_dtype=float8_e4m3fn",
        "--lazy_load_tensors=false",
        "eager_load_method=safetensors",
        "--simulated_cpu_devices_count=4",
        "steps=10",
    ]
    extracted, remaining = _extract_conversion_args(args)
    self.assertEqual(extracted["save_dtype"], "float8_e4m3fn")
    self.assertEqual(extracted["lazy_load_tensors"], False)
    self.assertEqual(extracted["eager_load_method"], "safetensors")
    self.assertEqual(extracted["simulated_cpu_devices_count"], 4)
    self.assertEqual(remaining, ["base.yml", "model_name=qwen3.5-35b-a3b-fp8", "steps=10"])


class BuildSingleAxisStackedTensorTest(unittest.TestCase):
  """Tests the stacking axis of _build_single_axis_stacked_tensor with scanned layers."""

  class _Config:
    scan_layers = True
    param_scan_axis = 1

  def _stack(self, mt_key, target_shape):
    tensors = {"k0": np.zeros((4, 8), np.float32), "k1": np.ones((4, 8), np.float32)}
    return _build_single_axis_stacked_tensor(
        ["k0", "k1"], tensors.__getitem__, None, target_shape, self._Config(), mt_key
    )

  def test_scanned_moe_block_stacks_layers_on_scan_axis(self):
    # e.g. mixtral and deepseek2/3: the list is over layers of a scanned stack.
    for mt_key in (
        "params-decoder-layers-MoeBlock_0-gate-kernel",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel",
    ):
      with self.subTest(mt_key=mt_key):
        result = self._stack(mt_key, (4, 2, 8))
        self.assertEqual(result.shape, (4, 2, 8))
        np.testing.assert_array_equal(result[:, 1], np.ones((4, 8)))

  def test_unscanned_layer_moe_block_stacks_experts_on_axis_0(self):
    # e.g. DeepSeek V4: an unscanned layer whose list is over experts.
    result = self._stack("params-decoder-layers_0-mlp-MoeBlock_0-wi_0", (2, 4, 8))
    self.assertEqual(result.shape, (2, 4, 8))
    np.testing.assert_array_equal(result[1], np.ones((4, 8)))


if __name__ == "__main__":
  unittest.main()
