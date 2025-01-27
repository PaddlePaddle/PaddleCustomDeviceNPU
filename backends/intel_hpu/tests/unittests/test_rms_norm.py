# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from tests.op_test import OpTest
import paddle

paddle.enable_static()

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def reference_rms_norm(x, norm_weight, epsilon, begin_norm_axis):
    variance = np.power(x, 2).mean(-1, keepdims=True)
    x = x / np.sqrt(variance + epsilon)
    rmsNorm = x * norm_weight
    return rmsNorm


def layer_norm_wrapper(x, scale=None, bias=None, epsilon=1e-05, begin_norm_axis=1):
    input_shape = list(x.shape)
    normalized_shape = input_shape[begin_norm_axis:]
    return paddle.nn.functional.layer_norm(
        x, normalized_shape, weight=scale, bias=bias, epsilon=epsilon
    )


class TestRMSNormOp(OpTest):
    def setUp(self):
        self.python_api = layer_norm_wrapper
        self.public_python_api = layer_norm_wrapper
        self.op_type = "rms_norm"
        self.set_hpu()
        self.init_dtype()
        self.init_input()
        self.attrs = {
            "begin_norm_axis": self.begin_norm_axis,
            "epsilon": 1e-6,
            "quant_scale": -1,
            "quant_round_type": 0,
            "quant_max_bound": 0,
            "quant_min_bound": 0,
        }
        self.inputs = {
            "x": OpTest.np_dtype_to_base_dtype(self.x),
            "norm_weight": OpTest.np_dtype_to_base_dtype(self.w),
        }
        self.outputs = {"out": self.result}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

    def init_dtype(self):
        self.dtype = np.float16

    def init_input(self):
        self.x = np.random.rand(2, 4).astype(self.dtype)
        self.w = np.random.rand(4).astype(self.dtype)
        self.begin_norm_axis = 1
        self.result = reference_rms_norm(self.x, self.w, 1e-6, self.begin_norm_axis)


if __name__ == "__main__":
    unittest.main()
