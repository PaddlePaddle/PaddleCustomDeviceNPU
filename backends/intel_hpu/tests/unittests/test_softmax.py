# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

SEED = 2021

import os
from util import enable_paddle_static_mode

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
intel_hpus_static_mode = os.environ.get(
    "FLAGS_static_mode_intel_hpus", 0
)  # default is dynamic mode test FLAGS_static_mode_intel_hpus=0


class TestSoftmax(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        enable_paddle_static_mode(int(intel_hpus_static_mode))

        self.op_type = "softmax"
        self.init_dtype()

        x = np.random.random([10, 10]).astype(self.dtype)
        np_out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.inputs = {"X": x}

        self.attrs = {}
        self.outputs = {"Out": np_out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
