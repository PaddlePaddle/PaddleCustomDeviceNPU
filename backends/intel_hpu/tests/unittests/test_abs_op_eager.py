# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function, division
import unittest

import paddle
import numpy as np


from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)


class TestNPUAbsBF16(OpTest):
    def setUp(self):
        self.op_type = "abs"
        self.set_npu()

        np.random.seed(1024)
        x = convert_float_to_uint16(
            np.random.uniform(-1, 1, [4, 25]).astype(np.float32)
        )
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(convert_uint16_to_float(x))

        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
