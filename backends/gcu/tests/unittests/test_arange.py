# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
ARANGE_CASE = [
    {"input_case": (0, 1, 0.2), "input_dtype": np.float32, "out_dtype": np.float32},
    {"input_case": (0, 1, 0.3), "input_dtype": np.float32, "out_dtype": np.float32},
    {"input_case": (0, 1, 0.2), "input_dtype": np.float64, "out_dtype": np.float64},
    {"input_case": (0, 1, 0.3), "input_dtype": np.float64, "out_dtype": np.float64},
    # {"input_case": (0, 1, 0.2), "input_dtype": np.float16, "out_dtype": np.float16},
    # {"input_case": (0, 1, 0.3), "input_dtype": np.float16, "out_dtype": np.float16},
    {"input_case": (0, 10, 2), "input_dtype": np.int32, "out_dtype": np.int32},
    {"input_case": (0, 10, 3), "input_dtype": np.int32, "out_dtype": np.int32},
    {"input_case": (0, 10, 2), "input_dtype": np.int64, "out_dtype": np.int64},
    {"input_case": (0, 10, 3), "input_dtype": np.int64, "out_dtype": np.int64},
    # {"input_case": (0, 1, 0.2), "input_dtype": np.float32, "out_dtype": np.int32},
    # {"input_case": (0, 1, 0.3), "input_dtype": np.float32, "out_dtype": np.int32},
    {"input_case": (0, 10, 2), "input_dtype": np.int32, "out_dtype": np.float32},
    {"input_case": (0, 10, 3), "input_dtype": np.int32, "out_dtype": np.float32},
    {"input_case": (0, 10, 3), "input_dtype": np.int32, "out_dtype": None},
]
# fmt: on


@ddt
class TestArange(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.input_case = (0, 1, 0.2)
        self.input_dtype = np.float32
        self.out_dtype = np.float32

    def prepare_data(self):
        self.data_start = np.array([self.input_case[0]]).astype(self.input_dtype)
        self.data_end = np.array([self.input_case[1]]).astype(self.input_dtype)
        self.data_step = np.array([self.input_case[2]]).astype(self.input_dtype)

    def forward(self):
        start = paddle.to_tensor(self.data_start, dtype=self.input_dtype)
        end = paddle.to_tensor(self.data_end, dtype=self.input_dtype)
        step = paddle.to_tensor(self.data_step, dtype=self.input_dtype)
        return paddle.arange(start, end, step, dtype=self.out_dtype)

    @data(*ARANGE_CASE)
    @unpack
    def test_check_output(self, input_case, input_dtype, out_dtype):
        self.input_case = input_case
        self.input_dtype = input_dtype
        self.out_dtype = out_dtype
        self.check_output_gcu_with_cpu(self.forward)


if __name__ == "__main__":
    unittest.main()
