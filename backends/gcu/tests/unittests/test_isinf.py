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
ISINF_CASE = [
    {"shape": [1], "dtype": np.float32},
    {"shape": [9], "dtype": np.float32},
    {"shape": [3, 6], "dtype": np.float32},
    {"shape": [2, 3, 6], "dtype": np.float32},
    {"shape": [2, 2, 3, 6], "dtype": np.float32},
    {"shape": [1], "dtype": np.float16},
    {"shape": [9], "dtype": np.float16},
    {"shape": [3, 6], "dtype": np.float16},
    {"shape": [2, 3, 6], "dtype": np.float16},
    {"shape": [2, 2, 3, 6], "dtype": np.float16},
    # {"shape": [1], "dtype": np.float64},
    # {"shape": [9], "dtype": np.float64},
    # {"shape": [3, 6], "dtype": np.float64},
    # {"shape": [2, 3, 6], "dtype": np.float64},
    # {"shape": [2, 2, 3, 6], "dtype": np.float64},
]
# fmt: on


@ddt
class TestIsInf(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 6]
        self.dtype = np.float32
        self.data_set = [np.inf, -np.inf, 0, 6, np.nan, -np.nan]
        self.data_set_size = len(self.data_set)

    def prepare_data(self):
        index = np.random.randint(
            low=0, high=self.data_set_size, size=self.shape, dtype=np.int32
        ).flatten()
        value = [self.data_set[i] for i in index]
        self.data_x = np.array(value, self.dtype).reshape(self.shape)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        return paddle.isinf(x)

    def expect_output(self):
        return np.isinf(self.data_x)

    @data(*ISINF_CASE)
    @unpack
    def test_check_output(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
