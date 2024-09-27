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
INDEX_SAMPLE_CASE = [
    # float32 int32
    {"x_shape": [3, 4], "x_dtype": np.float32, "index_shape": [3, 3], "index_dtype": np.int32},
    {"x_shape": [3, 4], "x_dtype": np.float32, "index_shape": [3, 2], "index_dtype": np.int32},
    {"x_shape": [3, 1], "x_dtype": np.float32, "index_shape": [3, 1], "index_dtype": np.int32},
    {"x_shape": [10, 128], "x_dtype": np.float32, "index_shape": [10, 64], "index_dtype": np.int32},
    # float16 int32
    {"x_shape": [3, 4], "x_dtype": np.float16, "index_shape": [3, 3], "index_dtype": np.int32},
    {"x_shape": [3, 4], "x_dtype": np.float16, "index_shape": [3, 2], "index_dtype": np.int32},
    {"x_shape": [3, 1], "x_dtype": np.float16, "index_shape": [3, 1], "index_dtype": np.int32},
    {"x_shape": [10, 128], "x_dtype": np.float16, "index_shape": [10, 64], "index_dtype": np.int32},
    # int64 int32
    {"x_shape": [3, 4], "x_dtype": np.int64, "index_shape": [3, 3], "index_dtype": np.int32},
    {"x_shape": [3, 4], "x_dtype": np.int64, "index_shape": [3, 2], "index_dtype": np.int32},
    {"x_shape": [3, 1], "x_dtype": np.int64, "index_shape": [3, 1], "index_dtype": np.int32},
    {"x_shape": [10, 128], "x_dtype": np.int64, "index_shape": [10, 64], "index_dtype": np.int32},
    # float16 int64
    {"x_shape": [3, 4], "x_dtype": np.float16, "index_shape": [3, 3], "index_dtype": np.int64},
    {"x_shape": [3, 4], "x_dtype": np.float16, "index_shape": [3, 2], "index_dtype": np.int64},
    {"x_shape": [3, 1], "x_dtype": np.float16, "index_shape": [3, 1], "index_dtype": np.int64},
    {"x_shape": [10, 128], "x_dtype": np.float16, "index_shape": [10, 64], "index_dtype": np.int64},
]
# fmt: on


@ddt
class TestIndexSample(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 4]
        self.x_dtype = np.float32
        self.index_shape = [3, 3]
        self.index_dtype = np.int32
        self.index = np.random.randint(
            low=0, high=self.x_shape[1], size=self.index_shape
        ).astype(self.index_dtype)

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        index = paddle.to_tensor(self.index, dtype=self.index_dtype)
        return paddle.index_sample(x, index)

    def get_numpy_output(self):
        index_array = []
        for i in range(self.index_shape[0]):
            for j in self.index[i]:
                index_array.append(self.data_x[i, j])
        index_array = np.array(index_array).astype(self.x_dtype)
        out = np.reshape(index_array, self.index_shape)
        return out

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_output()
        return out

    @data(*INDEX_SAMPLE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, index_shape, index_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.index_shape = index_shape
        self.index_dtype = index_dtype
        self.index = np.random.randint(
            low=0, high=self.x_shape[1], size=self.index_shape
        ).astype(self.index_dtype)
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
