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
GATHER_ND_CASE = [
    {"x_shape": [3, 2], "x_dtype": np.float32, "index": [[1]], "index_dtype": np.int32},
    {"x_shape": [3, 2], "x_dtype": np.float32, "index": [[1]], "index_dtype": np.int64},
    {"x_shape": [3, 2], "x_dtype": np.int64, "index": [[1]], "index_dtype": np.int32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "index": [[1]], "index_dtype": np.int32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "index": [[0, 2]], "index_dtype": np.int32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "index": [[1, 2, 3]], "index_dtype": np.int32},
    {"x_shape": [3, 4, 5, 6], "x_dtype": np.float32, "index": [[1, 2, 3]], "index_dtype": np.int32},
    {"x_shape": [3, 4, 5, 6, 7], "x_dtype": np.float32, "index": [[1, 2, 3]], "index_dtype": np.int32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float16, "index": [[1]], "index_dtype": np.int32},
    {"x_shape": [3, 4, 5, 6], "x_dtype": np.float16, "index": [[1, 2, 3]], "index_dtype": np.int32},
]
# fmt: on


@ddt
class TestGatherNd(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 2]
        self.x_dtype = np.int32
        self.index = [[0, 1]]
        self.index_dtype = np.int32

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        index = paddle.to_tensor(self.index, dtype=self.index_dtype)
        return paddle.gather_nd(x, index)

    def gather_nd_cast(self):
        x = paddle.to_tensor(self.data_x, dtype="float32")
        index = paddle.to_tensor(self.index, dtype=self.index_dtype)
        return paddle.gather_nd(x, index)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.gather_nd_cast()
        return out

    @data(*GATHER_ND_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, index, index_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.index = index
        self.index_dtype = index_dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
