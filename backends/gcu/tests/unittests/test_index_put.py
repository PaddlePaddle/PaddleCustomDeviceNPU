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
INDEX_PUT_CASE = [
    {"x_shape": [3, 2], "x_dtype": np.float32, "index1": [1, 2], "index2": [1, 0], "index_dtype": np.int32, "value_shape": [2], "accumulate": False},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index1": [1, 2, 1], "index2": [2, 1, 0], "index_dtype": np.int32, "value_shape": [3], "accumulate": False},
    {"x_shape": [6], "x_dtype": np.float32, "index1": [5, 0, 3, 4, 1, 2], "index2": None, "index_dtype": np.int32, "value_shape": [6], "accumulate": False},
    {"x_shape": [3, 2], "x_dtype": np.float32, "index1": [1, 2], "index2": [1, 0], "index_dtype": np.int32, "value_shape": [2], "accumulate": True},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index1": [1, 2, 1], "index2": [2, 1, 0], "index_dtype": np.int32, "value_shape": [3], "accumulate": True},
    {"x_shape": [6], "x_dtype": np.float32, "index1": [5, 0, 3, 4, 1, 2], "index2": None, "index_dtype": np.int32, "value_shape": [6], "accumulate": True},

    {"x_shape": [3, 2], "x_dtype": np.float32, "index1": [1, 2], "index2": [1, 0], "index_dtype": np.int64, "value_shape": [2], "accumulate": False},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index1": [1, 2, 1], "index2": [2, 1, 0], "index_dtype": np.int64, "value_shape": [3], "accumulate": False},
    {"x_shape": [6], "x_dtype": np.float32, "index1": [5, 0, 3, 4, 1, 2], "index2": None, "index_dtype": np.int64, "value_shape": [6], "accumulate": False},

    {"x_shape": [3, 2], "x_dtype": np.int64, "index1": [1, 2], "index2": [1, 0], "index_dtype": np.int32, "value_shape": [2], "accumulate": True},
    {"x_shape": [3, 3], "x_dtype": np.int64, "index1": [1, 2, 1], "index2": [2, 1, 0], "index_dtype": np.int32, "value_shape": [3], "accumulate": True},
    {"x_shape": [6], "x_dtype": np.int64, "index1": [5, 0, 3, 4, 1, 2], "index2": None, "index_dtype": np.int32, "value_shape": [6], "accumulate": True},

    {"x_shape": [3, 2], "x_dtype": np.int64, "index1": [1, 2], "index2": [1, 0], "index_dtype": np.int64, "value_shape": [2], "accumulate": False},
    {"x_shape": [3, 3], "x_dtype": np.int64, "index1": [1, 2, 1], "index2": [2, 1, 0], "index_dtype": np.int64, "value_shape": [3], "accumulate": False},
    {"x_shape": [6], "x_dtype": np.int64, "index1": [5, 0, 3, 4, 1, 2], "index2": None, "index_dtype": np.int64, "value_shape": [6], "accumulate": False},

    {"x_shape": [3, 2], "x_dtype": np.float16, "index1": [1, 2], "index2": [1, 0], "index_dtype": np.int64, "value_shape": [2], "accumulate": False},
    {"x_shape": [3, 3], "x_dtype": np.float16, "index1": [1, 2, 1], "index2": [2, 1, 0], "index_dtype": np.int64, "value_shape": [3], "accumulate": False},
    {"x_shape": [6], "x_dtype": np.float16, "index1": [5, 0, 3, 4, 1, 2], "index2": None, "index_dtype": np.int64, "value_shape": [6], "accumulate": False},
]
# fmt: on


@ddt
class TestIndexPut(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.x_dtype = np.float32
        self.index1 = [1]
        self.index2 = [1]
        self.index_dtype = np.int32
        self.value_shape = [2]
        self.accumulate = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        self.values = self.generate_data(self.value_shape, self.x_dtype)
        if self.x_dtype == np.int64:
            self.data_x = self.generate_integer_data(
                self.x_shape, self.x_dtype, -32768, 32767
            )
            self.values = self.generate_integer_data(
                self.value_shape, self.x_dtype, -32768, 32767
            )

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        index1 = paddle.to_tensor(self.index1, dtype=self.index_dtype)
        if self.index2 is not None:
            index2 = paddle.to_tensor(self.index2, dtype=self.index_dtype)
            index = [index1, index2]
        else:
            index = [index1]
        values = paddle.to_tensor(self.values, dtype=dtype)
        # print(x)
        # print(index)
        # print(values)
        return paddle.index_put(
            x, indices=index, value=values, accumulate=self.accumulate
        )

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def index_put_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.index_put_cast()
        return out

    @data(*INDEX_PUT_CASE)
    @unpack
    def test_check_output(
        self, x_shape, x_dtype, index1, index2, index_dtype, value_shape, accumulate
    ):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.index1 = index1
        self.index2 = index2
        self.index_dtype = index_dtype
        self.value_shape = value_shape
        self.accumulate = accumulate
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
