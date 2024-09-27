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
SCATTER_CASE = [
    {"x_shape": [3, 2], "x_dtype": np.float32, "index": [1], "index_dtype": np.int32, "updates_shape": [1, 2], "overwrite": True},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index": [1, 2], "index_dtype": np.int32, "updates_shape": [2, 3], "overwrite": True},
    {"x_shape": [3, 2], "x_dtype": np.float32, "index": [1], "index_dtype": np.int32, "updates_shape": [1, 2], "overwrite": False},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index": [1, 2], "index_dtype": np.int32, "updates_shape": [2, 3], "overwrite": False},
    # TopsFlame not support float16
    # {"x_shape": [3, 2], "x_dtype": np.float16, "index": [1], "index_dtype": np.int32, "updates_shape": [1, 2], "overwrite": True},
    # {"x_shape": [3, 3], "x_dtype": np.float16, "index": [1, 2], "index_dtype": np.int32, "updates_shape": [2, 3], "overwrite": True},

    {"x_shape": [3, 2], "x_dtype": np.float32, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": True},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": True},
    {"x_shape": [3, 2], "x_dtype": np.float32, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": False},
    {"x_shape": [3, 3], "x_dtype": np.float32, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": False},
    # TopsFlame not support float16
    # {"x_shape": [3, 2], "x_dtype": np.float16, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": True},
    # {"x_shape": [3, 3], "x_dtype": np.float16, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": True},

    {"x_shape": [3, 2], "x_dtype": np.int32, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": True},
    {"x_shape": [3, 3], "x_dtype": np.int32, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": True},
    {"x_shape": [3, 2], "x_dtype": np.int32, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": False},
    {"x_shape": [3, 3], "x_dtype": np.int32, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": False},

    {"x_shape": [3, 2], "x_dtype": np.int64, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": True},
    {"x_shape": [3, 3], "x_dtype": np.int64, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": True},
    {"x_shape": [3, 2], "x_dtype": np.int64, "index": [1], "index_dtype": np.int64, "updates_shape": [1, 2], "overwrite": False},
    {"x_shape": [3, 3], "x_dtype": np.int64, "index": [1, 2], "index_dtype": np.int64, "updates_shape": [2, 3], "overwrite": False},


]
# fmt: on


@ddt
class TestScatter(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.x_dtype = np.float32
        self.index = [1]
        self.index_dtype = np.int32
        self.updates_shape = [1, 2]
        self.overwrite = True

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        self.updates = self.generate_data(self.updates_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        index = paddle.to_tensor(self.index, dtype=self.index_dtype)
        updates = paddle.to_tensor(self.updates, dtype=self.x_dtype)
        return paddle.scatter(x, index=index, updates=updates, overwrite=self.overwrite)

    def scatter_numpy(self):
        output_np = np.copy(self.data_x)
        if self.overwrite:
            output_np[self.index] = self.updates
        else:
            zeros_np = np.zeros(self.updates_shape).astype(self.x_dtype)
            output_np[self.index] = zeros_np
            for i in range(0, len(self.index)):
                output_np[self.index[i]] += self.updates[i]
        return output_np

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.scatter_numpy()
        return out

    @data(*SCATTER_CASE)
    @unpack
    def test_check_output(
        self, x_shape, x_dtype, index, index_dtype, updates_shape, overwrite
    ):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.index = index
        self.index_dtype = index_dtype
        self.updates_shape = updates_shape
        self.overwrite = overwrite
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
