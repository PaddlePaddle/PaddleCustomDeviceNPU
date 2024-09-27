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
EXPAND_CASE = [
    {"x_shape": [6], "x_dtype": np.float32, "expand_shape": [1, 6], "reps": [1, 1]},
    {"x_shape": [1, 6], "x_dtype": np.float32, "expand_shape": [3, 6], "reps": [3, 1]},
    {"x_shape": [2, 3, 1], "x_dtype": np.float32, "expand_shape": [2, 3, 6], "reps": [1, 1, 6]},
    {"x_shape": [2, 10, 5], "x_dtype": np.float32, "expand_shape": [2, 10, 5], "reps": [1, 1, 1]},
    {"x_shape": [2, 4, 1, 15], "x_dtype": np.float32, "expand_shape": [2, 4, 4, 15], "reps": [1, 1, 4, 1]},
    {"x_shape": [2, 4, 1, 15], "x_dtype": np.float32, "expand_shape": [2, -1, 4, -1], "reps": [1, 1, 4, 1]},
    {"x_shape": [6], "x_dtype": np.float16, "expand_shape": [1, 6], "reps": [1, 1]},
    {"x_shape": [1, 6], "x_dtype": np.float16, "expand_shape": [3, 6], "reps": [3, 1]},
    {"x_shape": [2, 3, 1], "x_dtype": np.float16, "expand_shape": [2, 3, 6], "reps": [1, 1, 6]},
    {"x_shape": [2, 10, 5], "x_dtype": np.float16, "expand_shape": [2, 10, 5], "reps": [1, 1, 1]},
    {"x_shape": [2, 4, 1, 15], "x_dtype": np.float16, "expand_shape": [2, 4, 4, 15], "reps": [1, 1, 4, 1]},
    {"x_shape": [2, 4, 1, 15], "x_dtype": np.float16, "expand_shape": [2, -1, 4, -1], "reps": [1, 1, 4, 1]},
    {"x_shape": [6], "x_dtype": np.int32, "expand_shape": [1, 6], "reps": [1, 1]},
    {"x_shape": [6], "x_dtype": np.int64, "expand_shape": [1, 6], "reps": [1, 1]},
    {"x_shape": [2, 4, 1, 15], "x_dtype": np.int64, "expand_shape": [2, -1, 4, -1], "reps": [1, 1, 4, 1]},

]
# fmt: on


@ddt
class TestExpand(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [1, 3, 6]
        self.x_dtype = np.float32
        self.expand_shape = [2, 3]
        self.reps = [2, 1]

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.expand(x, self.expand_shape)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = np.tile(self.data_x, self.reps)
        return out

    @data(*EXPAND_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, expand_shape, reps):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.expand_shape = expand_shape
        self.reps = reps
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
