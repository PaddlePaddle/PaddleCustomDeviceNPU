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
TILE_CASE = [
    {"x_shape": [3, 2], "x_dtype": np.float32, "repeat_times": [2, 2]},
    {"x_shape": [3, 2], "x_dtype": np.float32, "repeat_times": [1, 2]},
    {"x_shape": [3, 2], "x_dtype": np.float32, "repeat_times": [3]},
    {"x_shape": [3, 2], "x_dtype": np.float32, "repeat_times": [2, 2, 2]},

    {"x_shape": [3, 2], "x_dtype": np.int32, "repeat_times": [2, 2]},
    {"x_shape": [3, 2], "x_dtype": np.int32, "repeat_times": [1, 2]},
    {"x_shape": [3, 2], "x_dtype": np.int32, "repeat_times": [3]},
    {"x_shape": [3, 2], "x_dtype": np.int32, "repeat_times": [2, 2, 2]},

    {"x_shape": [3, 2], "x_dtype": np.int64, "repeat_times": [2, 2]},
    {"x_shape": [3, 2], "x_dtype": np.int64, "repeat_times": [1, 2]},
    {"x_shape": [3, 2], "x_dtype": np.int64, "repeat_times": [3]},
    {"x_shape": [3, 2], "x_dtype": np.int64, "repeat_times": [2, 2, 2]},

    {"x_shape": [3, 2], "x_dtype": np.float16, "repeat_times": [2, 2]},
    {"x_shape": [3, 2], "x_dtype": np.float16, "repeat_times": [1, 2]},
    {"x_shape": [3, 2], "x_dtype": np.float16, "repeat_times": [3]},
    {"x_shape": [3, 2], "x_dtype": np.float16, "repeat_times": [2, 2, 2]},
]
# fmt: on


@ddt
class TestIndexPut(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.x_dtype = np.float32
        self.repeat_times = [2, 2]

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        return paddle.tile(x, repeat_times=self.repeat_times)

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def tile_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.tile_cast()
        return out

    @data(*TILE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, repeat_times):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.repeat_times = repeat_times
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
