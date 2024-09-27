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
DROPOUT_CASE = [
    {"x_shape": [6], "p": 0.5, "axis": None, "training": True, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [6], "p": 0.5, "axis": None, "training": False, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [2, 3], "p": 0.5, "axis": [0, 1], "training": True, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [2, 3], "p": 0.5, "axis": [0, 1], "training": False, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [2, 3], "p": 0.5, "axis": [1], "training": True, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [2, 3], "p": 0.5, "axis": [1], "training": False, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [2, 3], "p": 0.5, "axis": [0], "training": True, "mode": "upscale_in_train", "x_dtype": np.float32},
    {"x_shape": [2, 3], "p": 0.5, "axis": [0], "training": False, "mode": "upscale_in_train", "x_dtype": np.float32},
    # TODO(xuelei.wan): Add test cases about mode='downscale_in_infer'
]
# fmt: on


@ddt
class TestDropout(TestAPIBase):
    def setUp(self):
        self.data_x = None
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 5]
        self.p = 0.0
        self.axis = None
        self.training = False
        self.mode = "upscale_in_train"
        self.x_dtype = np.float32
        self.mask = None

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype) * 0 + 2

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        result = paddle.nn.functional.dropout(
            x, p=self.p, axis=self.axis, training=self.training, mode=self.mode
        )
        return result

    def expect_output(self):
        return self.calc_result(self.forward, "cpu")
        # numpy impl
        # c = 1 - self.p
        # mask = np.random.binomial(1, c, size=self.data_x.shape).astype(self.x_dtype) > self.p
        # if self.mode == 'upscale_in_train':
        #     if self.training:
        #         out = self.data_x * mask / c
        #         return out
        #     else:
        #         out = self.data_x
        #         return out
        # else:
        #     if self.training:
        #         out = self.data_x * mask
        #         return out
        #     else:
        #         out = self.data_x * c
        #         return out

    @data(*DROPOUT_CASE)
    @unpack
    def test_check_output(self, x_shape, p, axis, training, mode, x_dtype):
        self.x_shape = x_shape
        self.p = 1.0
        self.axis = axis
        self.training = training
        self.mode = mode
        self.x_dtype = x_dtype

        self.prepare_data()
        t0 = self.forward().numpy()
        t1 = self.expect_output()

        self.assertEqual(t0.max(), t1.max())


if __name__ == "__main__":
    unittest.main()
