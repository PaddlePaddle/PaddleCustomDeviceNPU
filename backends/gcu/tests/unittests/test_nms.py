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
NMS_CASE = [
    {"x_shape": [4, 4], "iou_threshold": 0.1, "scores": [4, 3, 2, 1], "x_dtype": np.float32},
    {"x_shape": [5, 4], "iou_threshold": 0.1, "scores": [5, 4, 3, 2, 1], "x_dtype": np.float32},
    {"x_shape": [6, 4], "iou_threshold": 0.3, "scores": [6, 5, 4, 3, 2, 1], "x_dtype": np.float32},
    {"x_shape": [7, 4], "iou_threshold": 0.3, "scores": [7, 6, 5, 4, 3, 2, 1], "x_dtype": np.float32},
]
# fmt: on


@ddt
class TestNMS(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [4, 4]
        self.x_dtype = np.float32
        self.iou_threshold = 0.1

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        scores = paddle.to_tensor(self.x_scores, dtype=np.int32)
        return paddle.vision.ops.nms(x, self.iou_threshold, scores)

    def expect_output(self):
        return self.calc_result(self.forward, "cpu")

    @data(*NMS_CASE)
    @unpack
    def test_check_output(self, x_shape, iou_threshold, scores, x_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.x_scores = scores
        self.iou_threshold = iou_threshold
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
