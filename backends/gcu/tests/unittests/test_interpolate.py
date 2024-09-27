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
INTERPOLATE_NEAREST_CASE = [
    # nearest
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "nearest", "align_corners": False, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 4, 4, 3], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "nearest", "align_corners": False, "align_mode": 0, "data_format": "NHWC"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "nearest", "align_corners": False, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": None, "scale_factor": [2, 2], "mode": "nearest", "align_corners": False, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 5, 10], "x_dtype": np.float32, "size": None, "scale_factor": [3, 5], "mode": "nearest", "align_corners": False, "align_mode": 1, "data_format": "NCHW"},

    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float16, "size": [8, 8], "scale_factor": None, "mode": "nearest", "align_corners": False, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 4, 4, 3], "x_dtype": np.float16, "size": [8, 8], "scale_factor": None, "mode": "nearest", "align_corners": False, "align_mode": 0, "data_format": "NHWC"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float16, "size": [8, 8], "scale_factor": None, "mode": "nearest", "align_corners": False, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float16, "size": None, "scale_factor": [2, 2], "mode": "nearest", "align_corners": False, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 5, 10], "x_dtype": np.float16, "size": None, "scale_factor": [3, 5], "mode": "nearest", "align_corners": False, "align_mode": 1, "data_format": "NCHW"},

    # bilinear Interpolate bilinear AOT kernel is unimplemented for align_corners(false) and align_mode(1)
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": False, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 4, 4, 3], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": False, "align_mode": 0, "data_format": "NHWC"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": False, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": None, "scale_factor": [2, 2], "mode": "bilinear", "align_corners": False, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 3, 5, 10], "x_dtype": np.float32, "size": None, "scale_factor": [3, 5], "mode": "bilinear", "align_corners": False, "align_mode": 0, "data_format": "NCHW"},

    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": True, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 4, 4, 3], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": True, "align_mode": 0, "data_format": "NHWC"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": True, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float32, "size": None, "scale_factor": [2, 2], "mode": "bilinear", "align_corners": True, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 5, 10], "x_dtype": np.float32, "size": None, "scale_factor": [3, 5], "mode": "bilinear", "align_corners": True, "align_mode": 1, "data_format": "NCHW"},

    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float16, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": True, "align_mode": 0, "data_format": "NCHW"},
    {"x_shape": [1, 4, 4, 3], "x_dtype": np.float16, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": True, "align_mode": 0, "data_format": "NHWC"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float16, "size": [8, 8], "scale_factor": None, "mode": "bilinear", "align_corners": True, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 4, 4], "x_dtype": np.float16, "size": None, "scale_factor": [2, 2], "mode": "bilinear", "align_corners": True, "align_mode": 1, "data_format": "NCHW"},
    {"x_shape": [1, 3, 5, 10], "x_dtype": np.float16, "size": None, "scale_factor": [3, 5], "mode": "bilinear", "align_corners": True, "align_mode": 1, "data_format": "NCHW"},

]
# fmt: on


@ddt
class TestNearestInterpolate(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [1, 3, 4, 4]
        self.x_dtype = np.float32
        self.size = None
        self.scale_factor = None
        self.mode = "nearest"
        self.align_corners = False
        self.align_mode = 0
        self.data_format = "NCHW"

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format,
        )

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            origin_dtype = self.x_dtype
            self.x_dtype = np.float32
            out = self.calc_result(self.forward, "cpu")
            out = out.astype("float16")
            self.x_dtype = origin_dtype
        return out

    @data(*INTERPOLATE_NEAREST_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        x_dtype,
        size,
        scale_factor,
        mode,
        align_corners,
        align_mode,
        data_format,
    ):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
