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
POOL2D_CASE = [
    # **********  avg_pool2d  ***********
    # for ppocr v2 det
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": False, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 9, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 18, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 240, 168], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 128, 120, 84], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 256, 60, 42], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    # {"x_shape": [1, 256, 65, 55], "x_dtype": np.float32, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},

    # for ppocr v2 det amp
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": False, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 9, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 18, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 240, 168], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 128, 120, 84], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 256, 60, 42], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    # {"x_shape": [1, 256, 65, 55], "x_dtype": np.float16, "pooling_type": 'avg', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},

    # **********  max_pool2d  ***********
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": False, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 9, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 18, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 240, 168], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 128, 120, 84], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 256, 60, 42], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 256, 65, 55], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},

    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": False, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 9, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 480, 336], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1], "ceil_mode": False, "exclusive": True, "divisor_override": 18, "data_format": 'NCHW'},
    {"x_shape": [1, 64, 240, 168], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 128, 120, 84], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 256, 60, 42], "x_dtype": np.float16, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
    {"x_shape": [1, 256, 65, 55], "x_dtype": np.float32, "pooling_type": 'max', "kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True, "exclusive": True, "divisor_override": None, "data_format": 'NCHW'},
]

ADAPTIVE_POOL2D_CASE = [

    # Only support 'output_size = [1, 1]' now
    {"x_shape": [126, 256, 12, 160], "x_dtype": np.float32, "output_size": [1, 1], "data_format": 'NCHW'},
    {"x_shape": [126, 512, 12, 80], "x_dtype": np.float32, "output_size": [1, 1], "data_format": 'NCHW'},
    {"x_shape": [126, 768, 6, 80], "x_dtype": np.float32, "output_size": [1, 1], "data_format": 'NCHW'},

    {"x_shape": [126, 256, 12, 160], "x_dtype": np.float16, "output_size": [1, 1], "data_format": 'NCHW'},
    {"x_shape": [126, 512, 12, 80], "x_dtype": np.float16, "output_size": [1, 1], "data_format": 'NCHW'},
    {"x_shape": [126, 768, 6, 80], "x_dtype": np.float16, "output_size": [1, 1], "data_format": 'NCHW'},

]
# fmt: on


@ddt
class TestAvgPool2d(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [1, 64, 480, 336]
        self.x_dtype = np.float32
        self.pooling_type = "avg"
        self.kernel_size = [3, 3]
        self.stride = [2, 2]
        self.padding = [1, 1]
        self.ceil_mode = False
        self.exclusive = True
        self.divisor_override = None
        self.data_format = "NCHW"
        self.return_mask = False

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        if self.pooling_type == "avg":
            return paddle.nn.functional.avg_pool2d(
                x=x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                exclusive=self.exclusive,
                divisor_override=self.divisor_override,
                data_format=self.data_format,
            )
        elif self.pooling_type == "max":
            return paddle.nn.functional.max_pool2d(
                x=x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                return_mask=self.return_mask,
                data_format=self.data_format,
            )

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def pool2d_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.pool2d_cast()
        return out

    @data(*POOL2D_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        x_dtype,
        pooling_type,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        exclusive,
        divisor_override,
        data_format,
    ):
        self.x_shape = x_shape
        self.x_dtype = np.float32
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.exclusive = exclusive
        self.divisor_override = divisor_override
        self.data_format = data_format
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


@ddt
class TestAdaptiveAvgPool2d(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [126, 256, 12, 160]
        self.x_dtype = np.float32
        self.output_size = [1, 1]
        self.data_format = "NCHW"

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        return paddle.nn.functional.adaptive_avg_pool2d(
            x=x, output_size=self.output_size, data_format=self.data_format
        )

    def forward(self):
        return self.forward_with_dtype(self.x_dtype)

    def adaptive_avg_pool2d_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.adaptive_avg_pool2d_cast()
        return out

    @data(*ADAPTIVE_POOL2D_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, output_size, data_format):
        self.x_shape = x_shape
        self.x_dtype = np.float32
        self.output_size = output_size
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
