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
CONV_2D_TRANSPOSE_CASE = [
    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 4, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NCHW"},
    {"x_shape": [1, 8, 8, 2], "weight_shape": [2, 4, 3, 3], "dtype": np.float32, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NHWC"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NCHW"},
    {"x_shape": [2, 32, 32, 3], "weight_shape": [3, 6, 3, 3], "dtype": np.float32, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NHWC"},

    {"x_shape": [1, 2, 8, 8], "weight_shape": [2, 4, 3, 3], "dtype": np.float16, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NCHW"},
    {"x_shape": [1, 8, 8, 2], "weight_shape": [2, 4, 3, 3], "dtype": np.float16, "stride": [1, 1], "padding": [0, 0], "output_padding": [0, 0], "dilation": [1, 1], "groups": 1, "data_format": "NHWC"},
    {"x_shape": [2, 3, 32, 32], "weight_shape": [3, 6, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NCHW"},
    {"x_shape": [2, 32, 32, 3], "weight_shape": [3, 6, 3, 3], "dtype": np.float16, "stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1], "groups": 1, "data_format": "NHWC"},

]

# The aten operator library does not support conv3d_transpose
CONV_3D_TRANSPOSE_CASE = [
    # {"x_shape": [1, 2, 8, 8, 8], "weight_shape": [2, 4, 3, 3, 3], "dtype": np.float32, "stride": [1, 1, 1], "padding": [0, 0, 0], "output_padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1, "data_format": "NCDHW"},
    # # {"x_shape": [1, 8, 8, 8, 2], "weight_shape": [2, 4, 3, 3, 3], "dtype": np.float32, "stride": [1, 1, 1], "padding": [0, 0, 0], "output_padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1, "data_format": "NDHWC"},
    # {"x_shape": [2, 3, 8, 32, 32], "weight_shape": [3, 6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2, 2], "padding": [1, 1, 1], "output_padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1, "data_format": "NCDHW"},
    # # {"x_shape": [2, 8, 32, 32, 3], "weight_shape": [3, 6, 3, 3, 3], "dtype": np.float32, "stride": [2, 2, 2], "padding": [1, 1, 1], "output_padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1, "data_format": "NDHWC"},

    # {"x_shape": [1, 2, 8, 8, 8], "weight_shape": [2, 4, 3, 3, 3], "dtype": np.float16, "stride": [1, 1, 1], "padding": [0, 0, 0], "output_padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1, "data_format": "NCDHW"},
    # # {"x_shape": [1, 8, 8, 8, 2], "weight_shape": [2, 4, 3, 3, 3], "dtype": np.float16, "stride": [1, 1, 1], "padding": [0, 0, 0], "output_padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1, "data_format": "NDHWC"},
    # {"x_shape": [2, 3, 8, 32, 32], "weight_shape": [3, 6, 3, 3, 3], "dtype": np.float16, "stride": [2, 2, 2], "padding": [1, 1, 1], "output_padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1, "data_format": "NCDHW"},
    # # {"x_shape": [2, 8, 32, 32, 3], "weight_shape": [3, 6, 3, 3, 3], "dtype": np.float16, "stride": [2, 2, 2], "padding": [1, 1, 1], "output_padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1, "data_format": "NDHWC"},
    # {"x_shape": [2, 3, 8, 8, 8], "weight_shape": [3, 6, 3, 3, 3], "dtype": np.float32, "stride": [1, 1, 1], "padding": [0, 0, 0], "output_padding": [0, 0, 0], "dilation": [1, 1, 1], "groups": 1, "data_format": "NCDHW"},
]
# fmt: on


@ddt
class TestConv2dTranspose(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.weight_shape = [6, 3, 3, 3]
        self.dtype = np.float32
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.output_padding = [0, 0]
        self.groups = 1
        self.dilation = [1, 1]
        self.data_format = "NCHW"
        self.output_size = None

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.weight_shape, self.dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        w = paddle.to_tensor(self.data_w, dtype=dtype)
        return paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=w,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
            data_format=self.data_format,
            output_size=self.output_size,
        )

    def forward(self):
        return self.forward_with_dtype(self.dtype)

    def conv_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.forward()
        else:
            out = self.conv_cast()
        return out

    @data(*CONV_2D_TRANSPOSE_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        weight_shape,
        dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        data_format,
    ):
        self.x_shape = x_shape
        self.weight_shape = weight_shape
        self.dtype = dtype
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.data_format = data_format
        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


@ddt
class TestConv3dTranspose(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 8, 32, 32]
        self.weight_shape = [6, 3, 3, 3, 3]
        self.dtype = np.float32
        self.stride = [1, 1, 1]
        self.padding = [0, 0, 0]
        self.output_padding = [0, 0, 0]
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.output_size = None
        self.data_format = "NCDHW"

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.dtype)
        self.data_w = self.generate_data(self.weight_shape, self.dtype)

    def forward_with_dtype(self, dtype):
        x = paddle.to_tensor(self.data_x, dtype=dtype)
        w = paddle.to_tensor(self.data_w, dtype=dtype)
        return paddle.nn.functional.conv3d_transpose(
            x=x,
            weight=w,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
            output_size=self.output_size,
            data_format=self.data_format,
        )

    def forward(self):
        return self.forward_with_dtype(self.dtype)

    def conv_cast(self):
        return self.forward_with_dtype(np.float32).astype("float16")

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.forward()
        else:
            out = self.conv_cast()
        return out

    @data(*CONV_3D_TRANSPOSE_CASE)
    @unpack
    def test_check_output(
        self,
        x_shape,
        weight_shape,
        dtype,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        data_format,
    ):
        self.x_shape = x_shape
        self.weight_shape = weight_shape
        self.dtype = dtype
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.data_format = data_format
        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
