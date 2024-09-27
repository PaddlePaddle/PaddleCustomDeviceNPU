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
import sys
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
CUMULATE_CASE = [
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": None, "out_value_dtype": np.float32, "out_indices_dtype": np.int64},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": None, "out_value_dtype": np.float16, "out_indices_dtype": np.int32},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": None, "out_value_dtype": np.float16, "out_indices_dtype": np.int32},

    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": 0, "out_value_dtype": np.float32, "out_indices_dtype": np.int64},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": 0, "out_value_dtype": np.float16, "out_indices_dtype": np.int32},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": 0, "out_value_dtype": np.float16, "out_indices_dtype": np.int32},

    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": -1, "out_value_dtype": np.float32, "out_indices_dtype": np.int64},
    {"x_shape": [3, 6], "x_dtype": np.float16, "axis": -1, "out_value_dtype": np.float16, "out_indices_dtype": np.int32},
    {"x_shape": [3, 6], "x_dtype": np.float32, "axis": -1, "out_value_dtype": np.float16, "out_indices_dtype": np.int32},
]
# fmt: on


@ddt
class TestCumulateBase(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.x_dtype = np.float32
        self.axis = None
        self.out_value_dtype = np.float32
        self.out_indices_dtype = np.int64
        self.init_test_api()

    def init_test_api(self):
        self.paddle_api = paddle.cumsum
        self.numpy_api = np.cumsum

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        api_dtype = (
            self.out_value_dtype
            if self.paddle_api in [paddle.cumsum, paddle.cumprod]
            else self.out_indices_dtype
        )
        api_axis = (
            -1
            if (self.paddle_api in [paddle.cumprod] and self.axis is None)
            else self.axis
        )
        return self.paddle_api(x, api_axis, dtype=api_dtype)

    def expect_output(self):
        if self.x_dtype != np.float16 and self.out_value_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            data_np = self.data_x.astype(np.float16)
            api_axis = (
                -1
                if (self.paddle_api in [paddle.cumprod] and self.axis is None)
                else self.axis
            )
            out = self.numpy_api(data_np, axis=api_axis)
        return out

    @data(*CUMULATE_CASE)
    @unpack
    def test_check_output(
        self, x_shape, x_dtype, axis, out_value_dtype, out_indices_dtype
    ):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.out_value_dtype = out_value_dtype
        self.out_indices_dtype = out_indices_dtype
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16 or (
            out_value_dtype == np.float16
            and self.paddle_api in [paddle.cumsum, paddle.cumprod]
        ):
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


def cummax_dim2(arr, axis=None):
    if axis is None:
        arr = arr.flatten()
        cummax = np.maximum.accumulate(arr)
        shape = arr.shape
        indices = np.zeros(shape).astype("int32")
        max_val = -sys.maxsize
        max_ind = 0
        for i in range(shape[0]):
            if arr[i] >= max_val:
                max_val = max(arr[i], max_val)
                max_ind = i
                indices[i] = i
            else:
                indices[i] = max_ind
    else:
        cummax = np.maximum.accumulate(arr, axis)
        shape = arr.shape
        indices = np.zeros(shape).astype("int32")
        if axis < 0:
            axis = axis + len(shape)
        if axis == 0:
            for j in range(shape[1]):
                max_ind = 0
                max_val = -sys.maxsize
                for i in range(shape[0]):
                    if arr[i][j] >= max_val:
                        max_val = arr[i][j]
                        max_ind = i
                        indices[i][j] = i
                    else:
                        indices[i][j] = max_ind
        elif axis == 1:
            for i in range(shape[0]):
                max_ind = 0
                max_val = -sys.maxsize
                for j in range(shape[1]):
                    if arr[i][j] >= max_val:
                        max_val = arr[i][j]
                        max_ind = j
                        indices[i][j] = j
                    else:
                        indices[i][j] = max_ind
        else:
            raise Exception("unfeasible axis")
    return cummax, indices


def cummin_dim2(arr, axis=None):
    if axis is None:
        arr = arr.flatten()
        cummin = np.minimum.accumulate(arr)
        shape = arr.shape
        indices = np.zeros(shape).astype("int32")
        min_val = sys.maxsize
        min_ind = 0
        for i in range(shape[0]):
            if arr[i] <= min_val:
                min_val = min(arr[i], min_val)
                min_ind = i
                indices[i] = i
            else:
                indices[i] = min_ind
    else:
        cummin = np.minimum.accumulate(arr, axis)
        shape = arr.shape
        indices = np.zeros(shape).astype("int32")
        if axis < 0:
            axis = axis + len(shape)
        if axis == 0:
            for j in range(shape[1]):
                min_ind = 0
                min_val = sys.maxsize
                for i in range(shape[0]):
                    if arr[i][j] <= min_val:
                        min_val = arr[i][j]
                        min_ind = i
                        indices[i][j] = i
                    else:
                        indices[i][j] = min_ind
        elif axis == 1:
            for i in range(shape[0]):
                min_ind = 0
                min_val = sys.maxsize
                for j in range(shape[1]):
                    if arr[i][j] <= min_val:
                        min_val = arr[i][j]
                        min_ind = j
                        indices[i][j] = j
                    else:
                        indices[i][j] = min_ind
        else:
            raise Exception("unfeasible axis")
    return cummin, indices


class TestCumsum(TestCumulateBase):
    def init_test_api(self):
        self.paddle_api = paddle.cumsum
        self.numpy_api = np.cumsum


# TODO: Kernel of topsatenCummax is unimplemented.
# class TestCummax(TestCumulateBase):
#     def init_test_api(self):
#         self.paddle_api = paddle.cummax
#         self.numpy_api = cummax_dim2

# TODO: Kernel of topsatenCummin is unimplemented.
# class TestCummin(TestCumulateBase):
#     def init_test_api(self):
#         self.paddle_api = paddle.cummin
#         self.numpy_api = cummin_dim2


class TestCumprod(TestCumulateBase):
    def init_test_api(self):
        self.paddle_api = paddle.cumprod
        self.numpy_api = np.cumprod


if __name__ == "__main__":
    unittest.main()
