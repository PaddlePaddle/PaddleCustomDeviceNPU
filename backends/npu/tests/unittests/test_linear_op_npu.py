#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from npu_utils import check_soc_version
from tests.op_test import convert_float_to_uint16, convert_uint16_to_float


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.input = np.random.random((3, 1, 2)).astype(self.dtype)
        self.weight = np.random.random((2, 2)).astype(self.dtype)
        self.bias = np.random.random(2).astype(self.dtype)
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def functional(self):
        input = paddle.to_tensor(self.input)
        weight = paddle.to_tensor(self.weight)
        bias = paddle.to_tensor(self.bias)
        out = F.linear(input, weight, bias)
        return out.numpy()

    def numpy_cal(self):
        res = np.matmul(self.input, self.weight) + self.bias
        return res

    def test_error(self):
        res_f = self.functional()
        res_np = self.numpy_cal()
        np.testing.assert_allclose(res_f, res_np, rtol=4e-3)

    def test_weight(self):
        paddle.seed(100)
        weight_attr = paddle.nn.initializer.Constant(value=0.5)
        paddle.set_device("cpu")
        linear_1 = paddle.nn.Linear(2, 3, weight_attr=weight_attr)
        paddle.nn.utils._stride_column(linear_1.weight)
        except_1 = linear_1.weight.numpy()

        linear_2 = paddle.nn.Linear(2, 3, weight_attr=weight_attr)
        except_2 = linear_2.weight.numpy()

        paddle.set_device("npu:0")
        linear_1 = paddle.nn.Linear(2, 3, weight_attr=weight_attr)
        paddle.nn.utils._stride_column(linear_1.weight)
        np.testing.assert_allclose(linear_1.weight.numpy(), except_1, rtol=1e-05)

        linear_2 = paddle.nn.Linear(2, 3, weight_attr=weight_attr)
        np.testing.assert_allclose(linear_2.weight.numpy(), except_2, rtol=1e-05)


class LinearTestFP16(LinearTestCase):
    def init_dtype(self):
        self.dtype = np.float16


class LinearTestBF16(LinearTestCase):
    def setUp(self):
        self.init_dtype()
        self.input = convert_float_to_uint16(
            np.random.random((3, 1, 2)).astype(self.dtype)
        )
        self.weight = convert_float_to_uint16(
            np.random.random((2, 2)).astype(self.dtype)
        )
        self.bias = convert_float_to_uint16(np.random.random(2).astype(self.dtype))
        self.place = paddle.CustomPlace("npu", 0)

    def functional(self):
        input = paddle.to_tensor(self.input)
        weight = paddle.to_tensor(self.weight)
        bias = paddle.to_tensor(self.bias)
        out = F.linear(input, weight, bias).astype("float32")
        return out.numpy()

    def numpy_cal(self):
        res = np.matmul(
            convert_uint16_to_float(self.input), convert_uint16_to_float(self.weight)
        ) + convert_uint16_to_float(self.bias)
        return res

    @check_soc_version
    def test_error(self):
        res_f = self.functional()
        res_np = self.numpy_cal()
        np.testing.assert_allclose(res_f, res_np, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
