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
from scipy.special import erf
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
UNARY_CASE = [
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.float32},
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.int32},
    {"x_shape": [2, 4, 4], "x_dtype": np.float32},
    {"x_shape": [2, 4, 4], "x_dtype": np.int32},
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.float16},
    {"x_shape": [2, 4, 4], "x_dtype": np.float16},
]
# fmt: on


@ddt
class TestUnary(TestAPIBase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-5
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.x_dtype = np.float32
        self.support_dtype = [np.float32, np.float16, np.int32]

    def prepare_data(self):
        self.init_api_and_data()

    def init_api_and_data(self):
        self.unary_api = paddle.log
        self.data_x = np.random.uniform(1, 2, self.x_shape).astype(self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x)

    def get_numpy_out(self):
        return np.log(self.data_x)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_out()
        return out

    @data(*UNARY_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        if x_dtype not in self.support_dtype:
            return
        rtol = self.rtol
        atol = self.atol
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


class TestCos(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.cos
        self.data_x = np.random.uniform(0.1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.cos(self.data_x)


class TestRsqrt(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float64]

    def init_api_and_data(self):
        self.unary_api = paddle.rsqrt
        self.data_x = np.random.uniform(0.1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return 1.0 / np.sqrt(self.data_x)


class TestSilu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.silu
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return self.data_x / (np.exp(-self.data_x) + 1)


class TestSin(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.sin
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.sin(self.data_x)


class TestTril(TestUnary):
    def init_api_and_data(self):
        self.unary_api = paddle.tril
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.tril(self.data_x)


class TestTriu(TestUnary):
    def init_api_and_data(self):
        self.unary_api = paddle.triu
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.triu(self.data_x)


class TestPow(TestUnary):
    def init_api_and_data(self):
        self.unary_api = paddle.pow
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, 3)

    def get_numpy_out(self):
        return np.power(self.data_x, 3)


class TestRelu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.relu
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.maximum(0, self.data_x)


class TestRelu6(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.relu6
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.minimum(np.maximum(0, self.data_x), 6.0)


class TestSwish(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.swish
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return self.data_x / (np.exp(-self.data_x) + 1)


class TestSigmoid(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.sigmoid
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return 1 / (np.exp(-self.data_x) + 1)


class TestLogSigmoid(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.log_sigmoid
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.log(1 / (1 + np.exp(-self.data_x)))


class TestHardSigmoid(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.hardsigmoid
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.clip(self.data_x / 6.0 + 0.5, 0, 1)


class TestHardSwish(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.hardswish
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        r = np.array(self.data_x <= -3)
        tmp = self.data_x * (self.data_x + 3) / 6.0
        zeros = np.zeros(self.data_x.shape, self.data_x.dtype)
        tmp_1 = np.select(r, zeros, tmp)
        tmp_2 = np.select(np.array(self.data_x >= 3), self.data_x, tmp_1)
        return tmp_2


class TestAbs(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.abs
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.absolute(self.data_x)


class TestAtan(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.atan
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.arctan(self.data_x)


class TestExp(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.exp
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.exp(self.data_x)


class TestFloor(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.floor
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.floor(self.data_x)


class TestCeil(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.ceil
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.ceil(self.data_x)


class TestGelu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]
        # np.float32 accuracy does not meet requirements(1e-5), only 1e-4
        self.rtol = 1e-4
        self.atol = 1e-4

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.gelu
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return [self.unary_api(x, True), self.unary_api(x, False)]

    def get_numpy_out(self):
        # x = paddle.to_tensor(self.data_x, dtype=np.float32)
        # def calc():
        #     return [self.unary_api(x, True), self.unary_api(x, False)]
        # result = self.calc_result(calc, "cpu")
        # return [result[0].astype("float16"), result[1].astype("float16")]
        def gelu(x, approximate):
            if approximate:
                y = (
                    0.5
                    * x
                    * (
                        1.0
                        + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
                    )
                )
            else:
                y = 0.5 * x * (1 + erf(x / np.sqrt(2)))
            return y

        return [gelu(self.data_x, True), gelu(self.data_x, False)]


class TestLeakyRelu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.leaky_relu
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, 0.02)

    def get_numpy_out(self):
        def ref_leaky_relu(x, alpha=0.01):
            out = np.copy(x)
            out[out < 0] *= alpha
            return out

        return ref_leaky_relu(self.data_x, 0.02)


class TestSquare(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.square
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.square(self.data_x)


class TestSqrt(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.sqrt
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.sqrt(self.data_x)


class TestTanh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.tanh
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.tanh(self.data_x)


class TestSquaredL2Norm(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle._C_ops.squared_l2_norm
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.sum(np.power(self.data_x, 2))


class TestLog2(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.log2
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.log2(self.data_x)


class TestLog10(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.log10
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.log10(self.data_x)


class TestLog1p(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.log1p
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.log1p(self.data_x)


class TestReciprocal(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.reciprocal
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.reciprocal(self.data_x)


class TestLogit(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.logit
        self.eps = 1e-5 + 1e-5 * np.random.uniform(low=-1, high=1)
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.eps)

    def get_numpy_out(self):
        x_min = np.minimum(self.data_x, 1.0 - self.eps)
        x_max = np.maximum(x_min, self.eps)
        return np.log(x_max / (1.0 - x_max))


class TestCelu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.celu
        self.alpha = 1 + np.random.uniform(low=-1, high=1)
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.alpha)

    def get_numpy_out(self):
        out = np.maximum(0, self.data_x) + np.minimum(
            0, self.alpha * (np.exp(self.data_x / self.alpha) - 1)
        )
        return out.astype(self.x_dtype)


class TestHardShrink(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.hardshrink
        self.threshold = 0.5 + np.random.uniform(low=-0.5, high=0.5)
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.threshold)

    def get_numpy_out(self):
        out = np.copy(self.data_x)
        out[(out >= -self.threshold) & (out <= self.threshold)] = 0
        return out


class TestSoftShrink(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.softshrink
        self.lambd = 0.5 + np.random.uniform(low=-0.5, high=0.5)
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.lambd)

    def get_numpy_out(self):
        out = np.copy(self.data_x)
        out = (out < -self.lambd) * (out + self.lambd) + (out > self.lambd) * (
            out - self.lambd
        )
        return out


class TestSoftplus(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.softplus
        self.beta = 1 + np.random.uniform(low=1, high=2)
        self.threshold = 10 + np.random.uniform(low=-10, high=10)
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.beta, self.threshold)

    def get_numpy_out(self):
        x_beta = self.beta * self.data_x
        out = np.select(
            [x_beta <= self.threshold, x_beta > self.threshold],
            [np.log(1 + np.exp(x_beta)) / self.beta, self.data_x],
        )
        return out


class TestAcos(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.acos
        self.data_x = np.random.uniform(-0.95, 0.95, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.arccos(self.data_x)


class TestAcosh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.acosh
        self.data_x = np.random.uniform(2, 3, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.arccosh(self.data_x)


class TestAsin(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.asin
        self.data_x = np.random.uniform(-0.95, 0.95, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.arcsin(self.data_x)


class TestAsinh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.asinh
        self.data_x = np.random.uniform(2, 3, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.arcsinh(self.data_x)


class TestAtanh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.atanh
        self.data_x = np.random.uniform(-0.9, 0.9, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.arctanh(self.data_x)


class TestCosh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.cosh
        self.data_x = np.random.uniform(0.1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.cosh(self.data_x)


class TestSinh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.sinh
        self.data_x = np.random.uniform(0.1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.sinh(self.data_x)


class TestTan(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.tan
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.tan(self.data_x)


class TestRound(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.round
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.round(self.data_x)


class TestElu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.elu
        self.data_x = np.random.uniform(-3, 3, self.x_shape).astype(self.x_dtype)
        self.alpha = 1 + np.random.uniform(low=-1, high=1)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.alpha)

    def get_numpy_out(self):
        out = np.where(
            self.data_x > 0, self.data_x, self.alpha * (np.exp(self.data_x) - 1)
        )
        return out


class TestErf(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]
        # np.float32 accuracy does not meet requirements(1e-5), only 1e-3
        self.rtol = 1e-3
        self.atol = 1e-3

    def init_api_and_data(self):
        self.unary_api = paddle.erf
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return erf(self.data_x).astype(self.x_dtype)


class TestHardtanh(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.hardtanh
        self.min = -10 + np.random.uniform(low=-10, high=10)
        self.max = 10 + np.random.uniform(low=-10, high=10)
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, self.min, self.max)

    def get_numpy_out(self):
        out = np.copy(self.data_x)
        out[np.abs(self.data_x - self.min) < 0.005] = self.min + 0.02
        out[np.abs(self.data_x - self.max) < 0.005] = self.max + 0.02
        out = np.minimum(np.maximum(self.data_x, self.min), self.max)
        return out


class TestExpm1(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.expm1
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.expm1(self.data_x)


if __name__ == "__main__":
    unittest.main()
