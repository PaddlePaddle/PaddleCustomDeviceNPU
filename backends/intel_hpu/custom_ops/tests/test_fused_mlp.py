#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
import paddlenlp_ops
from tests.op_test import convert_float_to_uint16, convert_uint16_to_float
from tests.op_test import skip_check_grad_ci


@skip_check_grad_ci(reason="fused_mlp_forward ops not support gradient calculation.")
class TestFusedMlpFP16(unittest.TestCase):
    def setUp(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.batch_size = 8
        self.seqence_len = 128
        self.hidden_size = 256
        self.intermediate_size = 512
        self.init_dtype()
        self.init_weight()

    def init_dtype(self):
        self.dtype = "float16"

    def init_weight(self):
        with paddle.no_grad():
            self.gate_weight = paddle.normal(
                mean=0.0, std=0.02, shape=[self.hidden_size, self.intermediate_size]
            ).astype(self.dtype)
            self.up_weight = paddle.normal(
                mean=1.0, std=0.05, shape=[self.hidden_size, self.intermediate_size]
            ).astype(self.dtype)
            self.down_weight = paddle.normal(
                mean=0.5, std=0.12, shape=[self.intermediate_size, self.hidden_size]
            ).astype(self.dtype)
            self.proj_weight = paddle.concat([self.gate_weight, self.up_weight], axis=1)

    def check_result(self, golden_res, fused_res):
        if self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-3
        elif self.dtype == "bfloat16":
            rtol = 8e-3
            atol = 8e-3
        elif self.dtype == "float32":
            rtol = 1e-5
            atol = 1e-5
        else:
            self.assertTrue(
                False,
                msg="FusedMlp input dtype only supports bfloat16, \
                     float16,float32, but got "
                + self.dtype,
            )
        np.testing.assert_allclose(golden_res, fused_res, rtol=rtol, atol=atol)

    @staticmethod
    def swiglu_naive(x, up=None):
        if up is not None:
            gate = x
        else:
            gate, up = paddle.chunk(x, chunks=2, axis=-1)
        silu = gate / (paddle.exp(-gate) + 1)
        return silu * up

    def golden_mlp(self, x):
        gate = paddle.matmul(x, self.gate_weight)
        up = paddle.matmul(x, self.up_weight)
        swiglu = self.swiglu_naive(x=gate, up=up)
        res = paddle.matmul(swiglu, self.down_weight)

        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()

    def fused_mlp(self, x):
        res = paddlenlp_ops.fused_mlp(
            x, self.gate_weight, self.up_weight, self.down_weight
        )
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()

    def gen_input(self):
        x = np.random.randn(self.batch_size, self.seqence_len, self.hidden_size)
        return x


@skip_check_grad_ci(reason="fused_mlp_forward ops not support gradient calculation.")
class TestFusedMlpBFP16(TestFusedMlpFP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

    def test_fused_mlp(self):
        np_x = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
        golden_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        fused_res = self.fused_mlp(fused_x)
        golden_res = self.golden_mlp(golden_x)
        self.check_result(golden_res, fused_res)


@skip_check_grad_ci(reason="fused_mlp_forward ops not support gradient calculation.")
class TestFusedMlpFP32(TestFusedMlpBFP16):
    def init_dtype(self):
        self.dtype = "float32"


@skip_check_grad_ci(reason="fused_mlp_forward ops not support gradient calculation.")
class TestFusedGateUpMlpFP16(TestFusedMlpFP16):
    def fused_mlp(self, x):
        res = paddlenlp_ops.fused_mlp(x, self.proj_weight, None, self.down_weight)
        if self.dtype == "bfloat16":
            res = convert_uint16_to_float(res.numpy())
            return res
        return res.numpy()


@skip_check_grad_ci(reason="fused_mlp_forward ops not support gradient calculation.")
class TestFusedGateUpMlpBFP16(TestFusedGateUpMlpFP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

    def test_fused_gateup_mlp(self):
        np_x = self.gen_input()
        if self.dtype == "bfloat16":
            np_x = convert_float_to_uint16(np_x)
        golden_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        fused_res = self.fused_mlp(fused_x)
        golden_res = self.golden_mlp(golden_x)
        self.check_result(golden_res, fused_res)


@skip_check_grad_ci(reason="fused_mlp_forward ops not support gradient calculation.")
class TestFusedGateUpMlpFP32(TestFusedGateUpMlpBFP16):
    def init_dtype(self):
        self.dtype = "float32"


if __name__ == "__main__":
    np.random.seed(2024)
    unittest.main()
