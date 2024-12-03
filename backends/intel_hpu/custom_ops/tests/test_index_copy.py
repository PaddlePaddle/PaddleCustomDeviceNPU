#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import torch
import paddle
import paddlenlp_ops

from tests.op_test import skip_check_grad_ci


def index_copy_torch(input, dim, index, source, dtype):
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
    }
    torch_dtype = dtype_map[dtype]
    input_tensor = torch.tensor(input).clone().detach().to(dtype=torch_dtype)
    index_tensor = torch.tensor(index).clone().detach().to(dtype=torch.int64)
    source_tensor = torch.tensor(source).clone().detach().to(dtype=torch_dtype)
    output = torch.index_copy(
        input=input_tensor, dim=dim, index=index_tensor, source=source_tensor
    )
    return output


@skip_check_grad_ci(reason="index_copy_forward ops not support gradient calculation.")
class TestIndexCopyOpFP32(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.init_dtype()
        self.batch_size = 16
        self.num_heads = 32
        self.seq_length = 256
        self.head_dim = 64

    def init_dtype(self):
        self.dtype = "float32"

    def check_result(self, torch_res, ops_res):
        if self.dtype == "float32":
            rtol = 1e-5
            atol = 1e-6
        elif self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-4
        elif self.dtype == "bfloat16":
            rtol = 1e-2
            atol = 1e-3
        else:
            self.assertTrue(
                False,
                msg="index_copy input dtype only supports bfloat16, \
                     float16 and float32, but got "
                + self.dtype,
            )
        np.testing.assert_allclose(torch_res, ops_res, rtol=rtol, atol=atol)

    def index_copy_custom(self, input, dim, index, source):
        input_tensor = paddle.to_tensor(input, dtype=self.dtype).clone()
        index_tensor = paddle.to_tensor(index, dtype="int64").clone()
        source_tensor = paddle.to_tensor(source, dtype=self.dtype).clone()
        paddlenlp_ops.index_copy(
            input=input_tensor, dim=dim, index=index_tensor, source=source_tensor
        )
        return input_tensor

    def prepare_input(
        self, batch_size=16, num_heads=32, seq_length=256, head_dim=64, dim=0, index=0
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim

        input = np.full(
            (num_heads, head_dim, seq_length, batch_size), -1, dtype=self.dtype
        )
        index = [index]
        if dim == 0:
            source = np.full(
                (len(index), head_dim, seq_length, batch_size), 0, dtype=self.dtype
            )
        elif dim == 1:
            source = np.full(
                (num_heads, len(index), seq_length, batch_size), 0, dtype=self.dtype
            )
        elif dim == 2:
            source = np.full(
                (num_heads, head_dim, len(index), batch_size), 0, dtype=self.dtype
            )
        elif dim == 3:
            source = np.full(
                (num_heads, head_dim, seq_length, len(index)), 0, dtype=self.dtype
            )
        else:
            raise ValueError(
                "Unsupported dimension. Only dim=0, dim=1 and dim=2 are supported."
            )
        return input, index, source, dim

    def test_index_copy_dim0_index0(self):
        input, index, source, dim = self.prepare_input(dim=0, index=0)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res)

    def test_index_copy_dim0_index1(self):
        input, index, source, dim = self.prepare_input(dim=0, index=1)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res)

    def test_index_copy_dim0_index_max(self):
        index = max(self.num_heads - 1, 0)
        input, index, source, dim = self.prepare_input(dim=0, index=index)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res)

    def test_index_copy_dim1_index0(self):
        input, index, source, dim = self.prepare_input(dim=1, index=0)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res)

    def test_index_copy_dim1_index1(self):
        input, index, source, dim = self.prepare_input(dim=1, index=1)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim1_index_max(self):
        index = max(self.head_dim - 1, 0)
        input, index, source, dim = self.prepare_input(dim=1, index=index)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim2_index0(self):
        input, index, source, dim = self.prepare_input(dim=2, index=0)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim2_index1(self):
        input, index, source, dim = self.prepare_input(dim=2, index=1)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim2_index_max(self):
        index = max(self.seq_length - 1, 0)
        input, index, source, dim = self.prepare_input(dim=2, index=index)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim3_index0(self):
        input, index, source, dim = self.prepare_input(dim=3, index=0)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim3_index1(self):
        input, index, source, dim = self.prepare_input(dim=3, index=1)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())

    def test_index_copy_dim3_index_max(self):
        index = max(self.batch_size - 1, 0)
        input, index, source, dim = self.prepare_input(dim=3, index=index)
        custom_res = self.index_copy_custom(input, dim, index, source)
        torch_res = index_copy_torch(input, dim, index, source, dtype=self.dtype)
        self.check_result(torch_res.numpy(), custom_res.numpy())


@skip_check_grad_ci(reason="index_copy_forward ops not support gradient calculation.")
class TestIndexCopyOpFP16(TestIndexCopyOpFP32):
    def init_dtype(self):
        self.dtype = "float16"


if __name__ == "__main__":
    unittest.main()
