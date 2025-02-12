# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

from tests.op_test import OpTest
import paddle
import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def topk_ref(x, k=1, axis=-1, largest=True):
    axis = len(x.shape) + axis if axis < 0 else axis
    indices = np.argsort(-x if largest else x, axis=axis)
    value = -np.sort(-x, axis=axis) if largest else np.sort(x, axis=axis)

    indices = indices.take(indices=range(0, k), axis=axis)
    value = value.take(indices=range(0, k), axis=axis)
    return value, indices


class TestTopkHPUOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "top_k_v2"

        self.set_hpu()
        self.set_dtype()
        self.set_input_data()
        self.set_attrs()
        output, indices = topk_ref(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest
        )

        self.inputs = {"X": self.input_data}
        self.attrs = {"k": self.k, "axis": self.axis, "largest": self.largest}
        self.outputs = {"Out": output, "Indices": indices}

    def set_dtype(self):
        self.dtype = np.int32

    def set_attrs(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def set_input_data(self):
        self.input_data = np.random.choice(10000, size=(10, 20), replace=False).astype(
            self.dtype
        )

    def test_check_output(self):
        self.__class__.no_need_check_grad = True
        if self.dtype == np.float32:
            self.check_output_with_place(self.place, atol=1e-3)
        else:
            self.check_output_with_place(self.place)

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))


class TestTopkOpFloat16(TestTopkHPUOp):
    def set_attrs(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def set_dtype(self):
        self.dtype = np.float32

    def set_input_data(self):
        self.input_data = np.random.rand(3, 4).astype(self.dtype)


class TestTopkOP1Int32(TestTopkHPUOp):
    def set_attrs(self):
        self.k = 3
        self.axis = 0
        self.largest = False


class TestTopkOP2Int32(TestTopkHPUOp):
    def set_attrs(self):
        self.k = 4
        self.axis = 0
        self.largest = False


class TestTopkOP3Int32(TestTopkHPUOp):
    def set_attrs(self):
        self.k = 6
        self.axis = 1
        self.largest = True


class TestTopkOP4Int32(TestTopkHPUOp):
    def set_attrs(self):
        self.k = 3
        self.axis = 1
        self.largest = True


class TestTopkOp1Float32(TestTopkOP1Int32):
    def set_dtype(self):
        self.dtype = np.float32

    def set_input_data(self):
        self.input_data = np.random.rand(10, 20).astype(self.dtype)


class TestTopkOp2Float32(TestTopkOP2Int32):
    def set_dtype(self):
        self.dtype = np.float32

    def set_input_data(self):
        self.input_data = np.random.rand(10, 20).astype(self.dtype)


class TestTopkOp3Float32(TestTopkOP3Int32):
    def set_dtype(self):
        self.dtype = np.float32

    def set_input_data(self):
        self.input_data = np.random.rand(10, 20).astype(self.dtype)


class TestTopkOp4Float32(TestTopkOP4Int32):
    def set_dtype(self):
        self.dtype = np.float32

    def set_input_data(self):
        self.input_data = np.random.rand(10, 20).astype(self.dtype)


if __name__ == "__main__":
    unittest.main()
