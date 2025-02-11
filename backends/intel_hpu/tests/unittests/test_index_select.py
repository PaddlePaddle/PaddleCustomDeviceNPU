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

import numpy as np
import unittest

from tests.op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


class TestHPUIndexSelect(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "index_select"
        self.config()

        x_np = np.random.random(self.x_shape).astype(self.x_type)
        index_np = np.random.randint(
            low=0,
            high=self.x_shape[self.dim],
            size=self.index_size,
            dtype=self.index_type,
        )

        # compute real output as baseline.
        outer_loop = np.prod(self.x_shape[: self.dim])
        outer_loop = outer_loop.astype(self.index_type)
        x_reshape = [outer_loop] + list(self.x_shape[self.dim :])
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))

        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                out_list.append(x_np_reshape[i, index_np[j]])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = self.index_size
        self.out_shape = tuple(self.out_shape)
        out = np.reshape(out_list, self.out_shape)

        self.inputs = {"X": x_np, "Index": index_np}
        self.attrs = {"dim": self.dim}
        self.outputs = {"Out": out}

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def config(self):
        self.x_shape = (100, 4, 5)
        self.x_type = np.float32
        self.dim = 1
        self.index_size = 100
        self.index_type = np.int64


class TestHPUIndexSelectCase2(TestHPUIndexSelect):
    def config(self):
        self.dim = -2
        self.x_type = np.float32
        self.index_type = np.int32
        self.x_shape = (10, 10, 4, 10)
        self.index_size = 10


class TestHPUIndexSelectCase3(TestHPUIndexSelect):
    def config(self):
        self.dim = 0
        self.x_type = np.float32
        self.index_type = np.int32
        self.x_shape = (10, 10, 4, 10)
        self.index_size = 10


class TestHPUIndexSelectCase4(TestHPUIndexSelect):
    def config(self):
        self.dim = -1
        self.x_type = np.float32
        self.index_type = np.int32
        self.x_shape = (10, 10, 4, 10)
        self.index_size = 10


class TestHPUIndexSelectDouble(TestHPUIndexSelect):
    def config(self):
        self.x_shape = (100, 4, 5)
        self.x_type = np.double
        self.dim = 1
        self.index_size = 100
        self.index_type = np.int64


if __name__ == "__main__":
    unittest.main()
