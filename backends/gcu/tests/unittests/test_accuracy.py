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
ACCURACY_CASE = [
    {"num_samples": 60, "class_dim": 100, "x_dtype": np.float32, "label_dtype": np.int64, "k": 5},
    {"num_samples": 8192, "class_dim": 1000, "x_dtype": np.float32, "label_dtype": np.int64, "k": 5},

    {"num_samples": 60, "class_dim": 100, "x_dtype": np.float32, "label_dtype": np.int64, "k": 1},
    {"num_samples": 8192, "class_dim": 1000, "x_dtype": np.float32, "label_dtype": np.int64, "k": 1},

    {"num_samples": 60, "class_dim": 100, "x_dtype": np.float16, "label_dtype": np.int64, "k": 5},
    {"num_samples": 8192, "class_dim": 1000, "x_dtype": np.float16, "label_dtype": np.int64, "k": 5},

]
# fmt: on


def accuracy_wrapper(infer, indices, label):
    return paddle._C_ops.accuracy(infer, indices, label)


@ddt
class TestAccuracy(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.num_samples = 60
        self.class_dim = 100
        self.x_dtype = np.float32
        self.label_dtype = np.int32
        self.k = 5
        # self.x_shape = [self.num_samples, self.class_dim]
        # after topk, Maybe there are some bugs in topsatenTopk
        self.x_shape = [self.num_samples, self.k]
        self.indices_shape = [self.num_samples, self.k]
        self.label_shape = [self.num_samples, 1]

    def prepare_data(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        self.data_label = np.random.randint(
            low=0, high=self.class_dim, size=self.label_shape, dtype=self.label_dtype
        )
        self.data_indices = np.random.randint(
            low=0, high=self.class_dim, size=self.indices_shape, dtype=self.label_dtype
        )

    def forward_with_dtype(self, x_dtype, label_dtype):
        x = paddle.to_tensor(self.data_x, dtype=x_dtype)
        label = paddle.to_tensor(self.data_label, dtype=label_dtype)
        # return paddle.static.accuracy(x, label, self.k)
        # after topk
        indices = paddle.to_tensor(self.data_indices, dtype=label_dtype)
        return accuracy_wrapper(x, indices, label)

    def forward(self):
        return self.forward_with_dtype(self.x_dtype, self.label_dtype)

    def input_cast(self):
        return self.forward_with_dtype(np.float32, self.label_dtype)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.forward()
        else:
            out = self.input_cast()
        return out

    @data(*ACCURACY_CASE)
    @unpack
    def test_check_output(self, num_samples, class_dim, x_dtype, label_dtype, k):
        self.num_samples = num_samples
        self.class_dim = class_dim
        self.x_dtype = x_dtype
        self.label_dtype = label_dtype
        self.k = k
        # self.x_shape = [self.num_samples, self.class_dim]
        # after topk, Maybe there are some bugs in topsatenTopk
        self.x_shape = [self.num_samples, self.k]
        self.indices_shape = [self.num_samples, self.k]
        self.label_shape = [self.num_samples, 1]
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
