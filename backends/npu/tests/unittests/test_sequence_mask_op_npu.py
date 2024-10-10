#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.base.core as core
import paddle.nn.functional as F

paddle.enable_static()


def calc_ground_truth_mask(x, maxlen):
    maxlen = np.max(x) if maxlen < 0 else maxlen
    maxlen = int(maxlen)
    shape = x.shape + (maxlen,)
    index_broadcast = np.broadcast_to(
        np.reshape(range(maxlen), newshape=[1] * x.ndim + [-1]),
        shape=shape,
    )
    x_broadcast = np.broadcast_to(np.reshape(x, newshape=x.shape + (-1,)), shape=shape)
    return (index_broadcast < x_broadcast).astype(np.int64)


class TestSequenceMask(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "sequence_mask"
        self.init_dtype()
        self.x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.set_attrs()

        out = calc_ground_truth_mask(self.x, self.maxlen)

        self.attrs = {
            "maxlen": self.maxlen,
            "out_dtype": int(core.VarDesc.VarType.INT64),
        }
        self.outputs = {"Y": out}

    def set_attrs(self):
        self.maxlen = 6

    def init_dtype(self):
        self.dtype = "int64"

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSequenceMaskINT64(TestSequenceMask):
    def init_dtype(self):
        self.dtype = "int32"


class TestSequenceMaskFP32(TestSequenceMask):
    def init_dtype(self):
        self.dtype = "float32"


class TestSequenceMaskFP64(TestSequenceMask):
    def init_dtype(self):
        self.dtype = "float64"


class TestSequenceMaskMaxlen(TestSequenceMask):
    def set_attrs(self):
        self.maxlen = -1


class TestSequenceMaskMaxlenTensor(TestSequenceMask):
    def set_attrs(self):
        self.max_len_tensor = np.ones((1), "int64") * 10
        self.inputs = {"X": self.x, "MaxLenTensor": self.max_len_tensor}
        self.maxlen = int(self.max_len_tensor)


class TestSequenceMaskAPI(unittest.TestCase):
    # test paddle.nn.functional.sequence_mask
    def setUp(self):
        self.x_np = np.random.uniform(-5, 5, [10, 12]).astype(np.int64)
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            maxlen = 4
            out = F.sequence_mask(x, maxlen)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np, "maxlen": maxlen}, fetch_list=[out])
        out_ref = calc_ground_truth_mask(self.x_np, maxlen)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        maxlen = 4
        out = F.sequence_mask(x, maxlen)
        out_ref = calc_ground_truth_mask(self.x_np, maxlen)
        for r in [out]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
