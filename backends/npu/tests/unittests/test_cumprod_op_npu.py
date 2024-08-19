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

import paddle
import paddle.base.core as core
import paddle.base as base

paddle.enable_static()


class TestCumprodOp(unittest.TestCase):
    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.cumprod(data)
        z = np.cumprod(data_np)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumprod(data, dim=0)
        z = np.cumprod(data_np, axis=0)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumprod(data, dim=-1)
        z = np.cumprod(data_np, axis=-1)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumprod(data, dtype="float32")
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP32)

        y = paddle.cumprod(data, dtype="int32")
        self.assertTrue(y.dtype == core.VarDesc.VarType.INT32)

        y = paddle.cumprod(data, dim=-2)
        z = np.cumprod(data_np, axis=-2)
        self.assertTrue(np.array_equal(z, y.numpy()))

    def run_static(self, use_custom_device=False):
        with base.program_guard(base.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data("X", [100, 100])
            y = paddle.cumprod(x, dim=0)
            y2 = paddle.cumprod(x, dim=1)
            y3 = paddle.cumprod(x, dim=-1)
            y4 = paddle.cumprod(x, dim=0, dtype="float32")
            y5 = paddle.cumprod(x, dim=0, dtype="int32")
            y6 = paddle.cumprod(x, dim=-2)

            place = base.CustomPlace("npu", 0) if use_custom_device else base.CPUPlace()
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            out = exe.run(
                feed={"X": data_np},
                fetch_list=[y.name, y2.name, y3.name, y4.name, y5.name, y6.name],
            )

            z = np.cumprod(data_np, axis=0)
            self.assertTrue(np.allclose(z, out[0]))
            z = np.cumprod(data_np, axis=1)
            self.assertTrue(np.allclose(z, out[1]))
            z = np.cumprod(data_np, axis=-1)
            self.assertTrue(np.allclose(z, out[2]))
            self.assertTrue(out[3].dtype == np.float32)
            self.assertTrue(out[4].dtype == np.int32)
            z = np.cumprod(data_np, axis=-2)
            self.assertTrue(np.allclose(z, out[5]))

    def test_npu(self):
        # Now, npu tests need setting paddle.enable_static()

        self.run_static(use_custom_device=True)

    def test_name(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data("x", [3, 4])
            y = paddle.cumprod(x, dim=0, name="out")
            self.assertTrue("out" in y.name)


if __name__ == "__main__":
    unittest.main()
