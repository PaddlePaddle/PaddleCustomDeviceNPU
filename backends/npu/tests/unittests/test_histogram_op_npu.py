#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base


class TestHistogram(unittest.TestCase):
    """Test histogram api."""

    def setUp(self):
        self.init_test_case()
        self.input_np = np.random.uniform(
            low=0.0, high=20.0, size=self.in_shape
        ).astype(np.float32)
        self.weight_np = np.random.uniform(
            low=0.0, high=1.0, size=self.in_shape
        ).astype(np.float32)

    def init_test_case(self):
        self.in_shape = (10, 12)
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = False
        self.is_weight = False

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            inputs = paddle.static.data(
                name="input", dtype="float32", shape=self.in_shape
            )
            if self.is_weight:
                weight = paddle.static.data(
                    name="weight", dtype="float32", shape=self.in_shape
                )
                output = paddle.histogram(
                    inputs,
                    bins=self.bins,
                    min=self.min,
                    max=self.max,
                    weight=weight,
                    density=self.density,
                )
            else:
                output = paddle.histogram(
                    inputs,
                    bins=self.bins,
                    min=self.min,
                    max=self.max,
                    density=self.density,
                )
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            if self.is_weight:
                res = exe.run(
                    feed={
                        "input": self.input_np,
                        "weight": self.weight_np,
                    },
                    fetch_list=[output],
                )
            else:
                res = exe.run(feed={"input": self.input_np}, fetch_list=[output])

            actual = np.array(res[0])
            Out, _ = np.histogram(
                self.input_np,
                bins=self.bins,
                range=(self.min, self.max),
                density=self.density,
                weights=self.weight_np if self.is_weight else None,
            )
            np.testing.assert_allclose(actual, Out, rtol=1e-58, atol=1e-5)
        paddle.disable_static()

    def test_dygraph(self):
        with base.dygraph.guard():
            inputs_np = np.random.uniform(
                low=0.0, high=20.0, size=self.in_shape
            ).astype(np.float32)

            self.inputs = paddle.to_tensor(inputs_np)

            weight_np = np.random.uniform(low=0.0, high=1.0, size=self.in_shape).astype(
                np.float32
            )
            weight = paddle.to_tensor(weight_np)

            actual = paddle.histogram(
                self.inputs,
                bins=5,
                min=1,
                max=5,
                weight=weight if self.is_weight else None,
                density=self.density,
            )

            Out, _ = np.histogram(
                inputs_np,
                bins=5,
                range=(1, 5),
                weights=weight_np if self.is_weight else None,
                density=self.density,
            )

            np.testing.assert_allclose(actual.numpy(), Out, rtol=1e-58, atol=1e-5)


class TestHistogramOp_ZeroDim(TestHistogram):
    def init_test_case(self):
        self.in_shape = []
        self.bins = 5
        self.min = 1
        self.max = 5
        self.density = False
        self.is_weight = False


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
