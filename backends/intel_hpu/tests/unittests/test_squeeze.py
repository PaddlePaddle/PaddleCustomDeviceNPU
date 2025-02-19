#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# paddle.enable_static()

import os
from util import enable_paddle_static_mode

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
intel_hpus_static_mode = os.environ.get(
    "FLAGS_static_mode_intel_hpus", 0
)  # default is dynamic mode test FLAGS_static_mode_intel_hpus=0


# Correct: General.
class TestSqueeze2Op(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "squeeze2"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        enable_paddle_static_mode(int(intel_hpus_static_mode))

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0,)
        self.new_shape = (3, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


"""
# Correct: There is mins axis.
class TestSqueeze2Op1(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)
"""


# Correct: No axes input.
class TestSqueeze2Op1(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (2,)
        self.new_shape = (1, 20, 5)


# Correct: No axes input.
class TestSqueeze2Op2(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: No axes input.
class TestSqueeze2Op3(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (-2,)
        self.new_shape = (1, 20, 5)


"""
# Correct: Just part of axes be squeezed.
class TestSqueeze2Op3(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)
"""

"""
# Correct: Just part of axes be squeezed.
class TestSqueeze2Op4(TestSqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)
"""


class TestUnsqueezeOp(unittest.TestCase):
    def setUp(self):
        self.set_hpu()
        self.init()

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

    def init(self, shape=[3, 40], axis=[0], dtype="float32"):
        self.ori_shape = shape
        self.axis = axis
        self.dtype = dtype
        self.x = np.random.randn(*shape)

    def check_result(self):
        rtol = 1e-3
        atol = 1e-3

        np_res = np.expand_dims(self.x, self.axis)
        paddle_x = paddle.to_tensor(self.x)
        op_res = paddle.unsqueeze(paddle_x, self.axis)

        np.testing.assert_allclose(op_res, np_res, rtol=rtol, atol=atol)

    def test_unsqueeze_axis_minus2(self):
        self.init(shape=[2, 5, 7], axis=-2)
        self.check_result()

    def test_unsqueeze_axis_minus1(self):
        self.init(shape=[2, 5, 7], axis=-1)
        self.check_result()

    def test_unsqueeze_axis_0(self):
        self.init(shape=[2, 5, 7], axis=0)
        self.check_result()

    def test_unsqueeze_axis_1(self):
        self.init(shape=[2, 5, 7], axis=1)
        self.check_result()

    def test_unsqueeze_axis_2(self):
        self.init(shape=[2, 5, 7], axis=2)
        self.check_result()

    def test_unsqueeze_axis_1_2(self):
        self.init(shape=[2, 5, 7], axis=[1, 2])
        self.check_result()

    def test_unsqueeze_axis_minus1_minus2(self):
        self.init(shape=[2, 5, 7], axis=[-1, -2])
        self.check_result()


if __name__ == "__main__":
    unittest.main()
