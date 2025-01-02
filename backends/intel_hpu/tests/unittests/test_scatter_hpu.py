#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest

from tests.op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


class test_gather_i64_ow(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)

        x = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index = np.array([2, 1, 0, 1]).astype("int64")
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype("float32")
        self.inputs = {"X": x, "Ids": index, "Updates": updates}
        self.attrs = {"overwrite": True}

        output = np.copy(x)
        output[index] = updates
        self.outputs = {"Out": output}

    def set_hpu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class test_gather_i64_no_ow(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.python_api = paddle.scatter

        x = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index = np.array([2, 1, 0, 1]).astype("int64")
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype("float32")
        self.inputs = {"X": x, "Ids": index, "Updates": updates}
        self.attrs = {"overwrite": False}

        zeros = np.zeros([4, 2]).astype("float32")
        output = np.copy(x)
        output[index] = zeros
        for i in range(0, len(index)):
            output[index[i]] += updates[i]
        self.outputs = {"Out": output}

    def set_hpu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class test_gather_i32_ow(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)

        x = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index = np.array([2, 1, 0, 1]).astype("int32")
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype("float32")

        output = np.copy(x)
        output[index] = updates
        self.inputs = {"X": x, "Ids": index, "Updates": updates}
        self.outputs = {"Out": output}
        self.attrs = {"overwrite": True}

    def set_hpu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class test_gather_i32_no_ow(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.python_api = paddle.scatter

        x = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
        index = np.array([2, 1, 0, 1]).astype("int32")
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype("float32")

        zeros = np.zeros([4, 2]).astype("float32")
        output = np.copy(x)
        output[index] = zeros
        for i in range(0, len(index)):
            output[index[i]] += updates[i]
        self.inputs = {"X": x, "Ids": index, "Updates": updates}
        self.outputs = {"Out": output}
        self.attrs = {"overwrite": False}

    def set_hpu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class test_gather_fp32_no_ow(OpTest):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.python_api = paddle.scatter

        ref_np = np.ones((3, 2)).astype("float32")
        index_np = np.array([1]).astype("int32")
        updates_np = np.random.random((1, 2)).astype("float32")

        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}
        self.attrs = {"overwrite": True}

    def set_hpu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Updates"],
            "Out",
            check_dygraph=False,
            numeric_place=paddle.CPUPlace(),
        )


class test_gather_fp32_no_ow_2(test_gather_fp32_no_ow):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.python_api = paddle.scatter

        ref_np = np.ones((3, 2)).astype("float32")
        index_np = np.array([1]).astype("int32")
        updates_np = np.random.random((1, 2)).astype("float32")

        output_np = np.copy(ref_np)
        output_np[index_np] += updates_np
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}
        self.attrs = {"overwrite": False}


class test_gather_fp32_no_ow_3(test_gather_fp32_no_ow):
    def setUp(self):
        self.set_hpu()
        self.op_type = "scatter"
        self.place = paddle.CustomPlace("intel_hpu", 0)

        ref_np = np.ones((3, 2)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 2)).astype("float32")

        output_np = np.copy(ref_np)
        output_np[1] = updates_np[0]
        output_np[2] = updates_np[1]
        self.inputs = {"X": ref_np, "Ids": index_np, "Updates": updates_np}
        self.outputs = {"Out": output_np}
        self.attrs = {"overwrite": True}


if __name__ == "__main__":
    unittest.main()
