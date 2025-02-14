#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def cumsum_numpy(x, axis=None, dtype=None):
    return np.cumsum(x, axis=axis)


class TestHPU_Cumsum_Inplace_OpInt(unittest.TestCase):
    def setUp(self):
        self.set_hpu()
        self.init_dtype()
        self.init_testcase()

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

    def init(self):
        attrs = {}
        inputs = {}
        outputs = {}
        outputs_inplace = {}

    def init_dtype(self):
        self.dtype = "int32"

    def init_testcase(self):
        self.attrs = {"axis": 0}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}

    def testcase_cumsum_op(self):
        paddle.disable_static()
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=0)}

    def testcase_cumsum_inplace(self):
        paddle.enable_static()
        self.outputs_inplace = {"Out": self.inputs["X"].cumsum(axis=0)}

    def check_result(self, numpy_res, ops_inplace_res, ops_res):
        if self.dtype == "float32":
            rtol = 1e-5
            atol = 1e-6
        elif self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-4
        elif self.dtype == "bfloat16":
            rtol = 1e-2
            atol = 1e-3
        elif self.dtype == "int32":
            rtol = 1e-2
            atol = 1e-3
        elif self.dtype == "int64":
            rtol = 1e-2
            atol = 1e-3
        else:
            self.assertTrue(
                False,
                msg="Cumsum_Inplace OP input dtype only supports, int, int64, bfloat16, \
                     float16 and float32, but got "
                + self.dtype,
            )
        np.testing.assert_allclose(numpy_res, ops_inplace_res, rtol=rtol, atol=atol)
        np.testing.assert_allclose(numpy_res, ops_res, rtol=rtol, atol=atol)

    def test_custom_inplace(self):
        self.testcase_cumsum_op()
        self.testcase_cumsum_inplace()
        numpy_res = cumsum_numpy(
            self.inputs.get("X", 5), self.attrs.get("axis", 0), dtype=self.dtype
        )
        self.check_result(
            numpy_res, self.outputs_inplace.get("Out", 0), self.outputs.get("Out", 0)
        )


class TestHPU_Cumsum_Inplace_OpInt_1(TestHPU_Cumsum_Inplace_OpInt):
    def init_testcase(self):
        self.attrs = {"axis": 1}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}

    def testcase_cumsum_op(self):
        paddle.disable_static()
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=1)}

    def testcase_cumsum_inplace(self):
        paddle.enable_static()
        self.outputs_inplace = {"Out": self.inputs["X"].cumsum(axis=1)}


class TestHPU_Cumsum_Inplace_OpInt_2(TestHPU_Cumsum_Inplace_OpInt):
    def testcase_op(self):
        self.attrs = {"axis": 2}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}

    def testcase_cumsum_op(self):
        paddle.disable_static()
        self.outputs = {"Out": self.inputs["X"].cumsum(axis=0)}

    def testcase_cumsum_inplace(self):
        paddle.enable_static()
        self.outputs_inplace = {"Out": self.inputs["X"].cumsum(axis=0)}


class TestHPU_Cumsum_Inplace_OpInt_3(TestHPU_Cumsum_Inplace_OpInt):
    def testcase_op(self):
        self.attrs = {"axis": -1, "reverse": True}
        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}

    def testcase_cumsum_op(self):
        paddle.disable_static()
        self.outputs = {
            "Out": np.flip(np.flip(self.inputs["X"], axis=2).cumsum(axis=2), axis=2)
        }

    def testcase_cumsum_inplace(self):
        paddle.enable_static()
        self.outputs_inplace = {
            "Out": np.flip(np.flip(self.inputs["X"], axis=2).cumsum(axis=2), axis=2)
        }


class Test_Cumsum_Inplace_OpInt64(TestHPU_Cumsum_Inplace_OpInt):
    def init_dtype(self):
        self.dtype = "int64"


class Test_Cumsum_Inplace_OpFloat(TestHPU_Cumsum_Inplace_OpInt):
    def init_dtype(self):
        self.dtype = "float32"


class Test_Cumsum_Inplace_OpFloat16(TestHPU_Cumsum_Inplace_OpInt):
    def init_dtype(self):
        self.dtype = "float16"


if __name__ == "__main__":
    unittest.main()
