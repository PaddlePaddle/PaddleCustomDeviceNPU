#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function


import paddle
import unittest
import numpy as np


class TestContiguous(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.input = np.random.random([3, 3]).astype(self.dtype)

    def run_test(self, use_npu):
        if not use_npu:
            paddle.set_device("cpu")
        else:
            paddle.set_device("intel_hpu")

        x = paddle.to_tensor(self.input)
        assert x.is_contiguous()

        y = paddle.as_strided(x, [3, 3], [1, 3])
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()
        return x.strides, y, y.strides, z, z.strides

    def test_contiguous(self):
        (
            input_strides,
            output0,
            output0_strides,
            output1,
            output1_strides,
        ) = self.run_test(False)
        (
            input_strides_npu,
            output0_npu,
            output0_strides_npu,
            output1_npu,
            output1_strides_npu,
        ) = self.run_test(True)

        np.testing.assert_allclose(input_strides, input_strides_npu)
        np.testing.assert_allclose(output0, output0_npu)
        np.testing.assert_allclose(output0_strides, output0_strides_npu)
        np.testing.assert_allclose(output1, output1_npu)
        np.testing.assert_allclose(output1_strides, output1_strides_npu)


if __name__ == "__main__":
    unittest.main()
