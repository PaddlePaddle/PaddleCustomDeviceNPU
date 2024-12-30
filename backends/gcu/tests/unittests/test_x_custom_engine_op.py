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
from paddle_custom_device.gcu import passes as gcu_passes


class TestCustomEngineOp(unittest.TestCase):
    def setUp(self):
        gcu_passes.setUp()

    def test_gcu_custom_engine_op(self):
        x_data = np.random.random([2, 2]).astype(np.float32)
        x = paddle.to_tensor(x_data, dtype=np.float32)
        out = paddle.base.core.eager._run_custom_op("test_for_custom_engine_op", x)[0]
        np.testing.assert_allclose(x, out, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
