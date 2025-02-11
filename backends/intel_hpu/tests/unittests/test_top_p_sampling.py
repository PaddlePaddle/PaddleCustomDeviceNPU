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
import unittest
import paddle

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class top_p_sampling_test(unittest.TestCase):
    def is_token_in_targets(self, tokens, target_tensors):
        for target in target_tensors:
            if paddle.all(paddle.equal(tokens, target)):
                return True
        return False

    def test_case1(self):
        probs = paddle.to_tensor([[0.1, 0.5, 0.3, 0.1], [0.5, 0.3, 0.1, 0.1]])
        top_p = paddle.to_tensor(0.7)
        _, tokens = paddle.tensor.top_p_sampling(probs, top_p)

        tensor1 = paddle.to_tensor([[1], [0]])
        tensor2 = paddle.to_tensor([[2], [0]])
        tensor3 = paddle.to_tensor([[1], [1]])
        tensor4 = paddle.to_tensor([[2], [1]])

        target_tensors = [tensor1, tensor2, tensor3, tensor4]
        self.assertTrue(self.is_token_in_targets(tokens, target_tensors))

    def test_case2(self):
        probs = paddle.to_tensor([[0.1, 0.5, 0.3, 0.1]])
        top_p = paddle.to_tensor(0.7)
        _, tokens = paddle.tensor.top_p_sampling(probs, top_p)

        tensor1 = paddle.to_tensor([[1]])
        tensor2 = paddle.to_tensor([[2]])

        target_tensors = [tensor1, tensor2]
        self.assertTrue(self.is_token_in_targets(tokens, target_tensors))

    def test_case3(self):
        probs = paddle.to_tensor([[0.5, 0.3, 0.1, 0.1]])
        top_p = paddle.to_tensor(0.7)
        _, tokens = paddle.tensor.top_p_sampling(probs, top_p)

        tensor1 = paddle.to_tensor([[0]])
        tensor2 = paddle.to_tensor([[1]])

        target_tensors = [tensor1, tensor2]
        self.assertTrue(self.is_token_in_targets(tokens, target_tensors))


if __name__ == "__main__":
    unittest.main()
