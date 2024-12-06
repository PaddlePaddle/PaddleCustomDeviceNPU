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
import paddle.distributed as dist

dist.init_parallel_env()
tensor_list = []
if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]], dtype="float32")
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]], dtype="float32")
dist.all_gather(tensor_list, data)
print(tensor_list)
# [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)
