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
out_tensor_list = []
if dist.get_rank() == 0:
    data1 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")
    data2 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype="float32")
else:
    data1 = paddle.to_tensor([[13, 14, 15], [16, 17, 18]], dtype="float32")
    data2 = paddle.to_tensor([[19, 20, 21], [22, 23, 24]], dtype="float32")
dist.alltoall([data1, data2], out_tensor_list)
print(out_tensor_list)
# [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]] (2 GPUs, out for rank 0)
# [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]] (2 GPUs, out for rank 1)
