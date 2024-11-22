# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.profiler as profiler
import paddle
from paddle.static import InputSpec

paddle.set_device("intel_hpu")

BATCH_SIZE = 1
T = 2
H = 16


class ADDMUL(paddle.nn.Layer):
    def __init__(self):
        super(ADDMUL, self).__init__()

    def forward(self, inputX, inputY, bias):
        out = paddle.matmul(inputX, inputY)
        out = paddle.add(out, bias)
        return out


class ADD(paddle.nn.Layer):
    def __init__(self):
        super(ADD, self).__init__()

    def forward(self, inputY, bias):
        out = paddle.add(inputY, bias)
        out = paddle.add(out, bias)
        out = paddle.add(out, inputY)
        return out


# bfloat16
x_spec = InputSpec(shape=[T, H], dtype="bfloat16", name="x")
y_spec = InputSpec(shape=[H, T], dtype="bfloat16", name="y")
b_spec = InputSpec(shape=[T], dtype="bfloat16", name="b")

model = ADDMUL()

with profiler.Profiler(
    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE],
    scheduler=(0, 5),
    on_trace_ready=profiler.export_chrome_tracing("./log"),
) as p:
    X = paddle.randn([T, H], dtype="bfloat16")
    Y = paddle.randn([H, T], dtype="bfloat16")
    B = paddle.randn([T], dtype="bfloat16")

    print(X.shape)
    print(Y.shape)
    print(B.shape)
    out = model(X, Y, B)
    p.step()
    print(out.shape)
    print(Y)
    print(B)
    print(out)
