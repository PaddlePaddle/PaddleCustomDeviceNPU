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

import paddle
from paddle.static import InputSpec

paddle.set_device("intel_hpu")
# paddle.set_device("custom_cpu")

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

# model = ADDMUL()
# net = paddle.jit.to_static(model, input_spec=[x_spec, y_spec, b_spec])
# paddle.jit.save(net, "addmul_model")

model = ADD()
# net = paddle.jit.to_static(model, input_spec=[y_spec, b_spec])
# paddle.jit.save(net, "sum_model")


X = paddle.randn([T, H], dtype="bfloat16")
Y = paddle.randn([H, T], dtype="bfloat16")
B = paddle.randn([T], dtype="bfloat16")

print(X.shape)
print(Y.shape)
print(B.shape)
# out = model(X, Y, B)
out = model(Y, B)
print(out.shape)
print(Y)
print(B)
print(out)
