#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.incubate.passes import ir


@paddle.incubate.passes.ir.RegisterPass
def fused_linear_pass():
    def pattern(input, weight, bias):
        mat = ir.PassDesc.OP.matmul_v2(X=input, Y=weight)
        out = ir.PassDesc.OP.elementwise_add(X=mat, Y=bias)
        return out

    def replace(input, weight, bias):
        linear = ir.PassDesc.OP.fc(Input=input, W=weight, Bias=bias)
        linear.SetAttr("in_num_col_dims", input.Attr("shape").Size() - 1)
        linear.SetAttr("activation_type", "")
        linear.SetAttr("padding_weights", False)
        linear.SetAttr("support_int8", False)
        return linear.Output("Out")

    return pattern, replace
