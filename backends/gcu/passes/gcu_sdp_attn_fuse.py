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
def fused_sdp_attention():
    def pattern(input):
        input_reshape = ir.PassDesc.OP.reshape2(X=input)
        input_trans = ir.PassDesc.OP.transpose2(X=input_reshape.Output("Out"))
        # QKV
        q = ir.PassDesc.OP.slice(Input=input_trans.Output("Out"))
        k = ir.PassDesc.OP.slice(Input=input_trans.Output("Out"))
        v = ir.PassDesc.OP.slice(Input=input_trans.Output("Out"))

        q_scale = ir.PassDesc.OP.scale(X=q)
        k_T = ir.PassDesc.OP.transpose2(X=k)
        qkT = ir.PassDesc.OP.matmul(X=q_scale, Y=k_T.Output("Out"))
        attn_weights = ir.PassDesc.OP.softmax(X=qkT)
        attn_scores = ir.PassDesc.OP.matmul(X=attn_weights, Y=v)

        attn_scores_trans = ir.PassDesc.OP.transpose2(X=attn_scores)
        attn_scores_out = ir.PassDesc.OP.reshape2(X=attn_scores_trans.Output("Out"))

        return attn_scores_out.Output("Out")

    def replace(input):
        attn = ir.PassDesc.OP.fused_self_attn(X=input)
        attn.Attr("head_dim").MappedPattern(
            op="reshape2", name="shape", index=0, element_index=4
        )
        return attn.Output("Out")

    return pattern, replace
