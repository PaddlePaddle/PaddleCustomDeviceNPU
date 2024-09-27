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
def fused_multi_head_attention_pass():
    def pattern(query, key, value):
        q_trans = ir.PassDesc.OP.transpose2(X=query)
        k_trans = ir.PassDesc.OP.transpose2(X=key)

        qkT = ir.PassDesc.OP.matmul_v2(X=q_trans.Output("Out"), Y=k_trans.Output("Out"))
        qkT_scale = ir.PassDesc.OP.scale(X=qkT)
        attn_weights = ir.PassDesc.OP.softmax(X=qkT_scale)

        v_trans = ir.PassDesc.OP.transpose2(X=value)
        attn_scores = ir.PassDesc.OP.matmul_v2(X=attn_weights, Y=v_trans.Output("Out"))

        attn_scores_trans = ir.PassDesc.OP.transpose2(X=attn_scores)

        return attn_scores_trans.Output("Out")

    def replace(query, key, value):
        attn = ir.PassDesc.OP.fused_multi_head_attention(Q=query, K=key, V=value)
        attn.SetAttr("head_dim", query.Attr("shape")[-1])
        return attn.Output("Out")

    return pattern, replace
