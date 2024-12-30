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

from paddle_custom_device.intel_hpu.ops import *  # noqa


class Fused_Rms_Qkv_Rope(paddle.nn.Layer):
    def __init__(self, ln_scales, qkv_weights, epsilon, head_dim, num_head):
        super().__init__()
        self.ln_scales = ln_scales
        self.qkv_weights = qkv_weights
        self.epsilon = epsilon
        self.head_dim = head_dim
        self.num_head = num_head

    def forward(self, i, src, cos, sin, position_ids):
        query_states, key_states, value_states = fused_rms_qkv_rope(
            src,
            self.ln_scales[i],
            self.qkv_weights[i],
            cos,
            sin,
            position_ids,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )
        return query_states, key_states, value_states


class Fused_Rms_Qkv_Rope_v2(paddle.nn.Layer):
    def __init__(self, ln_scales, qkv_weights, epsilon, head_dim, num_head):
        super().__init__()
        self.ln_scales = ln_scales
        self.qkv_weights = qkv_weights
        self.epsilon = epsilon
        self.head_dim = head_dim
        self.num_head = num_head

    def forward(self, i, src, rotary_embs):
        query_states, key_states, value_states = fused_rms_qkv_rope_v2(
            src,
            self.ln_scales[i],
            self.qkv_weights[i],
            rotary_embs,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )
        return query_states, key_states, value_states


class Fused_Sdpa_Proj(paddle.nn.Layer):
    def __init__(self, scaling_factor, linear_weights):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.linear_weights = linear_weights

    def forward(self, i, query_states, key_states, value_states, attention_mask):
        out_linear_out = fused_sdpa_proj(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.linear_weights[i],
            self.scaling_factor,
        )
        return out_linear_out


class Fused_Mlp(paddle.nn.Layer):
    def __init__(self, proj_weight, up_weight, down_weight):
        super().__init__()
        self.proj_weight = proj_weight
        self.down_weight = down_weight
        self.up_weight = up_weight

    def forward(self, i, x):
        fused_mlp_out = fused_mlp(
            x,
            self.proj_weight[i],
            self.up_weight[i],
            self.down_weight[i],
        )
        return fused_mlp_out


class Fused_Rms_Mlp(paddle.nn.Layer):
    def __init__(self, ln_scales, epsilon, proj_weight, down_weight):
        super().__init__()
        self.ln_scales = ln_scales
        self.epsilon = epsilon
        self.proj_weight = proj_weight
        self.down_weight = down_weight

    def forward(self, i, x):
        fused_rms_mlp_out = fused_rms_mlp(
            x,
            self.ln_scales[i],
            self.proj_weight[i],
            self.down_weight[i],
            self.epsilon,
        )
        return fused_rms_mlp_out
