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

from __future__ import print_function, division

import os
import paddle

from . import llama  # noqa: F401
from . import chatglm  # noqa: F401


def setUp():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )


def register_pass(pass_builder, pass_name):
    pass_builder.append_pass(pass_name)
    paddle.base.core.register_subgraph_pass(pass_name)


def addPasses(pass_builder, model_type, quant_type):
    if model_type == "llama" and quant_type == "a8w8":
        register_pass(pass_builder, "remove_residual_in_fused_bias_residual_layernorm")
        register_pass(pass_builder, "remove_residual_in_rms_norm")
        register_pass(pass_builder, "remove_blha_get_max_len")
        register_pass(pass_builder, "llama_fuse_attention_smooth_quant_layer_begin")
        register_pass(pass_builder, "llama_fuse_attention_smooth_quant_layer_end")
        register_pass(pass_builder, "llama_fuse_attention_smooth_quant_layer")
        register_pass(pass_builder, "llama_fuse_lm_head_with_slice")
        register_pass(pass_builder, "llama_fuse_lm_head")
        register_pass(pass_builder, "llama_fuse_get_padding_offset")
        register_pass(pass_builder, "replace_top_p_sampling_by_top_k_v2")
    elif model_type == "llama":
        register_pass(pass_builder, "remove_residual_in_fused_bias_residual_layernorm")
        register_pass(pass_builder, "remove_residual_in_rms_norm")
        register_pass(pass_builder, "remove_blha_get_max_len")
        register_pass(pass_builder, "llama_fuse_attention_layer_begin")
        register_pass(pass_builder, "llama_fuse_attention_layer_end")
        register_pass(pass_builder, "llama_fuse_attention_layer")
        register_pass(pass_builder, "llama_fuse_lm_head_with_slice")
        register_pass(pass_builder, "llama_fuse_lm_head")
        register_pass(pass_builder, "llama_fuse_get_padding_offset")
        register_pass(pass_builder, "replace_top_p_sampling_by_top_k_v2")
    elif model_type.startswith("chatglm"):
        register_pass(pass_builder, "remove_residual_in_fused_bias_residual_layernorm")
        register_pass(pass_builder, "remove_residual_in_rms_norm")
        register_pass(pass_builder, "remove_blha_get_max_len")
        register_pass(pass_builder, "chatglm_fuse_attention_layer_begin")
        register_pass(pass_builder, "chatglm_fuse_attention_layer_end")
        register_pass(pass_builder, "chatglm_fuse_attention_layer")
        register_pass(pass_builder, "llama_fuse_attention_layer_begin")
        register_pass(pass_builder, "llama_fuse_attention_layer_end")
        register_pass(pass_builder, "llama_fuse_attention_layer")
        register_pass(pass_builder, "llama_fuse_lm_head_with_slice")
        register_pass(pass_builder, "llama_fuse_lm_head")
        register_pass(pass_builder, "llama_fuse_get_padding_offset")
        register_pass(pass_builder, "replace_top_p_sampling_by_top_k_v2")
    else:
        print("NPU pass not support")
