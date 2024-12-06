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

import os
import paddle


def setUp():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )


def register_pass(pass_builder, pass_name):
    pass_builder.append_pass(pass_name)
    paddle.base.core.register_subgraph_pass(pass_name)


def inference_common_passes(use_pir=True):
    if use_pir:
        return [
            # Functional pass
            "add_shadow_output_after_dead_parameter_pass",
            "delete_quant_dequant_linear_op_pass",
            "delete_weight_dequant_linear_op_pass",
            "map_op_to_another_pass",
            "identity_op_clean_pass",
            # Operator fusion pass
            "silu_fuse_pass",
            "conv2d_bn_fuse_pass",
            "conv2d_add_act_fuse_pass",
            "conv2d_add_fuse_pass",
        ]
    else:
        return [
            "map_op_to_another_pass",
            "is_test_pass",
            "simplify_with_basic_ops_pass",
            "constant_folding_pass",
            "conv_bn_fuse_pass",
            "conv_eltwiseadd_bn_fuse_pass",
            "conv_transpose_bn_fuse_pass",
            "conv_transpose_eltwiseadd_bn_fuse_pass",
            "conv_elementwise_add_act_fuse_pass",
            "conv_elementwise_add2_act_fuse_pass",
            "conv_elementwise_add_fuse_pass",
            "depthwise_conv_bn_fuse_pass",
            "depthwise_conv_eltwiseadd_bn_fuse_pass",
            "multihead_matmul_fuse_pass_v2",
            "vit_attention_fuse_pass",
            "gpu_cpu_squeeze2_matmul_fuse_pass",
            "gpu_cpu_reshape2_matmul_fuse_pass",
            "gpu_cpu_flatten2_matmul_fuse_pass",
            "gpu_cpu_map_matmul_v2_to_mul_pass",
            "gpu_cpu_map_matmul_v2_to_matmul_pass",
            "matmul_scale_fuse_pass",
            "multihead_matmul_fuse_pass_v3",
            "gpu_cpu_map_matmul_to_mul_pass",
            "fc_fuse_pass",
            "fc_elementwise_layernorm_fuse_pass",
            "transpose_flatten_concat_fuse_pass",
            "transfer_layout_pass",
            "transfer_layout_elim_pass",
            "fused_sdp_attention",
            "conv2d_transpose_elementwise_add_fuse_pass",
            "conv2d_depthwise_elementwise_add_fuse_pass",
            "conv2d_elementwise_add_fuse_pass",
            # always last
            "add_netoutput_op_pass",
        ]


def inference_ocr_passes(use_pir=True):
    return inference_common_passes(use_pir)


def inference_detection_passes(use_pir=True):
    return inference_common_passes(use_pir)


def inference_passes(use_pir=True, name="common"):
    PASS_MAP = {
        "common": inference_common_passes,
        "PaddleOCR": inference_ocr_passes,
        "PaddleDetection": inference_detection_passes,
    }
    if name not in PASS_MAP.keys():
        print(
            "[INFO] Not found passes for {}, common passes will be used instead.".format(
                name
            ),
            flush=True,
        )
        name = "common"
    return PASS_MAP[name](use_pir)


def append_passes_for_legacy_ir(pass_builder, name="common"):
    gcu_passes = inference_passes(use_pir=False, name=name)
    for common_pass in gcu_passes:
        pass_builder.append_pass(common_pass)


def set_exp_enable_mixed_precision_ops(config):
    config.exp_enable_mixed_precision_ops(
        {
            "fused_self_attn",
            "fused_conv2d_add",
            "fused_conv2d_transpose_bias_act",
            "gcu_netoutput",
        }
    )
