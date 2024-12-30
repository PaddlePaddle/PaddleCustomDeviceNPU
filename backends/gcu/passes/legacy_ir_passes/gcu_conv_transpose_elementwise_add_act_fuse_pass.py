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
def conv2d_transpose_elementwise_add_relu_fuse_pass():
    def pattern(input, filter, bias):
        conv_t = ir.PassDesc.OP.conv2d_transpose(Input=input, Filter=filter)
        add = ir.PassDesc.OP.elementwise_add(X=conv_t, Y=bias)
        act = ir.PassDesc.OP.relu(X=add)
        return act

    def replace(input, filter, bias):
        conv_t_act = ir.PassDesc.OP.fused_conv2d_transpose_bias_act(
            Input=input, Filter=filter, Bias=bias
        )
        conv_t_act.Attr("strides").MappedPattern(
            op="conv2d_transpose", name="strides", index=0
        )
        conv_t_act.Attr("paddings").MappedPattern(
            op="conv2d_transpose", name="paddings", index=0
        )
        conv_t_act.Attr("output_padding").MappedPattern(
            op="conv2d_transpose", name="output_padding", index=0
        )
        conv_t_act.Attr("output_size").MappedPattern(
            op="conv2d_transpose", name="output_size", index=0
        )
        conv_t_act.Attr("padding_algorithm").MappedPattern(
            op="conv2d_transpose", name="padding_algorithm", index=0
        )
        conv_t_act.Attr("groups").MappedPattern(
            op="conv2d_transpose", name="groups", index=0
        )
        conv_t_act.Attr("dilations").MappedPattern(
            op="conv2d_transpose", name="dilations", index=0
        )
        conv_t_act.Attr("data_format").MappedPattern(
            op="conv2d_transpose", name="data_format", index=0
        )
        conv_t_act.SetAttr("activation", "relu")
        return conv_t_act

    return pattern, replace


@paddle.incubate.passes.ir.RegisterPass
def conv2d_transpose_elementwise_add_sigmoid_fuse_pass():
    def pattern(input, filter, bias):
        conv_t = ir.PassDesc.OP.conv2d_transpose(Input=input, Filter=filter)
        add = ir.PassDesc.OP.elementwise_add(X=conv_t, Y=bias)
        act = ir.PassDesc.OP.sigmoid(X=add)
        return act

    def replace(input, filter, bias):
        conv_t_act = ir.PassDesc.OP.fused_conv2d_transpose_bias_act(
            Input=input, Filter=filter, Bias=bias
        )
        conv_t_act.Attr("strides").MappedPattern(
            op="conv2d_transpose", name="strides", index=0
        )
        conv_t_act.Attr("paddings").MappedPattern(
            op="conv2d_transpose", name="paddings", index=0
        )
        conv_t_act.Attr("output_padding").MappedPattern(
            op="conv2d_transpose", name="output_padding", index=0
        )
        conv_t_act.Attr("output_size").MappedPattern(
            op="conv2d_transpose", name="output_size", index=0
        )
        conv_t_act.Attr("padding_algorithm").MappedPattern(
            op="conv2d_transpose", name="padding_algorithm", index=0
        )
        conv_t_act.Attr("groups").MappedPattern(
            op="conv2d_transpose", name="groups", index=0
        )
        conv_t_act.Attr("dilations").MappedPattern(
            op="conv2d_transpose", name="dilations", index=0
        )
        conv_t_act.Attr("data_format").MappedPattern(
            op="conv2d_transpose", name="data_format", index=0
        )
        conv_t_act.SetAttr("activation", "sigmoid")
        return conv_t_act

    return pattern, replace
