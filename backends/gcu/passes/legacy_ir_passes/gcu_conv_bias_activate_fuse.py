# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

from paddle.incubate.passes import ir
import logging


@ir.RegisterPass
def gcu_fuse_conv_bias():
    def pattern_conv_bias(input, filter, y):
        conv2d = ir.PassDesc.OP.conv2d(Input=input, Filter=filter)
        return ir.PassDesc.OP.elementwise_add(X=conv2d, Y=y)

    def replace_conv_bias(input, filter, y):
        logging.info("======= start to do conv bias op replace ======")
        gcu_conv_bias = ir.PassDesc.OP.GCUConvBias(X=input, Filter=filter, Y=y)
        gcu_conv_bias.Attr("strides").MappedPattern(
            op="conv2d", name="strides", element_index=None
        )
        gcu_conv_bias.Attr("data_format").MappedPattern(
            op="conv2d", name="data_format", element_index=None
        )
        gcu_conv_bias.Attr("dilations").MappedPattern(
            op="conv2d", name="dilations", element_index=None
        )
        gcu_conv_bias.Attr("paddings").MappedPattern(
            op="conv2d", name="paddings", element_index=None
        )
        gcu_conv_bias.Attr("padding_algorithm").MappedPattern(
            op="conv2d", name="padding_algorithm", element_index=None
        )
        gcu_conv_bias.Attr("groups").MappedPattern(op="conv2d", name="groups")
        gcu_conv_bias.Attr("axis").MappedPattern(op="elementwise_add", name="axis")
        return gcu_conv_bias.Output("Out")

    return pattern_conv_bias, replace_conv_bias


@ir.RegisterPass
def gcu_fuse_conv_bias_activate():
    def pattern_conv_bias_relu(input, filter, y):
        conv_bias = ir.PassDesc.OP.GCUConvBias(X=input, Filter=filter, Y=y)
        return ir.PassDesc.OP.relu(X=conv_bias.Output("Out"))

    def pattern_conv_bias_hard_sigmoid(input, filter, y):
        conv_bias = ir.PassDesc.OP.GCUConvBias(X=input, Filter=filter, Y=y)
        return ir.PassDesc.OP.hard_sigmoid(X=conv_bias.Output("Out"))

    def replace_conv_bias_relu(input, filter, y):
        logging.info("======= start to do conv bias relu op replace ======")
        gcu_conv_bias_relu = ir.PassDesc.OP.GCUConvBiasRelu(X=input, Filter=filter, Y=y)
        gcu_conv_bias_relu.Attr("strides").MappedPattern(
            op="GCUConvBias", name="strides", element_index=None
        )
        gcu_conv_bias_relu.Attr("data_format").MappedPattern(
            op="GCUConvBias", name="data_format", element_index=None
        )
        gcu_conv_bias_relu.Attr("dilations").MappedPattern(
            op="GCUConvBias", name="dilations", element_index=None
        )
        gcu_conv_bias_relu.Attr("paddings").MappedPattern(
            op="GCUConvBias", name="paddings", element_index=None
        )
        gcu_conv_bias_relu.Attr("padding_algorithm").MappedPattern(
            op="GCUConvBias", name="padding_algorithm", element_index=None
        )
        gcu_conv_bias_relu.Attr("groups").MappedPattern(op="GCUConvBias", name="groups")
        gcu_conv_bias_relu.Attr("axis").MappedPattern(op="GCUConvBias", name="axis")
        return gcu_conv_bias_relu.Output("Out")

    def replace_conv_bias_hard_sigmoid(input, filter, y):
        logging.info("======= start to do conv bias hard_sigmoid op replace ======")
        gcu_conv_bias_relu = ir.PassDesc.OP.GCUConvBiasHardSigmoid(
            X=input, Filter=filter, Y=y
        )
        gcu_conv_bias_relu.Attr("strides").MappedPattern(
            op="GCUConvBias", name="strides", element_index=None
        )
        gcu_conv_bias_relu.Attr("data_format").MappedPattern(
            op="GCUConvBias", name="data_format", element_index=None
        )
        gcu_conv_bias_relu.Attr("dilations").MappedPattern(
            op="GCUConvBias", name="dilations", element_index=None
        )
        gcu_conv_bias_relu.Attr("paddings").MappedPattern(
            op="GCUConvBias", name="paddings", element_index=None
        )
        gcu_conv_bias_relu.Attr("padding_algorithm").MappedPattern(
            op="GCUConvBias", name="padding_algorithm", element_index=None
        )
        gcu_conv_bias_relu.Attr("groups").MappedPattern(op="GCUConvBias", name="groups")
        gcu_conv_bias_relu.Attr("axis").MappedPattern(op="GCUConvBias", name="axis")
        gcu_conv_bias_relu.Attr("slope").MappedPattern(op="hard_sigmoid", name="slope")
        gcu_conv_bias_relu.Attr("offset").MappedPattern(
            op="hard_sigmoid", name="offset"
        )
        return gcu_conv_bias_relu.Output("Out")

    return (pattern_conv_bias_relu, replace_conv_bias_relu), (
        pattern_conv_bias_hard_sigmoid,
        replace_conv_bias_hard_sigmoid,
    )
