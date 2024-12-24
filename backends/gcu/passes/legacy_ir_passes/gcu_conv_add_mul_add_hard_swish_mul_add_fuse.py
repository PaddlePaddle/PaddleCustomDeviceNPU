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
def gcu_fuse_conv_add_mul_add_hard_swish_mul_add():
    def pattern(input, filter, y1, x1, y2, x2, y3):
        conv2d = ir.PassDesc.OP.conv2d(Input=input, Filter=filter)
        add1 = ir.PassDesc.OP.elementwise_add(X=conv2d, Y=y1)
        mul1 = ir.PassDesc.OP.elementwise_mul(X=x1, Y=add1.Output("Out"))
        add2 = ir.PassDesc.OP.elementwise_add(X=mul1.Output("Out"), Y=y2)
        hard_swish = ir.PassDesc.OP.hard_swish(X=add2.Output("Out"))
        mul2 = ir.PassDesc.OP.elementwise_mul(X=x2, Y=hard_swish.Output("Out"))
        return ir.PassDesc.OP.elementwise_add(X=mul2.Output("Out"), Y=y3)

    def replace(input, filter, y1, x1, y2, x2, y3):
        logging.info("======= start to do conv bn hard_swish op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUConvAddMulAddHardSwishMulAdd(
            X=input, Filter=filter, Y1=y1, X1=x1, Y2=y2, X2=x2, Y3=y3
        )
        gcu_conv_bn.Attr("strides").MappedPattern(
            op="conv2d", name="strides", element_index=None
        )
        gcu_conv_bn.Attr("data_format").MappedPattern(
            op="conv2d", name="data_format", element_index=None
        )
        gcu_conv_bn.Attr("dilations").MappedPattern(
            op="conv2d", name="dilations", element_index=None
        )
        gcu_conv_bn.Attr("paddings").MappedPattern(
            op="conv2d", name="paddings", element_index=None
        )
        gcu_conv_bn.Attr("padding_algorithm").MappedPattern(
            op="conv2d", name="padding_algorithm", element_index=None
        )
        gcu_conv_bn.Attr("groups").MappedPattern(op="conv2d", name="groups")
        gcu_conv_bn.Attr("axis1").MappedPattern(op="elementwise_add", name="axis")
        gcu_conv_bn.Attr("axis2").MappedPattern(op="elementwise_mul", name="axis")
        gcu_conv_bn.Attr("axis3").MappedPattern(op="elementwise_add", name="axis")
        gcu_conv_bn.Attr("axis4").MappedPattern(op="elementwise_mul", name="axis")
        gcu_conv_bn.Attr("axis5").MappedPattern(op="elementwise_add", name="axis")
        return gcu_conv_bn.Output("Out")

    return pattern, replace


@ir.RegisterPass
def gcu_fuse_depthwise_conv_add_mul_add_hard_swish_mul_add():
    def pattern(input, filter, y1, x1, y2, x2, y3):
        depthwise_conv2d = ir.PassDesc.OP.depthwise_conv2d(Input=input, Filter=filter)
        add1 = ir.PassDesc.OP.elementwise_add(X=depthwise_conv2d, Y=y1)
        mul1 = ir.PassDesc.OP.elementwise_mul(X=x1, Y=add1.Output("Out"))
        add2 = ir.PassDesc.OP.elementwise_add(X=mul1.Output("Out"), Y=y2)
        hard_swish = ir.PassDesc.OP.hard_swish(X=add2.Output("Out"))
        mul2 = ir.PassDesc.OP.elementwise_mul(X=x2, Y=hard_swish.Output("Out"))
        return ir.PassDesc.OP.elementwise_add(X=mul2.Output("Out"), Y=y3)

    def replace(input, filter, y1, x1, y2, x2, y3):
        logging.info(
            "======= start to do depthwise conv bn hard_swish op replace ======"
        )
        gcu_conv_bn = ir.PassDesc.OP.GCUDepthwiseConvAddMulAddHardSwishMulAdd(
            X=input, Filter=filter, Y1=y1, X1=x1, Y2=y2, X2=x2, Y3=y3
        )
        gcu_conv_bn.Attr("strides").MappedPattern(
            op="depthwise_conv2d", name="strides", element_index=None
        )
        gcu_conv_bn.Attr("data_format").MappedPattern(
            op="depthwise_conv2d", name="data_format", element_index=None
        )
        gcu_conv_bn.Attr("dilations").MappedPattern(
            op="depthwise_conv2d", name="dilations", element_index=None
        )
        gcu_conv_bn.Attr("paddings").MappedPattern(
            op="depthwise_conv2d", name="paddings", element_index=None
        )
        gcu_conv_bn.Attr("padding_algorithm").MappedPattern(
            op="depthwise_conv2d", name="padding_algorithm", element_index=None
        )
        gcu_conv_bn.Attr("groups").MappedPattern(op="depthwise_conv2d", name="groups")
        gcu_conv_bn.Attr("axis1").MappedPattern(op="elementwise_add", name="axis")
        gcu_conv_bn.Attr("axis2").MappedPattern(op="elementwise_mul", name="axis")
        gcu_conv_bn.Attr("axis3").MappedPattern(op="elementwise_add", name="axis")
        gcu_conv_bn.Attr("axis4").MappedPattern(op="elementwise_mul", name="axis")
        gcu_conv_bn.Attr("axis5").MappedPattern(op="elementwise_add", name="axis")
        return gcu_conv_bn.Output("Out")

    return pattern, replace


@ir.RegisterPass
def gcu_fuse_depthwise_conv_add_mul_add():
    def pattern(input, filter, y1, x1, y2):
        depthwise_conv2d = ir.PassDesc.OP.depthwise_conv2d(Input=input, Filter=filter)
        add1 = ir.PassDesc.OP.elementwise_add(X=depthwise_conv2d, Y=y1)
        mul1 = ir.PassDesc.OP.elementwise_mul(X=x1, Y=add1.Output("Out"))
        return ir.PassDesc.OP.elementwise_add(X=mul1.Output("Out"), Y=y2)

    def replace(input, filter, y1, x1, y2):
        logging.info("======= start to do depthwise conv add mul add op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUDepthwiseConvAddMulAdd(
            X=input, Filter=filter, Y1=y1, X1=x1, Y2=y2
        )
        gcu_conv_bn.Attr("strides").MappedPattern(
            op="depthwise_conv2d", name="strides", element_index=None
        )
        gcu_conv_bn.Attr("data_format").MappedPattern(
            op="depthwise_conv2d", name="data_format", element_index=None
        )
        gcu_conv_bn.Attr("dilations").MappedPattern(
            op="depthwise_conv2d", name="dilations", element_index=None
        )
        gcu_conv_bn.Attr("paddings").MappedPattern(
            op="depthwise_conv2d", name="paddings", element_index=None
        )
        gcu_conv_bn.Attr("padding_algorithm").MappedPattern(
            op="depthwise_conv2d", name="padding_algorithm", element_index=None
        )
        gcu_conv_bn.Attr("groups").MappedPattern(op="depthwise_conv2d", name="groups")
        gcu_conv_bn.Attr("axis1").MappedPattern(op="elementwise_add", name="axis")
        gcu_conv_bn.Attr("axis2").MappedPattern(op="elementwise_mul", name="axis")
        gcu_conv_bn.Attr("axis3").MappedPattern(op="elementwise_add", name="axis")
        return gcu_conv_bn.Output("Out")

    return pattern, replace
