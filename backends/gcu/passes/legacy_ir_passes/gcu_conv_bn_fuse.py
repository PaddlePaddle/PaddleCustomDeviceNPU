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
def gcu_fuse_conv_bn():
    def pattern(input, filter, scale, bias, mean, var):
        conv2d = ir.PassDesc.OP.conv2d(Input=input, Filter=filter)
        bn = ir.PassDesc.OP.batch_norm(
            X=conv2d, Bias=bias, Mean=mean, Scale=scale, Variance=var
        )
        return bn.Output("Y")

    def replace(input, filter, scale, bias, mean, var):
        logging.info("======= start to do conv bn op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBn(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
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
        gcu_conv_bn.Attr("momentum").MappedPattern(op="batch_norm", name="momentum")
        gcu_conv_bn.Attr("epsilon").MappedPattern(op="batch_norm", name="epsilon")
        gcu_conv_bn.Attr("data_layout").MappedPattern(
            op="batch_norm", name="data_layout"
        )
        gcu_conv_bn.Attr("is_test").MappedPattern(op="batch_norm", name="is_test")
        gcu_conv_bn.Attr("trainable_statistics").MappedPattern(
            op="batch_norm", name="trainable_statistics"
        )
        gcu_conv_bn.Attr("use_global_stats").MappedPattern(
            op="batch_norm", name="use_global_stats"
        )

        gcu_conv_bn.SetOutputs(
            MeanOut=mean, SavedMean=mean, VarianceOut=var, SavedVariance=var
        )
        return gcu_conv_bn.Output("Out")

    return pattern, replace


@ir.RegisterPass
def gcu_fuse_conv_bn_swish():
    def pattern(input, filter, scale, bias, mean, var):
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBn(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
        )
        return ir.PassDesc.OP.swish(X=gcu_conv_bn.Output("Out"))

    def replace(input, filter, scale, bias, mean, var):
        logging.info("======= start to do conv bn swish op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBnSwish(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
        )
        gcu_conv_bn.Attr("strides").MappedPattern(
            op="GCUConvBn", name="strides", element_index=None
        )
        gcu_conv_bn.Attr("data_format").MappedPattern(
            op="GCUConvBn", name="data_format", element_index=None
        )
        gcu_conv_bn.Attr("dilations").MappedPattern(
            op="GCUConvBn", name="dilations", element_index=None
        )
        gcu_conv_bn.Attr("paddings").MappedPattern(
            op="GCUConvBn", name="paddings", element_index=None
        )
        gcu_conv_bn.Attr("padding_algorithm").MappedPattern(
            op="GCUConvBn", name="padding_algorithm", element_index=None
        )
        gcu_conv_bn.Attr("groups").MappedPattern(op="GCUConvBn", name="groups")
        gcu_conv_bn.Attr("momentum").MappedPattern(op="GCUConvBn", name="momentum")
        gcu_conv_bn.Attr("epsilon").MappedPattern(op="GCUConvBn", name="epsilon")
        gcu_conv_bn.Attr("data_layout").MappedPattern(
            op="GCUConvBn", name="data_layout"
        )
        gcu_conv_bn.Attr("is_test").MappedPattern(op="GCUConvBn", name="is_test")
        gcu_conv_bn.Attr("trainable_statistics").MappedPattern(
            op="GCUConvBn", name="trainable_statistics"
        )
        gcu_conv_bn.Attr("use_global_stats").MappedPattern(
            op="GCUConvBn", name="use_global_stats"
        )

        gcu_conv_bn.SetOutputs(
            MeanOut=mean, SavedMean=mean, VarianceOut=var, SavedVariance=var
        )
        return gcu_conv_bn.Output("Out")

    return pattern, replace


# //  has_shortcut = True:       else:
# //          X                         X
# //        /                         /
# //      |       |                 |       |
# //    CONV1     |               CONV1     |
# //      |       |                 |       |
# //     BN1      |                BN1      |
# //      |       |                 |       |
# //    RELU1     |               RELU1     |
# //      |       |                 |       |
# //    CONV2   CONV3             CONV2     |
# //      |       |                 |       |
# //     BN2     BN3               BN2      |
# //      \       /                 \       /
# //         ADD                       ADD
# //          |                         |
# //         RELU                      RELU
# //          |                         |
# //          Y                         Y


@ir.RegisterPass
def gcu_fuse_conv_bn_relu():
    def pattern(input, filter, scale, bias, mean, var):
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBn(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
        )
        return ir.PassDesc.OP.relu(X=gcu_conv_bn.Output("Out"))

    def replace(input, filter, scale, bias, mean, var):
        logging.info("======= start to relpace one input ======")
        gcu_conv_bn_relu = ir.PassDesc.OP.GCUConvBnRelu(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
        )
        gcu_conv_bn_relu.Attr("strides").MappedPattern(
            op="GCUConvBn", name="strides", element_index=None
        )
        gcu_conv_bn_relu.Attr("data_format").MappedPattern(
            op="GCUConvBn", name="data_format", element_index=None
        )
        gcu_conv_bn_relu.Attr("dilations").MappedPattern(
            op="GCUConvBn", name="dilations", element_index=None
        )
        gcu_conv_bn_relu.Attr("paddings").MappedPattern(
            op="GCUConvBn", name="paddings", element_index=None
        )
        gcu_conv_bn_relu.Attr("padding_algorithm").MappedPattern(
            op="GCUConvBn", name="padding_algorithm", element_index=None
        )
        gcu_conv_bn_relu.Attr("groups").MappedPattern(op="GCUConvBn", name="groups")
        gcu_conv_bn_relu.Attr("momentum").MappedPattern(op="GCUConvBn", name="momentum")
        gcu_conv_bn_relu.Attr("epsilon").MappedPattern(op="GCUConvBn", name="epsilon")
        gcu_conv_bn_relu.Attr("data_layout").MappedPattern(
            op="GCUConvBn", name="data_layout"
        )
        gcu_conv_bn_relu.Attr("is_test").MappedPattern(op="GCUConvBn", name="is_test")
        gcu_conv_bn_relu.Attr("trainable_statistics").MappedPattern(
            op="GCUConvBn", name="trainable_statistics"
        )
        gcu_conv_bn_relu.Attr("use_global_stats").MappedPattern(
            op="GCUConvBn", name="use_global_stats"
        )

        gcu_conv_bn_relu.SetOutputs(
            MeanOut=mean, SavedMean=mean, VarianceOut=var, SavedVariance=var
        )
        return gcu_conv_bn_relu.Output("Out")

    return pattern, replace


@ir.RegisterPass
def gcu_fuse_conv_bn_hard_swish():
    def pattern(input, filter, scale, bias, mean, var):
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBn(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
        )
        return ir.PassDesc.OP.hard_swish(X=gcu_conv_bn.Output("Out"))

    def replace(input, filter, scale, bias, mean, var):
        logging.info("======= start to do conv bn hard_swish op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBnHardSwish(
            X=input, Filter=filter, Mean=mean, Bias=bias, Scale=scale, Var=var
        )
        gcu_conv_bn.Attr("strides").MappedPattern(
            op="GCUConvBn", name="strides", element_index=None
        )
        gcu_conv_bn.Attr("data_format").MappedPattern(
            op="GCUConvBn", name="data_format", element_index=None
        )
        gcu_conv_bn.Attr("dilations").MappedPattern(
            op="GCUConvBn", name="dilations", element_index=None
        )
        gcu_conv_bn.Attr("paddings").MappedPattern(
            op="GCUConvBn", name="paddings", element_index=None
        )
        gcu_conv_bn.Attr("padding_algorithm").MappedPattern(
            op="GCUConvBn", name="padding_algorithm", element_index=None
        )
        gcu_conv_bn.Attr("groups").MappedPattern(op="GCUConvBn", name="groups")
        gcu_conv_bn.Attr("momentum").MappedPattern(op="GCUConvBn", name="momentum")
        gcu_conv_bn.Attr("epsilon").MappedPattern(op="GCUConvBn", name="epsilon")
        gcu_conv_bn.Attr("data_layout").MappedPattern(
            op="GCUConvBn", name="data_layout"
        )
        gcu_conv_bn.Attr("is_test").MappedPattern(op="GCUConvBn", name="is_test")
        gcu_conv_bn.Attr("trainable_statistics").MappedPattern(
            op="GCUConvBn", name="trainable_statistics"
        )
        gcu_conv_bn.Attr("use_global_stats").MappedPattern(
            op="GCUConvBn", name="use_global_stats"
        )

        gcu_conv_bn.SetOutputs(
            MeanOut=mean, SavedMean=mean, VarianceOut=var, SavedVariance=var
        )
        return gcu_conv_bn.Output("Out")

    return pattern, replace
