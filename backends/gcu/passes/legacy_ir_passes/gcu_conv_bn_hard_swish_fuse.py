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
def gcu_fuse_depthwise_conv_bn_hard_swish():
    def pattern(input, filter, scale, bias, mean, var):
        depth_conv2d = ir.PassDesc.OP.depthwise_conv2d(Input=input, Filter=filter)
        bn = ir.PassDesc.OP.batch_norm(
            X=depth_conv2d, Bias=bias, Mean=mean, Scale=scale, Variance=var
        )
        return ir.PassDesc.OP.hard_swish(X=bn.Output("Y"))

    def replace(input, filter, scale, bias, mean, var):
        logging.info(
            "======= start to do depthwise_conv2d bn hard_swish op replace ======"
        )
        gcu_conv_bn = ir.PassDesc.OP.GCUConvBnHardSwish(
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
