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
def conv2d_depthwise_elementwise_add_fuse_pass():
    def pattern(input, filter, bias):
        conv_t = ir.PassDesc.OP.depthwise_conv2d(Input=input, Filter=filter)
        add = ir.PassDesc.OP.elementwise_add(X=conv_t, Y=bias)
        return add

    def replace(input, filter, bias):
        conv_depthwise_add = ir.PassDesc.OP.fused_conv2d_add_act(
            Input=input, Filter=filter, Bias=bias
        )
        conv_depthwise_add.Attr("strides").MappedPattern(
            op="depthwise_conv2d", name="strides", index=0
        )
        conv_depthwise_add.Attr("paddings").MappedPattern(
            op="depthwise_conv2d", name="paddings", index=0
        )
        conv_depthwise_add.Attr("padding_algorithm").MappedPattern(
            op="depthwise_conv2d", name="padding_algorithm", index=0
        )
        conv_depthwise_add.Attr("dilations").MappedPattern(
            op="depthwise_conv2d", name="dilations", index=0
        )
        conv_depthwise_add.Attr("groups").MappedPattern(
            op="depthwise_conv2d", name="groups", index=0
        )
        conv_depthwise_add.Attr("data_format").MappedPattern(
            op="depthwise_conv2d", name="data_format", index=0
        )
        conv_depthwise_add.Attr("exhaustive_search").MappedPattern(
            op="depthwise_conv2d", name="exhaustive_search", index=0
        )
        conv_depthwise_add.Attr("workspace_size_MB").MappedPattern(
            op="depthwise_conv2d", name="workspace_size_MB", index=0
        )
        conv_depthwise_add.Attr("fuse_alpha").MappedPattern(
            op="depthwise_conv2d", name="fuse_alpha", index=0
        )
        conv_depthwise_add.SetAttr("split_channels", [])
        conv_depthwise_add.SetAttr("activation", "identity")
        return conv_depthwise_add.Output("Output")

    return pattern, replace
