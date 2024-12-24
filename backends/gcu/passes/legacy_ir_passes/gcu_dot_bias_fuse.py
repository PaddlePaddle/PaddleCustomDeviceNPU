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
def gcu_fuse_dot_bias():
    def pattern(x, y1, y2):
        mul = ir.PassDesc.OP.matmul_v2(X=x, Y=y1)
        return ir.PassDesc.OP.elementwise_add(X=mul.Output("Out"), Y=y2)

    def replace(x, y1, y2):
        logging.info("======= start to do dot bias op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUDotBias(X=x, Y=y1, Y2=y2)
        gcu_conv_bn.Attr("trans_x").MappedPattern(op="matmul_v2", name="trans_x")
        gcu_conv_bn.Attr("trans_y").MappedPattern(op="matmul_v2", name="trans_y")
        gcu_conv_bn.Attr("axis").MappedPattern(op="elementwise_add", name="axis")
        return gcu_conv_bn.Output("Out")

    return pattern, replace
