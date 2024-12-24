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
def gcu_fuse_mul_add():
    def pattern(x, y):
        mul = ir.PassDesc.OP.elementwise_mul(X=x, Y=y)
        return ir.PassDesc.OP.elementwise_add(X=x, Y=mul.Output("Out"))

    def replace(x, y):
        logging.info("======= start to do mul bias op replace ======")
        gcu_conv_bn = ir.PassDesc.OP.GCUMulAdd(X=x, Y=y)
        gcu_conv_bn.Attr("axis1").MappedPattern(op="elementwise_mul", name="axis")
        gcu_conv_bn.Attr("axis2").MappedPattern(op="elementwise_add", name="axis")
        return gcu_conv_bn.Output("Out")

    return pattern, replace
