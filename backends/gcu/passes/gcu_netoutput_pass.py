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
def add_netoutput_op_pass():
    def pattern(input):
        fetch = ir.PassDesc.OP.fetch(X=input)
        return fetch

    def replace(input):
        proc = ir.PassDesc.OP.gcu_netoutput(X=input)
        proc.SetAttr("origin_out_dtype", input.dtype)
        proc_var = proc.Output("Out")[0]
        proc_var._set_attr("type", input.type)
        proc_var._set_attr("dtype", input.dtype)
        proc_var._set_attr("shape", input.shape)
        fetch = ir.PassDesc.OP.fetch(X=proc_var)
        fetch.Attr("col").MappedPattern(op="fetch", name="col", index=0)
        return fetch

    return pattern, replace
