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
from paddle.distributed import fleet
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

paddle.set_device("intel_hpu")

model = "meta-llama/Llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model)
strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 2,
    "pp_degree": 1,
    "sharding_degree": 1,
}
fleet.init(is_collective=True, strategy=strategy)
hcg = fleet.get_hybrid_communicate_group()
tensor_parallel_rank = hcg.get_model_parallel_rank()

model = AutoModelForCausalLM.from_pretrained(
    model,
    tensor_parallel_degree=2,
    tensor_parallel_rank=tensor_parallel_rank,
    dtype="float32",
)
input_features = tokenizer("I am a student", return_tensors="pd")

outputs = model.generate(**input_features, max_length=20)
print(outputs[0])
print(tokenizer.batch_decode(outputs[0]))

"""
# example (multi cards):

export INTEL_HPU_VISIBLE_DEVICES=0,1
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
# export FLAGS_intel_hpu_runtime_debug=1
# echo $INTEL_HPU_VISIBLE_DEVICES
python -m paddle.distributed.launch --devices "0,1" test_llama.py
"""
