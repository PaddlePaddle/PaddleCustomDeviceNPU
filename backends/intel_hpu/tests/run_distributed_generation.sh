# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# /bin/sh

#set -ex

world_size="4"
model_name="meta-llama/Llama-2-13b-chat"
batch_size="4"
max_new_tokens="100"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --world_size) world_size="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --max_new_tokens) max_new_tokens="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure world_size is one of 1, 2, 4, or 8
if [ "$world_size" -le 1 ]; then
    world_size=1
elif [ "$world_size" -le 2 ]; then
    world_size=2
elif [ "$world_size" -le 4 ]; then
    world_size=4
else
    world_size=8
fi

# Generate the --devices string based on world_size
devices=""
for ((i=0; i<${world_size}; i++)); do
    if [ $i -ne 0 ]; then
        devices+=","
    fi
    devices+="$i"
done

echo "devices: ${devices}"

export LOG_LEVEL_ALL=0
export HABANA_LOGS=./logs

# export HCCL_COMM_ID=127.0.0.1:5555
export INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
# PYTHONPATH=../../:$PYTHONPATH  \
export FLAGS_intel_hpu_runtime_debug=0

# export HABANA_PROFILE=1
# export HABANA_PROFILE_WRITE_HLTV_WITH_HOST=1

# export GRAPH_VISUALIZATION=1
# export ENABLE_EXPERIMENTAL_FLAGS=1
# export VISUALIZATION_MODE=0

#GLOG_v=10

python -m paddle.distributed.launch --devices "${devices}" run_generation.py --world_size ${world_size} --model_name ${model_name} --batch_size ${batch_size} --max_new_tokens ${max_new_tokens}
