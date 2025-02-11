#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# set -x 

# Format: model_name, model_path, hpu_number
models="Llama-2-7b-chat /models/meta-llama/Llama-2-7b-chat 1"
# Llama-65b /models/facebook/llama-65b 4
# Llama-65b /models/facebook/llama-65b 8

input_lengths=(128 512 1024 2048)
output_lengths=(128 512 1024 2048)
batch_sizes=(1 2 4 8 16 32 64)
dtype=bfloat16
device=intel_hpu:0

FORCE_PULL_REINSTALL=false
NEW_CSW_DATA=false

show_help() {
    echo "Usage: $0 [-f|-h]"
    echo
    echo "Options:"
    echo "  -f, --force       force pull latest code and reinstall"
    echo "  -c, --csv         save data to new csv file, default benchmark_paddlepaddle_data.csv"
    echo "  -h, --help        show help info"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_PULL_REINSTALL=true
            ;;
        -c|--csv)
            NEW_CSW_DATA=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

workspace=$(pwd)

if $FORCE_PULL_REINSTALL; then
    if ! pip list 2>/dev/null | grep -q paddlepaddle; then
        echo "paddlepaddle not found in pip list. Try to install..."
        pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
        if [ $? -ne 0 ]; then
            echo "Failed to install paddlepaddle. Exiting..."
            exit 1
        fi
    fi

    if [ ! -d "PaddleNLP" ]; then
        echo "PaddleNLP not found. Try to download..."
        git clone -b e2e_integration https://github.com/LeoZhao-Intel/PaddleNLP.git
        if [ $? -ne 0 ]; then
            echo "Failed to download PaddleNLP. Exiting..."
            exit 1
        fi
    fi

    if [ ! -d "PaddleCustomDevice" ]; then
        echo "PaddleCustomDevice not found. Try to download..."
        git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice
        if [ $? -ne 0 ]; then
            echo "Failed to download PaddleCustomDevice. Exiting..."
            exit 1
        fi
    fi

    cd $workspace
    echo "Pull latest code for PaddleNLP and install"
    cd PaddleNLP
    git pull
    pip install -e .
    if [ $? -ne 0 ]; then
        echo "Failed to install PaddleNLP. Exiting..."
        exit 1
    fi

    cd $workspace
    echo "Pull latest code for PaddleCustomDevice and install"
    cd PaddleCustomDevice/backends/intel_hpu/
    mkdir build
    cd build
    cmake ..
    make -j
    pip install --force-reinstall dist/paddle_intel_hpu*.whl
    if [ $? -ne 0 ]; then
        echo "Failed to install paddle_intel_hpu. Exiting..."
        exit 1
    fi

    cd $workspace
    cd PaddleCustomDevice/backends/intel_hpu/custom_ops
    python setup.py install
    if [ $? -ne 0 ]; then
        echo "Failed to install intel_hpu custom_ops. Exiting..."
        exit 1
    fi 
fi

cd $workspace

if $NEW_CSW_DATA; then
    csv_file="benchmark_paddlepaddle_data_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S).csv"
else
    csv_file="benchmark_paddlepaddle_data.csv"
fi

if [ ! -f "$csv_file" ]; then
    echo "model,HPU,input_len,output_len,batch_size,IPS,QPS,logfile,lastline" > "$csv_file"
fi

cd $workspace
cd PaddleNLP/llm/predict
while IFS=' ' read -r model_name model_path hpu_number; do
    for input_len in "${input_lengths[@]}"
    do
        for output_len in "${output_lengths[@]}"
        do
            for batch_size in "${batch_sizes[@]}"
            do
                echo ""
                echo "==========================================================="
                echo "Model: $model_name"
                echo "Path: $model_path"
                echo "input_len: $input_len"
                echo "output_len: $output_len"
                echo "batch_size: $batch_size"
                total_max_length=$((input_len + output_len))
                echo "total_max_length: $total_max_length"

                log_name_prefix=benchmark_paddlepaddle_${model_name}_datatype_${dtype}_batchsize_${batch_size}_inputlen_${input_len}_outputlen_${output_len}
                skip=false

                log_files=($workspace/${log_name_prefix}*.log)
                for log_file in "${log_files[@]}"; do
                    if [ -f "$log_file" ]; then
                        last_line=$(tail -n 1 "$log_file")
                        if [[ $last_line == *IPS* ]]; then
                            echo "skip the item with performance data: $last_line Log: $log_file"
                            skip=true
                            break
                        fi
                    fi
                done

                if [[ "$skip" == "true" ]]; then
                    continue
                fi

                log_name=${log_name_prefix}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)
                cmd="python3 predictor.py --model_name_or_path $model_path --inference_model --dtype $dtype --device $device --src_length $input_len --max_length $output_len --total_max_length $total_max_length --batch_size $batch_size --output_file ${log_name}.json --decode_strategy greedy_search --benchmark true"
                echo $cmd
                # eval $cmd 2>&1 | tee $workspace/${log_name}.log
                eval $cmd > $workspace/${log_name}.log 2>&1
                mv ${log_name}.json $workspace/ 2>/dev/null
                

                last_line=$(tail -n 1 "$workspace/${log_name}.log")
                if [[ $last_line != *IPS* ]]; then
                    echo "performance data: not found. Log: $workspace/${log_name}.log"
                    echo "$model_name,$hpu_number,$input_len,$output_len,$batch_size,'Error',,${log_name}.log, \"$last_line\"" >> $workspace/"$csv_file"
                    echo "break the batch size loop"
                    break 
                else
                    echo "performance data: $last_line Log: $workspace/${log_name}.log"
                    log_input_length=$(echo "$last_line" | grep -oP 'Input length is: \K[0-9]+')
                    log_output_length=$(echo "$last_line" | grep -oP 'Output length is: \K[0-9]+')
                    log_batch_size=$(echo "$last_line" | grep -oP 'bs is: \K[0-9]+')
                    log_ips=$(echo "$last_line" | grep -oP 'IPS: \K[0-9.]+')
                    log_qps=$(echo "$last_line" | grep -oP 'QPS: \K[0-9.]+')
                    if [ "$log_input_length" -ne "$input_len" ] || [ "$log_output_length" -ne "$output_len" ] || [ "$log_batch_size" -ne "$batch_size" ]; then
                        echo "$model_name,$hpu_number,$input_len,$output_len,$batch_size, 'Not Match',${log_name}.log, \"$last_line\"" >> $workspace/"$csv_file"
                    else
                        echo "$model_name,$hpu_number,$input_len,$output_len,$batch_size,$log_ips,$log_qps,${log_name}.log, \"$last_line\"" >> $workspace/"$csv_file"
                    fi
                fi
            done
        done
    done
done <<< "$models"
