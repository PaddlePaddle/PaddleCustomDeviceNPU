#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   For Paddle CI
#=================================================

set -ex

CODE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
export CODE_ROOT

# For paddle easy debugging
export FLAGS_call_stack_level=2

failed_test_lists=''
tmp_dir=`mktemp -d`

function collect_failed_tests() {
    for file in `ls $tmp_dir`; do
        exit_code=0
        grep -q 'The following tests FAILED:' $tmp_dir/$file||exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
}

function show_ut_retry_result() {
    SYSTEM=`uname -s`
    if [[ "$is_retry_execuate" != "0" ]]  && [[ "${exec_times}" == "0" ]] ;then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than ${parallel_failed_tests_exec_retry_threshold} failed unit tests in test, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    elif [[ "$is_retry_execuate" != "0" ]] && [[ "${exec_times}" == "1" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than 10 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=4) {print $2}}')
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-ran:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk BEGIN{RS=EOF}'{gsub(/\n/,"|");print}')
            echo "========================================"
            echo "There are failed tests, which have been executed re-run,but success rate is less than 50%:"
            echo "Summary Failed Tests... "
            echo "========================================"
            echo "The following tests FAILED: "
            echo "${retry_unittests_record}" | sort -u | grep -E "$failed_ut_re"
            exit 8;
        fi
    fi
}

function main() {
    # skip paddlepaddle cpu install as mlu docker image already have cpu whl package installed

    # custom_mlu build and install
    export MLU_VISIBLE_DEVICES=0,1
    export PADDLE_MLU_ALLOW_TF32=0
    export CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE=1
    cd ${CODE_ROOT}
    git submodule update --init
    bash tools/compile.sh
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    cd ${CODE_ROOT}/build
    pip install dist/*.whl --force-reinstall

    # get changed ut and kernels
    set +e
    changed_uts=$(git diff --name-only develop | grep "backends/mlu/tests/unittests")
    changed_ut_list=()
    if [ ${#changed_uts[*]} -gt 0 ]; then 
        for line in ${changed_uts[@]} ;
            do
                tmp=${line##*/}
                changed_ut=${tmp%.py*}
                changed_ut_list+=(${changed_ut})
            done
    fi

    # transform changed kernels to changed ut
    set +e
    changed_kernels=$(git diff --name-only develop | grep "backends/mlu/kernels")
    set +x
    all_ut_lists=$(ls "${CODE_ROOT}/tests/unittests")
    set -x
    if [ ${#changed_kernels[*]} -gt 0 ]; then 
        for line in ${changed_kernels[@]} ;
            do
                tmp=${line##*/}
                changed_kernel=${tmp%_kernel.cc*}
                changed_kernel_ut=$(echo "${all_ut_lists[@]}" | grep "${changed_kernel}")
                filtered_ut=${changed_kernel_ut%.py*}
                res=$(echo "${changed_ut_list[@]}" | grep "${filtered_ut}" | wc -l)
                if [ $res -eq 0 ]; then
                    changed_ut_list+=(${filtered_ut})
                fi
            done
    fi
    echo "changed_ut_list=${changed_ut_list[@]}"
    set -e
    # read disable ut list
    IFS=$'\n'
    disable_ut_mlu=$(cat "${CODE_ROOT}/tools/disable_ut_mlu")
    disable_ut_list=''
    while read -r line; do
        res=$(echo "${changed_ut_list[@]}" | grep "${line}" | wc -l)
        if [ $res -eq 0 ]; then
            disable_ut_list+="^"${line}"$|"
        else
            echo "Found ${line} code changed, ignore ut list disabled in disable_ut_mlu"
        fi
    done <<< "$disable_ut_mlu";
    disable_ut_list+="^disable_ut_mlu$"
    echo "disable_ut_list=${disable_ut_list}"

    # run ut
    ut_total_startTime_s=`date +%s`
    tmpfile_rand=`date +%s%N`
    tmpfile=$tmp_dir/$tmpfile_rand

    set +e
    
    NUM_PROC=8
    EXIT_CODE=0
    pids=()
    for (( i = 0; i < $NUM_PROC; i++ )); do
        mlu_list="$((i*2)),$((i*2+1))"
        (env MLU_VISIBLE_DEVICES=$mlu_list ctest -I $i,,$NUM_PROC --output-on-failure -E "($disable_ut_list)" -j1 | tee -a $tmpfile; test ${PIPESTATUS[0]} -eq 0)&
        pids+=($!)
    done
    
    for pid in "${pids[@]}"; do
        wait $pid
        status=$?
        if [ $status -ne 0 ]; then
            EXIT_CODE=8
        fi
    done
    
    set -e

    #ctest -E "($disable_ut_list)" --output-on-failure | tee $tmpfile;
    collect_failed_tests

    # add unit test retry for MLU
    rm -f $tmp_dir/*
    exec_times=0
    retry_unittests_record=''
    retry_time=4
    exec_time_array=('first' 'second' 'third' 'fourth')
    parallel_failed_tests_exec_retry_threshold=120
    exec_retry_threshold=30
    is_retry_execuate=0
    rerun_ut_startTime_s=`date +%s`

    if [ -n "$failed_test_lists" ];then
        need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/\s(.\+)//' | sed 's/- //' )
        need_retry_ut_arr=(${need_retry_ut_str})
        need_retry_ut_count=${#need_retry_ut_arr[@]}
        retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/\s(.\+)//' | sed 's/- //' )
        while ( [ $exec_times -lt $retry_time ] )
            do
                if [[ "${exec_times}" == "0" ]] ;then
                    if [ $need_retry_ut_count -lt $parallel_failed_tests_exec_retry_threshold ];then
                        is_retry_execuate=0
                    else
                        is_retry_execuate=1
                    fi
                elif [[ "${exec_times}" == "1" ]] ;then
                    need_retry_ut_str=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/\s(.\+)//' | sed 's/- //' )
                    need_retry_ut_arr=(${need_retry_ut_str})
                    need_retry_ut_count=${#need_retry_ut_arr[@]} 
                    if [ $need_retry_ut_count -lt $exec_retry_threshold ];then
                        is_retry_execuate=0
                    else
                        is_retry_execuate=1
                    fi
                fi
                if [[ "$is_retry_execuate" == "0" ]];then
                    set +e
                    retry_unittests_record="$retry_unittests_record$failed_test_lists"
                    failed_test_lists_ult=`echo "${failed_test_lists}" |grep -Po '[^ ].*$'`
                    set -e
                    if [[ "${exec_times}" == "1" ]] || [[ "${exec_times}" == "3" ]];then
                        if [[ "${failed_test_lists}" == "" ]];then
                            break
                        else
                            retry_unittests=$(echo "$failed_test_lists" | grep -oEi "\-.+\(.+\)" | sed 's/\s(.\+)//' | sed 's/- //' )
                        fi
                    fi
                    echo "========================================="
                    echo "This is the ${exec_time_array[$exec_times]} time to re-run"
                    echo "========================================="
                    echo "The following unittest will be re-run:"
                    echo "${retry_unittests}"                    
                    for line in ${retry_unittests[@]} ;
                        do
                            if [[ "$one_card_retry" == "" ]]; then
                                one_card_retry="^$line$"
                            else
                                one_card_retry="$one_card_retry|^$line$"
                            fi
                        done

                    if [[ "$one_card_retry" != "" ]]; then
                        ctest -R "$one_card_retry" --output-on-failure | tee $tmpfile;
                    fi
                    exec_times=$[$exec_times+1]
                    failed_test_lists=''
                    collect_failed_tests
                    rm -f $tmp_dir/*
                    one_card_retry=''
                else 
                    break
                fi

            done
        retry_unittests_record="$retry_unittests_record$failed_test_lists"
    fi
    rerun_ut_endTime_s=`date +%s` 
    echo "Rerun TestCases Total Time: $[ $rerun_ut_endTime_s - $rerun_ut_startTime_s ]s" 
    ut_total_endTime_s=`date +%s`
    echo "TestCases Total Time: $[ $ut_total_endTime_s - $ut_total_startTime_s ]s"
    if [[ "$EXIT_CODE" != "0" ]];then
        show_ut_retry_result
    fi

    # PaddleX test
    unset MLU_VISIBLE_DEVICES
    echo "Start Download"
    git clone --depth 1000 https://gitee.com/PaddlePaddle/PaddleX.git
    cd PaddleX
    pip install -e .
    paddlex --install PaddleClas
    paddlex --install PaddleSeg

    wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
    tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
    wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar -P ./dataset
    tar -xf ./dataset/seg_optic_examples.tar -C ./dataset/
    echo "End Download"

    echo "Start PaddleX ResNet50"
    python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o Global.output=resnet50_output \
    -o Global.device="mlu:0,1,2,3"

    python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./resnet50_output/best_model" \
    -o Predict.input_path="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg" \
    -o Global.device="mlu:0"
    echo "End PaddleX ResNet50"

    echo "Start DeepLabv3"
    python main.py -c paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R50.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/seg_optic_examples \
    -o Global.output=deeplabv3p_output \
    -o Global.device="mlu:0,1,2,3"

    python main.py -c paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./deeplabv3p_output/best_model/model/" \
    -o Predict.input_path="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_001.jpg" \
    -o Global.device="mlu:0"
    echo "End DeepLabv3"
}

main $@
