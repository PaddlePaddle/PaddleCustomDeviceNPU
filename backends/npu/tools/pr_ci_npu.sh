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


if [ -z ${PADDLE_BRANCH} ]; then
    PADDLE_BRANCH="develop"
fi

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
EXIT_CODE=0;
function caught_error() {
 for job in `jobs -p`; do
        # echo "PID => ${job}"
        if ! wait ${job} ; then
            echo "At least one test failed with exit code => $?" ;
            EXIT_CODE=1;
        fi
    done
}


function card_test() {
    set -m
    ut_startTime_s=`date +%s`

    testcases=$1
    cardnumber=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}

    # get the NPU device count, XPU device count is one
    ascend_rt_visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-0}

    if [[ $ascend_rt_visible_devices =~ ([0-9,]+) ]]; then
        extracted_numbers="${BASH_REMATCH[1]}"
        extracted_numbers="${extracted_numbers//,/ }"
        numbers_array=($extracted_numbers)
        NPU_DEVICE_COUNT=${#numbers_array[@]}
        echo "The visible cards for the current task are: $extracted_numbers"
        echo "card_nums: ${NPU_DEVICE_COUNT}"
    fi

    if (( $cardnumber == -1 ));then
        cardnumber=$NPU_DEVICE_COUNT
    fi


    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    trap 'caught_error' CHLD
    tmpfile_rand=`date +%s%N`
    NUM_PROC=$[NPU_DEVICE_COUNT/$cardnumber]
    for (( i = 0; i < $NUM_PROC; i++ )); do
        npu_list=()
        for (( j = 0; j < cardnumber; j++ )); do
            if [ $j -eq 0 ]; then
                    npu_list=("${numbers_array[$[i*cardnumber]]}")
                else
                    npu_list="$npu_list,${numbers_array[$[i*cardnumber+j]]}"
            fi
        done
        tmpfile=$tmp_dir/$tmpfile_rand"_"$ix
        if [[ $cardnumber == $NPU_DEVICE_COUNT ]]; then
           echo "================"
           echo ASCEND_RT_VISIBLE_DEVICES=$npu_list
           echo "================"
           (ctest -I $i,,$NUM_PROC -R "($testcases)" | tee $tmpfile;test ${PIPESTATUS[0] -eq 0}) &
        else
           echo "================"
           echo ASCEND_RT_VISIBLE_DEVICES=$npu_list
           echo "================"
           (env ASCEND_RT_VISIBLE_DEVICES=$npu_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_list)" --output-on-failure | tee $tmpfile; test "${PIPESTATUS[0]}" -eq 0) &
        fi
    done
    wait; # wait for all subshells to finish
    ut_endTime_s=`date +%s`
    echo "Run TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    set +m
}


function main() {
    # skip paddlepaddle cpu install as npu docker image already have cpu whl package installed

    # custom_npu build and install
    cd ${CODE_ROOT}
    bash tools/compile.sh
    if [[ "$?" != "0" ]];then
        exit 7;
    fi
    cd ${CODE_ROOT}/build
    pip install dist/*.whl
    # get changed ut and kernels
    set +e
    changed_uts=$(git diff --name-only ${PADDLE_BRANCH} | grep "backends/npu/tests/unittests")
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
    changed_kernels=$(git diff --name-only ${PADDLE_BRANCH} | grep "backends/npu/kernels")
    set +x
    all_ut_lists=$(ls "${CODE_ROOT}/tests/unittests")
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
    IFS_DEFAULT=$IFS
    IFS=$'\n'
    if [ $(lspci | grep d801 | wc -l) -ne 0 ]; then
      disable_ut_npu=$(cat "${CODE_ROOT}/tools/disable_ut_npu")
    elif [ $(lspci | grep d802 | wc -l) -ne 0 ]; then
      disable_ut_npu=$(cat "${CODE_ROOT}/tools/disable_ut_npu_910b")
    else
      echo "Please make sure Ascend 910A or 910B NPUs exists!"
      exit 1
    fi
    important_ut_npu=$(cat "${CODE_ROOT}/tools/important_ut_npu")
    disable_ut_list=''
    while read -r line; do
        res=$(echo "${changed_ut_list[@]}" | grep "${line}" | wc -l)
        if [ $res -eq 0 ]; then
            disable_ut_list+="^"${line}"$|"
        else
            echo "Found ${line} code changed, ignore ut list disabled in disable_ut_npu"
        fi
    done <<< "$disable_ut_npu";
    disable_ut_list+="^disable_ut_npu$"

    if [ "${TEST_IMPORTANT:-OFF}" == "OFF" ]; then
        disable_ut_list+="|"
        while read -r line; do
            res=$(echo "${changed_ut_list[@]}" | grep "${line}" | wc -l)
            if [ $res -eq 0 ]; then
                disable_ut_list+="^"${line}"$|"
            else
                echo "Found ${line} code changed, ignore ut list disabled in disable_ut_npu"
            fi
        done <<< "$important_ut_npu";
        disable_ut_list+="^important_ut_npu$"
    fi
    
    echo "disable_ut_list=${disable_ut_list}"
    IFS=$IFS_DEFAULT
    if [ "${TEST_IMPORTANT:-OFF}" == "OFF" ]; then
        test_cases=$(ctest -N -V)
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
            matchstr=$(echo $line|grep -oEi 'Test[ \t]+#') || true
            if [[ "$matchstr" == "" ]]; then
                continue
            fi
            testcase=$(echo "$line"|grep -oEi "\w+$")
            if [[ "$single_card_tests" == "" ]]; then
                single_card_tests="^$testcase$"
            else
                single_card_tests="$single_card_tests|^$testcase$"
            fi
        done <<< "$test_cases";
    else
        while read -r line; do
            if [[ "$line" == "" ]]; then
                continue
            fi
            if [[ "$single_card_tests" == "" ]]; then
                single_card_tests="^$line$"
            else
                single_card_tests="$single_card_tests|^$line$"
            fi
        done <<< "$important_ut_npu";
    fi
    

    # run ut
    ut_total_startTime_s=`date +%s`
    card_test ${single_card_tests} 1
    collect_failed_tests

    # add unit test retry for NPU
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
    if [[ "${WITH_COVERAGE:-OFF}" == "ON" ]];then
        bash ${CODE_ROOT}/tools/coverage/coverage_process.sh
    fi
}

main $@
