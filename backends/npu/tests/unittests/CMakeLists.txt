# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()

set_tests_properties(${TEST_OPS} PROPERTIES TIMEOUT 1000)

set_tests_properties(test_tril_triu_op_npu PROPERTIES ENVIRONMENT
                                                      FLAGS_enable_pir_api=0)

set_tests_properties(test_check_nan_inf_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_npu_check_nan_inf=1)
set_tests_properties(test_adam_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_adamw_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_div_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_max_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_sub_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_mod_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_pow_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_elementwise_min_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_dropout_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_increment_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_gather_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_momentum_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_reduce_sum_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_rmsprop_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_sgd_op_npu PROPERTIES ENVIRONMENT
                                                FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_strided_slice_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_softmax_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_softmax_with_cross_entropy_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_USE_STANDALONE_EXECUTOR=0)
set_tests_properties(test_batch_norm_op_npu
                     PROPERTIES ENVIRONMENT FLAGS_npu_storage_format=0)
set_tests_properties(test_conv2d_op_npu PROPERTIES ENVIRONMENT
                                                   FLAGS_npu_storage_format=0)
set_tests_properties(
  test_transpose_op_npu PROPERTIES ENVIRONMENT
                                   FLAGS_allocator_strategy=auto_growth)
set_tests_properties(test_einsum_op_npu PROPERTIES ENVIRONMENT
                                                   FLAGS_npu_storage_format=0)
set_tests_properties(test_matmulv2_op_npu PROPERTIES ENVIRONMENT
                                                     FLAGS_npu_storage_format=0)
