# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
  GLOB TEST_FUSE_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_FUSE_PASSES "${TEST_FUSE_PASSES}")

foreach(TEST_PASS ${TEST_FUSE_PASSES})
  py_test_modules(${TEST_PASS} false MODULES fuse_pass/${TEST_PASS}.py)
  message(STATUS "with fuse pass: ${TEST_PASS}")
endforeach()

set_tests_properties(${TEST_FUSE_PASSES} PROPERTIES TIMEOUT 1000)
