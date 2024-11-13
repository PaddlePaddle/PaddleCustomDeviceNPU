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

function(py_test_modules TARGET_NAME RUN_JIT_KERNELS)
  set(options SERIAL)
  set(oneValueArgs "")
  set(multiValueArgs MODULES DEPS ENVS)
  cmake_parse_arguments(py_test_modules "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  add_test(
    NAME ${TARGET_NAME}
    COMMAND
      ${CMAKE_COMMAND} -E env PADDLE_GCU_USE_JIT_KERNELS_ONLY=${RUN_JIT_KERNELS}
      FLAGS_use_stride_kernel=false ${py_test_modules_ENVS} python
      ${CMAKE_BINARY_DIR}/tests/${py_test_modules_MODULES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if(py_test_modules_SERIAL)
    set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
  endif()
endfunction()

add_subdirectory(unittests)
# add_subdirectory(unittests_jit)
add_subdirectory(fuse_pass)
