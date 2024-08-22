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
# the License.

include(ExternalProject)

set(GIT_URL "https://github.com")
set(THIRD_PARTY_PATH
    "${CMAKE_BINARY_DIR}/third_party"
    CACHE STRING
          "A path setting third party libraries download & build directories.")

set(THIRD_PARTY_BUILD_TYPE Release)
set(third_party_deps)

# define submodule update macro
macro(check_update_submodule MODULE_NAME)
  message(
    STATUS
      "Run 'git submodule update --init ${MODULE_NAME}' in ${PADDLE_SOURCE_DIR}/third_party"
  )
  execute_process(
    COMMAND git submodule update --init ${MODULE_NAME}
    WORKING_DIRECTORY ${PADDLE_SOURCE_DIR}/third_party
    RESULT_VARIABLE result_var)
  if(NOT result_var EQUAL 0)
    message(
      FATAL_ERROR
        "Failed to get submodule '${MODULE_NAME}', please check your network !")
  endif()
endmacro()

# ########################## include third_party ###############################
if(NOT paddle_submodule)
  check_update_submodule(gflags)
  check_update_submodule(glog)
endif()
include(external/gflags) # gflags
include(external/glog) # glog

list(APPEND third_party_deps extern_gflags extern_glog)
if(NOT ON_INFER)
  if(NOT paddle_submodule)
    check_update_submodule(pybind)
  endif()
  include(external/pybind11) # pybind
  list(APPEND third_party_deps extern_pybind)
endif()

if(WITH_TESTING)
  if(NOT paddle_submodule)
    check_update_submodule(gtest)
  endif()
  include(external/gtest) # gtest
  list(APPEND third_party_deps extern_gtest)
endif()

if(WITH_MKL AND NOT WITH_ARM)
  if(NOT paddle_submodule)
    check_update_submodule(onednn)
  endif()
  include(external/onednn) # onednn
  list(APPEND third_party_deps extern_onednn)
endif()

add_custom_target(third_party ALL DEPENDS ${third_party_deps})
