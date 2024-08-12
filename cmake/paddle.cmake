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

if(NOT PYTHON_VERSION)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
  find_package(
    Python ${PYTHON_VERSION} REQUIRED
    COMPONENTS Interpreter Development
    EXACT)
endif()

message(STATUS "Python_EXECUTABLE is ${Python_EXECUTABLE}")
include_directories(${Python_INCLUDE_DIRS})

if(DEFINED ENV{PADDLE_CUSTOM_PATH})
  set(PADDLE_DIR $ENV{PADDLE_CUSTOM_PATH})
else()
  execute_process(
    COMMAND
      "env" "CUSTOM_DEVICE_ROOT=\"\"" "${Python_EXECUTABLE}" "-c"
      "import re, paddle; print(re.compile('/__init__.py.*').sub('',paddle.__file__))"
    OUTPUT_VARIABLE PADDLE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT EXISTS ${PADDLE_DIR})
  message(FATAL_ERROR "NO Installed Paddle Found in ${PADDLE_DIR}")
endif()

set(PADDLE_INC_DIR "${PADDLE_DIR}/include/")
set(PADDLE_LIB_DIR "${PADDLE_DIR}/fluid/")

if(NOT EXISTS ${PADDLE_LIB_DIR})
  set(PADDLE_LIB_DIR "${PADDLE_DIR}/base/")
endif()

include_directories(${PADDLE_INC_DIR})

if(EXISTS "${PADDLE_LIB_DIR}/libpaddle.so")
  set(paddle_lib_name libpaddle.so)
elseif(EXISTS "${PADDLE_LIB_DIR}/core_avx.so")
  set(paddle_lib_name core_avx.so)
else()
  set(paddle_lib_name core_noavx.so)
  message(WANRING "Cannot find core_avx.so, using core_noavx.so instead.")
endif()

find_library(PADDLE_CORE_LIB ${paddle_lib_name} PATHS ${PADDLE_LIB_DIR})
if(NOT PADDLE_CORE_LIB)
  message(FATAL "${paddle_lib_name} NOT found in ${PADDLE_LIB_DIR}")
else()
  message(STATUS "PADDLE_CORE_LIB: ${PADDLE_CORE_LIB}")
endif()

if(NO_PADDLE_SUBMODULE)
  return()
endif()

# submodule Paddle first
get_filename_component(REPO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../"
                       ABSOLUTE)
message(STATUS "Run 'git submodule update --init Paddle' in ${REPO_SOURCE_DIR}")
execute_process(
  COMMAND git submodule update --init Paddle
  WORKING_DIRECTORY ${REPO_SOURCE_DIR}
  RESULT_VARIABLE result_var)
if(NOT result_var EQUAL 0)
  message(
    FATAL_ERROR "Failed to get submodule Paddle', please check your network !")
endif()

get_filename_component(PADDLE_SOURCE_DIR "${REPO_SOURCE_DIR}/Paddle" ABSOLUTE)
message(STATUS "PADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR}")
