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

cmake_minimum_required(VERSION 3.10)

project(paddle-custom-npu CXX C)
set(CUSTOM_NPU_NAME "paddle-custom-npu")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(paddle)
include(version)
include(generic)
include(external/ascend)
include_directories(${CMAKE_SOURCE_DIR})

option(WITH_TESTING "compile with unit testing" OFF)
option(WITH_ATB "compile with ascend transformer boost" ON)
option(WITH_MKL "compile with mkl support" ON)
option(WITH_ARM "compile with arm support" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_COVERAGE "Compile PaddlePaddle with code coverage" OFF)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wno-narrowing -Wno-terminate -Wno-write-strings -Wno-return-type"
)
# Note: _GLIBCXX_USE_CXX11_ABI should be same with libpaddle.so, default is 1
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

if(WITH_COVERAGE)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
endif()
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
# custom runtime
set(CUSTOM_NPU_SRCS runtime/runtime.cc)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
if(WITH_ARM)
  add_definitions(-DPADDLE_WITH_ARM)
endif()

# custom kernels
file(
  GLOB_RECURSE CUSTOM_KERNEL_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc)
list(APPEND CUSTOM_NPU_SRCS ${CUSTOM_KERNEL_SRCS})

# custom op with kernel
file(
  GLOB_RECURSE CUSTOM_OPERATOR_SRCS FOLLOW_SYMLINKS
  RELATIVE ${CMAKE_SOURCE_DIR}
  custom_op/*.cc)
message(STATUS "CUSTOM_OPERATOR_SRCS=${CUSTOM_OPERATOR_SRCS}")
list(APPEND CUSTOM_NPU_SRCS ${CUSTOM_OPERATOR_SRCS})

if(ON_INFER)
  link_directories(${PADDLE_INFERENCE_LIB_DIR})
endif()

# build shared library
add_library(${CUSTOM_NPU_NAME} SHARED ${CUSTOM_NPU_SRCS})

# link ascend libs
add_dependencies(${CUSTOM_NPU_NAME} ascend_cl)
target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE ${ascendcl_lib})
target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE ${ascend_hccl_lib})
target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE ${acl_op_compiler_lib})

# link atb libs
if(WITH_ATB)
  add_definitions(-DPADDLE_WITH_ATB)
  target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE ${ascend_atb_lib})
  target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE ${ascend_ops_lib})
endif()

# link third_party
include(third_party)
add_dependencies(${CUSTOM_NPU_NAME} third_party)
target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE gflags glog)

# link paddle libs
if(ON_INFER)
  target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE paddle_inference)
  target_compile_definitions(${CUSTOM_NPU_NAME} PRIVATE PADDLE_ON_INFERENCE)
else()
  target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE ${PADDLE_CORE_LIB})
  target_link_libraries(${CUSTOM_NPU_NAME} PRIVATE pybind)
endif()

# testing
if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../python")
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()

# get git commit id
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Git commit id is: ${GIT_HASH}")

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/custom_op
  OUTPUT_VARIABLE CUSTOM_OP_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Custom op git commit id is: ${CUSTOM_OP_GIT_HASH}")

# configure setup.py
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# packing wheel package
add_custom_command(
  TARGET ${CUSTOM_NPU_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${CUSTOM_NPU_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating custom device directories------>>>")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/npu/passes
  COMMAND ${CMAKE_COMMAND} -E remove -f
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/npu/passes
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/passes/*
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/npu/passes
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${CUSTOM_NPU_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
