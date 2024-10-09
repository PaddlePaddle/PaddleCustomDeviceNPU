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

cmake_minimum_required(VERSION 3.10)

project(paddle-intel-hpu CXX C)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_MKL "compile with mkl support" ON)
option(WITH_ARM "compile with arm support" OFF)

set(PLUGIN_NAME "paddle-intel-hpu")
set(PLUGIN_VERSION "0.0.1")

include(paddle)
include(generic)
include(version)

include_directories(${PADDLE_INC_DIR} ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/kernels /usr/include/habanalabs)
link_directories(${PADDLE_LIB_DIR} /usr/lib/habanalabs)

add_definitions(-std=c++14)

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  kernels/*.cc utils/*.cc)
list(APPEND PLUGIN_SRCS runtime/runtime.cc)

# build shared library
add_library(${PLUGIN_NAME} SHARED ${PLUGIN_SRCS})
if(ON_INFER)
  target_link_directories(${PLUGIN_NAME} PRIVATE ${PADDLE_INFERENCE_LIB_DIR})
  target_link_libraries(${PLUGIN_NAME} PRIVATE paddle_inference glog Synapse)
else()
  target_link_libraries(${PLUGIN_NAME} PRIVATE ${PADDLE_CORE_LIB} glog Synapse)
endif()

include(third_party)
add_dependencies(${PLUGIN_NAME} third_party)

# packing wheel package
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(
  TARGET ${PLUGIN_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating plugin dirrectories------>>>")

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${PLUGIN_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()
