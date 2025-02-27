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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(paddle-custom-gcu CXX C)
set(CUSTOM_GCU_NAME "paddle-custom-gcu")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if(DEFINED PY_VERSION)
  message(STATUS "User define PY_VERSION: ${PY_VERSION}")
else()
  set(PY_VERSION "3.10")
  message(STATUS "Use default PY_VERSION: ${PY_VERSION}")
endif()
set(PYTHON_VERSION ${PY_VERSION})
set(Python_EXECUTABLE "python${PY_VERSION}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")

include(paddle)
include(version)
include(generic)
include(external/gcu)
include(external/topscc)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(/opt/tops/include)
include_directories(${PADDLE_INC_DIR}/build)

option(WITH_KERNELS "compile with custom kernels" ON)
option(WITH_TESTING "compile with unit testing" OFF)
option(WITH_MKL "compile with mkl support" ON)
option(WITH_ARM "compile with arm support" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

# custom runtime
set(CUSTOM_GCU_SRCS runtime/runtime.cc)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
if(WITH_ARM)
  add_definitions(-DPADDLE_WITH_ARM)
endif()

# custom kernels
if(WITH_KERNELS)
  add_compile_definitions(HLIR_BUILDER_ABI_COMPATIABLE)
  file(
    GLOB_RECURSE CUSTOM_KERNEL_SRCS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    kernels/*.cc)
  list(APPEND CUSTOM_GCU_SRCS ${CUSTOM_KERNEL_SRCS})
  file(
    GLOB_RECURSE GCU_KERNEL_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    backend/*.cc common/*.cc)
  # build static library
  set(GCU_DEPENDS_NAME "gcu-kernels-depend")
  add_library(${GCU_DEPENDS_NAME} STATIC ${GCU_KERNEL_DEPENDS})
  set_target_properties(${GCU_DEPENDS_NAME} PROPERTIES CXX_VISIBILITY_PRESET
                                                       hidden)
  target_compile_options(${GCU_DEPENDS_NAME} PUBLIC -fPIC)
  add_dependencies(${GCU_DEPENDS_NAME} third_party)

  # custom op with kernel
  file(
    GLOB_RECURSE CUSTOM_CUSTOM_OP_SRCS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    custom_op/*.cc)
  list(APPEND CUSTOM_GCU_SRCS ${CUSTOM_CUSTOM_OP_SRCS})
endif()

file(
  GLOB_RECURSE CUSTOM_PASSES_SRCS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  passes/*.cc)
list(APPEND CUSTOM_GCU_SRCS ${CUSTOM_PASSES_SRCS})

file(
  GLOB_RECURSE CUSTOM_ENGINE_SRCS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  custom_engine/*.cc)
list(APPEND CUSTOM_GCU_SRCS ${CUSTOM_ENGINE_SRCS})

if(ON_INFER)
  link_directories(${PADDLE_INFERENCE_LIB_DIR})
endif()

# topscc kernels
file(GLOB_RECURSE TOPSCC_KERNEL_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/topscc_custom_kernels/*.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/topscc_custom_kernels/*.cpp)
list(LENGTH TOPSCC_KERNEL_SRCS TOPSCC_KERNEL_SRCS_LEN)
unset(TOPSCC_LIBS)
if(${TOPSCC_KERNEL_SRCS_LEN} GREATER 0)
  topscc_compile(FALSE ${TOPSCC_KERNEL_SRCS} "-shared -fPIC -O3 " TOPSCC_LIBS)
  message(STATUS "TOPSCC_LIBS: ${TOPSCC_LIBS}")
endif()

# build shared library
add_library(${CUSTOM_GCU_NAME} SHARED ${CUSTOM_GCU_SRCS})

# for link topscc custom kernels
set_target_properties(
  ${CUSTOM_GCU_NAME}
  PROPERTIES
    LINK_FLAGS
    "-Wl,-rpath,/usr/local/lib/python${PY_VERSION}/dist-packages/paddle_custom_device/gcu/"
)

target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE ${GCU_LIBS})
if(WITH_KERNELS)
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE ${GCU_DEPENDS_NAME})
endif()
if(TOPS_MODULE_SOURCE_DIR)
  target_include_directories(${CUSTOM_GCU_NAME}
                             PRIVATE ${TOPS_MODULE_SOURCE_DIR}/usr/include)
  target_include_directories(${CUSTOM_GCU_NAME}
                             PRIVATE ${TOPS_MODULE_SOURCE_DIR}/opt/tops/include)
endif()
# link topscc kernel libs
add_custom_target(CUSTOM_TOPSCC_LIBS ALL DEPENDS ${TOPSCC_LIBS})
if(${TOPSCC_KERNEL_SRCS_LEN} GREATER 0)
  add_dependencies(${CUSTOM_GCU_NAME} CUSTOM_TOPSCC_LIBS)
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE ${TOPSCC_LIBS})
endif()

# link paddle shared library
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party.cmake)
add_dependencies(${CUSTOM_GCU_NAME} third_party)
target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE glog gflags)
if(ON_INFER)
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE paddle_inference)
  target_compile_definitions(${CUSTOM_GCU_NAME} PRIVATE PADDLE_ON_INFERENCE)
else()
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE ${PADDLE_CORE_LIB})
  target_link_libraries(${CUSTOM_GCU_NAME} PRIVATE pybind)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION} REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIR})
  string(REPLACE "." "" PY_VERSION_NO_DOT ${PY_VERSION})
endif()

# export paddle_gcu interface with version map
set_property(
  TARGET ${CUSTOM_GCU_NAME}
  PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/paddle_gcu_export.map)
target_link_options(
  ${CUSTOM_GCU_NAME} PRIVATE
  -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/paddle_gcu_export.map)

# testing
if(WITH_TESTING)
  enable_testing()
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../python")
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# packing wheel package
add_custom_command(
  TARGET ${CUSTOM_GCU_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${CUSTOM_GCU_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating custom device directories------>>>")

set(topscc_kernel_lib_targets "")
unset(topscc_kernel_lib_targets)
set(_passes_target_dir
    "${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu")
file(MAKE_DIRECTORY ${_passes_target_dir})
foreach(topscc_kernel_lib ${TOPSCC_LIBS})
  get_filename_component(topscc_kernel_lib_name ${topscc_kernel_lib} NAME_WLE)
  get_filename_component(topscc_kernel_lib_target_name ${topscc_kernel_lib}
                         NAME_WE)
  message(STATUS "topscc_kernel_lib_name: ${topscc_kernel_lib_name}")
  add_custom_command(
    OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/${topscc_kernel_lib_name}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${topscc_kernel_lib}
            ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu
    USES_TERMINAL
    DEPENDS ${topscc_kernel_lib})
  list(
    APPEND
    topscc_kernel_lib_targets
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/${topscc_kernel_lib_name}
  )
endforeach()
add_custom_target(topscc_targets ALL DEPENDS ${topscc_kernel_lib_targets})

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${CMAKE_COMMAND} -E remove -f
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/passes
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/passes
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/passes/*.py
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/passes
  COMMAND
    ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/passes/legacy_ir_passes
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/passes/legacy_ir_passes/*.py
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/gcu/passes/legacy_ir_passes
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${CUSTOM_GCU_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
