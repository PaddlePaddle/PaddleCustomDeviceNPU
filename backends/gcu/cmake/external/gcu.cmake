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

set(GCU_INCLUDE_DIR "/usr/include/gcu")
set(TOPS_INCLUDE_DIR "/usr/include/tops")
set(TOPSTX_INCLUDE_DIR "/usr/include/topstx")
set(GCU_LIB_DIR "/usr/lib")
set(TOPS_LIB_DIR "/opt/tops/lib")

set(TOPS_VERSION "???")

include_directories(${GCU_INCLUDE_DIR} ${TOPS_INCLUDE_DIR}
                    ${TOPSTX_INCLUDE_DIR})

find_library(SDK_LIB NAMES dtu_sdk ${GCU_LIB_DIR})
find_library(ECCL_LIB NAMES eccl ${GCU_LIB_DIR})
find_library(
  TOPS_RT_LIB
  NAMES topsrt ${GCU_LIB_DIR}
  HINTS ${TOPS_LIB_DIR})
find_library(
  TOPSTX_LIB
  NAMES topstx ${GCU_LIB_DIR}
  HINTS ${TOPS_LIB_DIR})
find_library(TOPSATENOP_LIB NAMES topsaten ${GCU_LIB_DIR})
find_library(TOPSCL_LIB NAMES topscl ${GCU_LIB_DIR})

set(GCU_LIBS
    -Wl,--no-as-needed
    ${SDK_LIB}
    ${ECCL_LIB}
    ${TOPS_RT_LIB}
    ${TOPSTX_LIB}
    ${TOPSATENOP_LIB}
    ${TOPSCL_LIB}
    -Wl,--as-needed)

set(VERSION_REGEX "( [0-9]+\.[0-9]+\.(RC)?[0-9]*) ")
macro(find_gcu_version component)
  execute_process(
    COMMAND dpkg -l | grep ${component}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SRC_VERSION_INFO
    RESULT_VARIABLE VERSION_INFO_RESULT
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${SRC_VERSION_INFO})
    string(REGEX MATCH ${VERSION_REGEX} VERSION_CONTENTS "${SRC_VERSION_INFO}")
    string(STRIP "${VERSION_CONTENTS}" VERSION_STR)
    set(COMPONENT_VERSION ${VERSION_STR})
  else()
    set(COMPONENT_VERSION "0.0.0")
  endif()
  message(STATUS "${component} version is ${COMPONENT_VERSION}")
  set(TOPS_VERSION ${COMPONENT_VERSION})
endmacro()

find_gcu_version("tops-sdk")
message(STATUS "TOPS_VERSION is ${TOPS_VERSION}")
