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

if(NOT EXISTS /usr/bin/topscc)
  message(FATAL_ERROR "TOPSCC not found")
else()
  set(TOPSCC /usr/bin/topscc)
endif()

function(topscc_compile is_static_lib topscc_files compile_options out_libs)
  message(STATUS "== Start compiling topscc kernels ==")
  message(STATUS "[KERNEL FILES]: ${topscc_files}")
  message(STATUS "[COMPILE OPTIONS]: ${compile_options}")
  message(STATUS "[IS STATIC LIB]: ${is_static_lib}")

  set(outs)
  # string(CONCAT compile_options ${compile_options} "-ltops")
  foreach(file ${topscc_files})
    message(STATUS "[TOPSCC FILE]: ${file}")
    get_filename_component(file_name ${file} NAME_WE)
    message(STATUS "[TOPSCC Name]: ${file_name}")

    if(is_static_lib)
      set(out ${CMAKE_CURRENT_BINARY_DIR}/lib${file_name}.a)
    else()
      set(out ${CMAKE_CURRENT_BINARY_DIR}/lib${file_name}.so)
    endif()
    add_custom_command(
      OUTPUT ${out}
      COMMAND ${TOPSCC} ${file} ${compile_options} -o ${out} -ltops
      DEPENDS ${file}
      COMMENT "Compiling ${file} to ${out_file}"
      VERBATIM)
    list(APPEND outs ${out})
  endforeach()
  message(STATUS "== Finish compiling topscc kernels ==")
  set(${out_libs}
      ${outs}
      PARENT_SCOPE)
endfunction()
