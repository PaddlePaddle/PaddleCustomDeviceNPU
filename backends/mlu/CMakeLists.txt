cmake_minimum_required(VERSION 3.10)

project(paddle-custom-mlu CXX C)
set(CUSTOM_MLU_NAME "paddle-custom-mlu")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include(paddle)
include(version)
include(generic)
include(external/neuware)
include_directories(${CMAKE_SOURCE_DIR})

option(WITH_KERNELS "compile with custom kernels" ON)
option(WITH_TESTING "compile with unit testing" OFF)
option(WITH_MKL "compile with mkl support" ON)
option(WITH_ARM "compile with arm support" OFF)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_PROFILE "compile with profile" ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

# custom runtime
set(CUSTOM_MLU_SRCS runtime/runtime.cc runtime/CNRTEvent.h)
add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)
# TODO(qiil93): avoid compile error, to be removed
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)
if(ON_INFER)
  add_definitions(-DON_INFER)
endif()
if(WITH_PROFILE)
  add_definitions(-DWITH_PROFILE)
endif()

# custom kernels
if(WITH_KERNELS)
  file(
    GLOB_RECURSE CUSTOM_KERNEL_SRCS
    RELATIVE ${CMAKE_SOURCE_DIR}
    kernels/*.cc)
  list(APPEND CUSTOM_MLU_SRCS ${CUSTOM_KERNEL_SRCS})

  file(
    GLOB_RECURSE CUSTOM_PLUGIN_OP_SRCS
    RELATIVE ${CMAKE_SOURCE_DIR}
    plugin_ops/*.cc)
  list(APPEND CUSTOM_MLU_SRCS ${CUSTOM_PLUGIN_OP_SRCS})
endif()

if(WITH_PROFILE)
  list(APPEND CUSTOM_MLU_SRCS runtime/process_cnpapi_data.h
       runtime/process_cnpapi_data.cc)
endif()

# C++ infer lib need to compatiable with CXX11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
if(ON_INFER)
  link_directories($ENV{PADDLE_INFERENCE_LIB_DIR})
endif()

# build shared library
add_library(${CUSTOM_MLU_NAME} SHARED ${CUSTOM_MLU_SRCS})
target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE ${NEUWARE_LIBS} glog gflags)

# link paddle shared library
include(third_party)
add_dependencies(${CUSTOM_MLU_NAME} third_party)
# concurrentqueue
include(external/concurrentqueue)
add_dependencies(${CUSTOM_MLU_NAME} extern_concurrentqueue)
if(ON_INFER)
  target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE paddle_inference)
  target_compile_definitions(${CUSTOM_MLU_NAME} PRIVATE PADDLE_ON_INFERENCE)
else()
  target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE ${PADDLE_CORE_LIB})
  target_link_libraries(${CUSTOM_MLU_NAME} PRIVATE pybind)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION} REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIR})
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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# packing wheel package
add_custom_command(
  TARGET ${CUSTOM_MLU_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
  COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/lib${CUSTOM_MLU_NAME}.so
    ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
  COMMENT "Creating custom device directories------>>>")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
  DEPENDS ${CUSTOM_MLU_NAME}
  COMMENT "Packing whl packages------>>>")

add_custom_target(python_package ALL
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
