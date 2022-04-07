# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(ExternalProject)

# update eigen to the commit id f612df27 on 03/16/2021
set(EIGEN_PREFIX_DIR ${THIRD_PARTY_PATH}/eigen3)
set(EIGEN_SOURCE_DIR ${THIRD_PARTY_PATH}/eigen3/src/extern_eigen3)
set(EIGEN_REPOSITORY https://gitlab.com/libeigen/eigen.git)
set(EIGEN_TAG        f612df273689a19d25b45ca4f8269463207c4fee)

set(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

ExternalProject_Add(
    extern_eigen3
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    GIT_REPOSITORY    ${EIGEN_REPOSITORY}
    GIT_TAG           ${EIGEN_TAG}
    PREFIX            ${EIGEN_PREFIX_DIR}
    UPDATE_COMMAND    ""
    PATCH_COMMAND     ${EIGEN_PATCH_COMMAND}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

add_library(eigen3 INTERFACE)

add_dependencies(eigen3 extern_eigen3)
