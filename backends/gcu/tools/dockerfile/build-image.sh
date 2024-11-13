#!/bin/bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Usage:
# bash build-image.sh ${TOPS_RIDER_VERSION} ${TOPS_RIDER_PACKAGE_LINK}

TOPS_RIDER_VERSION=${1:-3.2.109} # default 3.2.109

# Please contact Enflame (Email: developer-enflame@enflame-tech.com) to
# obtain software driver packages and other assistance.
TOPS_RIDER_PACKAGE_LINK=${2}

if [ $(uname -i) == 'x86_64' ]; then
  docker pull registry.baidubce.com/device/paddle-cpu:ubuntu20-x86_64-gcc84-py310
  docker build --network=host -f Dockerfile.gcu.ubuntu20.gcc84 \
       --build-arg TOPS_RIDER_VERSION=${TOPS_RIDER_VERSION} \
       --build-arg TOPS_RIDER_PACKAGE_LINK=${TOPS_RIDER_PACKAGE_LINK} \
       --build-arg http_proxy=${proxy} \
       --build-arg https_proxy=${proxy} \
       --build-arg no_proxy=bcebos.com \
       -t registry.baidubce.com/device/paddle-gcu:topsrider${TOPS_RIDER_VERSION}-ubuntu20-x86_64-gcc84 .
#   docker push registry.baidubce.com/device/paddle-gcu:topsrider${TOPS_RIDER_VERSION}-ubuntu20-x86_64-gcc84
else
  echo "Os $(uname -i) is not supported."
fi
