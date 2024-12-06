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

WORKSPACE=`pwd`

python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.version.commit)"

export CCACHE_MAXSIZE=80G
export CCACHE_LIMIT_MULTIPLE=0.8
export CCACHE_SLOPPINESS=clang_index_store,time_macros,include_file_mtime
export PADDLE_CUSTOM_PATH=`python -c "import re, paddle; print(re.compile('__init__.py.*').sub('',paddle.__file__))"`

cd backends/gcu
mkdir -p build && cd build
cmake .. -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPY_VERSION=3.10
make -j $(nproc)

python -m pip install --force-reinstall -U dist/paddle_custom_gcu*.whl

ctest -j4
