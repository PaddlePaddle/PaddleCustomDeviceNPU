#!/bin/bash

WORKSPACE=`pwd`

python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.version.commit)"

export CCACHE_MAXSIZE=80G
export CCACHE_LIMIT_MULTIPLE=0.8
export CCACHE_SLOPPINESS=clang_index_store,time_macros,include_file_mtime

cd backends/gcu
mkdir -p build && cd build
export PADDLE_CUSTOM_PATH=`python -c "import re, paddle; print(re.compile('__init__.py.*').sub('',paddle.__file__))"`
cmake .. -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPY_VERSION=3.10
make -j $(nproc)

python -m pip install --force-reinstall -U dist/paddle_custom_gcu*.whl

ctest
