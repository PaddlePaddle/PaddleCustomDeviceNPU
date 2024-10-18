# 飞桨自定义接入硬件后端(GCU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(GCU)的编译安装与验证

## 环境准备与源码同步

```bash
# 1) 获取PaddlePaddle Docker镜像，并安装燧原GCU软件栈

# 2) 克隆PaddleCustomDevice源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddleCustomDevice安装与运行

### 编译安装

```bash
# 1) 进入硬件后端(燧原GCU)目录
cd backends/gcu

# 2) 编译之前需确保环境下装有飞桨安装包，直接安装飞桨CPU版本即可
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 3) 编译，编译时会按需下载submodule
mkdir -p build && cd build
export PADDLE_CUSTOM_PATH=`python -c "import re, paddle; print(re.compile('/__init__.py.*').sub('',paddle.__file__))"`
cmake .. -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPY_VERSION=3.10
make -j $(nproc)

# 4) 编译产出在build/dist路径下，使用pip安装
python -m pip install --force-reinstall -U build/dist/paddle_custom_gcu*.whl
```

### 功能验证

```bash
# 1) 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 预期得到如下输出结果
['gcu']

# 2) 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.gcu.version()"
# 预期得到类似以下的输出结果
version: 0.0.0.9e03b0a
commit: 9e03b0a42a530d07fb60e141ee618fc02595bd96
tops-sdk: 2.5.20231128

# 3) 单元测试，带上-DWITH_TESTING=ON编译后在build目录下执行
ctest
```
