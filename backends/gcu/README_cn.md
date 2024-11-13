# 飞桨自定义接入硬件后端(GCU)

简体中文 | [English](./README.md)

请参考以下步骤进行硬件后端(GCU)的编译安装与验证

## 环境准备与源码同步

```bash
# 1) 拉取镜像，注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
#    此镜像的构建脚本与dockerfile位于tools/dockerfile目录下
#    注意： 此docker正在发布流程中(20241113)
registry.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84

# 2) 参考如下命令启动容器
docker run --name paddle-gcu-dev -v /home:/home \
    --network=host --ipc=host -it --privileged \
    registry.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84 /bin/bash

# 3) 克隆PaddleCustomDevice源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 4) 机器准备，初始化环境(仅用于执行的设备需要)
# 4a) 驱动获取：docker内提前放置了全量软件包，需拷贝至docker外目录，如：/home/workspace/deps/
mkdir -p /home/workspace/deps/ && cp /root/TopsRider_i3x_*/TopsRider_i3x_*_deb_amd64.run /home/workspace/deps/

# 4b) 验证机器是否插有燧原S60加速卡，系统环境下查看如下命令是否有输出
#     注：需Ctrl+D退出docker， 以下初始化环境相关操作均在系统环境下执行
lspci | grep S60

# 4c) 安装驱动
cd /home/workspace/deps/
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load

# 4d) 驱动安装完成后重新进入docker，参考如下命令
docker start paddle-gcu-dev
docker exec -it paddle-gcu-dev bash
```

## PaddleCustomDevice安装与运行

### 编译安装

```bash
# 1) 进入硬件后端(燧原GCU)目录
cd PaddleCustomDevice/backends/gcu

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
version: 3.0.0b1+3.1.0.20241113
commit: f05823682bf607deb1b4adf9a9309f81225958fe
TopsPlatform: 1.2.0.301

# 3) 单元测试，带上-DWITH_TESTING=ON编译后在build目录下执行
ctest
```
