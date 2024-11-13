# PaddlePaddle Custom Device Implementation for Enflame GCU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the hardware backend (GCU).

## Prepare environment and source code

```bash
# 1) Pull the image. Note that this image is only for development environment
#    and does not contain precompiled PaddlePaddle installation package.
#    The build script and dockerfile of this image are located in the tools/dockerfile directory.
#    Note: This docker is in the release process now. (20241113)
registry.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84

# 2) Refer to the following command to start the container.
docker run --name paddle-gcu-dev -v /home:/home \
    --network=host --ipc=host -it --privileged \
    registry.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84 /bin/bash

# 3) Clone the source code.
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice

# 4) Prepare the machine and initialize the environment (only required for device used for execution).
# 4a) Get the driver: The full software package is placed in docker in advance
#     and needs to be copied to the directory outside docker, such as: /home/workspace/deps/.
mkdir -p /home/workspace/deps/ && cp /root/TopsRider_i3x_*/TopsRider_i3x_*_deb_amd64.run /home/workspace/deps/

# 4b) Verify whether the machine has Enflame S60 accelerators.
#     Check whether the following command has output in the system environment.
#     Note: You need to press Ctrl+D to exit docker.
#     The following initialization environment operations are all performed in the system environment.
lspci | grep S60

# 4c) Install the driver.
cd /home/workspace/deps/
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load

# 4d) After the driver is installed, refer to the following command to re-enter docker.
docker start paddle-gcu-dev
docker exec -it paddle-gcu-dev bash
```

## PaddleCustomDevice Installation and Verification

### Source code compilation and installation

```bash
# 1) Enter the hardware backend (Enflame GCU) directory.
cd PaddleCustomDevice/backends/gcu

# 2) Before compiling, you need to ensure that the PaddlePaddle installation package is installed in the environment.
#    Just install the PaddlePaddle CPU version directly.
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 3) Start compiling, and submodules will be downloaded on demand during compilation.
mkdir -p build && cd build
export PADDLE_CUSTOM_PATH=`python -c "import re, paddle; print(re.compile('/__init__.py.*').sub('',paddle.__file__))"`
cmake .. -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPY_VERSION=3.10
make -j $(nproc)

# 4) The compiled product is in the build/dist path and installed using pip.
python -m pip install --force-reinstall -U build/dist/paddle_custom_gcu*.whl
```

### Functional Verification

```bash
# 1) List available hardware backends.
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# Expect the following output.
['gcu']

# 2) Check currently installed version.
python -c "import paddle_custom_device; paddle_custom_device.gcu.version()"
# Expect to get output like this.
version: 3.0.0b1+3.1.0.20241113
commit: f05823682bf607deb1b4adf9a9309f81225958fe
TopsPlatform: 1.2.0.301

# 3) Unit test, compiled with -DWITH_TESTING=ON and executed in the build directory.
ctest
```
