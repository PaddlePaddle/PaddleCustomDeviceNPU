# PaddlePaddle Custom Device Implementaion for Cambricon MLU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementaion for Cambricon MLU.

## Neuware Version

| Module    | Version  |
| --------- | -------- |
| cntoolkit | 3.10.2-1  |
| cnnl      | 1.25.1-1 |
| cnnlextra | 1.8.1-1  |
| cncl      | 1.16.0-1 |
| mluops    | 1.1.1-1 |

## Prepare environment and source code

```bash
# 1. pull PaddlePaddle Cambricon MLU development docker image
#    dockerfile of the image is in tools/dockerfile directory
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-x86_64-gcc84-py310
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-kylinv10-aarch64-gcc82-py310

# 2. refer to the following commands to start docker container
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-x86_64-gcc84-py310 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

### Install Wheel Pacakge

Install nighlty built PaddlePaddle wheel packages as following:

```bash
# Wheel packages for X86_64
https://paddle-device.bj.bcebos.com/0.0.0/mlu/paddlepaddle-0.0.0-cp310-cp310-linux_x86_64.whl
https://paddle-device.bj.bcebos.com/0.0.0/mlu/paddle_custom_mlu-0.0.0-cp310-cp310-linux_x86_64.whl

# Wheel packages for Aarch64
https://paddle-device.bj.bcebos.com/0.0.0/mlu/paddlepaddle-0.0.0-cp310-cp310-linux_aarch64.whl
https://paddle-device.bj.bcebos.com/0.0.0/mlu/paddle_custom_mlu-0.0.0-cp310-cp310-linux_aarch64.whl

# Install two wheel packages after download
pip install paddlepaddle*.whl paddle_custom_mlu*.whl
```

### Source Code Compilation

```bash
# 1. navigate to implementaion for Cambricon MLU
cd backends/mlu

# 2. before compiling, ensure that PaddlePaddle (CPU version) is installed, you can run the following command
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_mlu*.whl
```

## Verification

```bash
# 1. list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# output as following
['mlu']

# 2. check installed custom mlu version
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# output as following
version: 0.0.0
commit: 83dfe3de33f0a915fb189161568fc3804b5f9c1b
cntoolkit: 3.10.2
cnnl: 1.25.1
cnnlextra: 1.8.1
cncl: 1.16.0
mluops: 1.1.1

# 3. health check
python -c "import paddle; paddle.utils.run_check()"
# output as following
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 16 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## Train and Inference Demo

```bash
# demo for training, evaluation and inference
python tests/test_LeNet_MNIST.py

# training output as following
Epoch [1/2], Iter [01/14], reader_cost: 2.73611 s, batch_cost: 2.78069 s, ips: 1473.01483 samples/s, eta: 0:01:17
Epoch [1/2], Iter [02/14], reader_cost: 1.37505 s, batch_cost: 1.41733 s, ips: 2889.94454 samples/s, eta: 0:00:38
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.19809 s, batch_cost: 0.23765 s, ips: 17235.35966 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 3.46918 s, reader_cost: 2.77321 s, batch_cost: 3.32711 s, avg ips: 16529.56425 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.86230, Top5 accurary:: 0.98950

# inference output as following
I0521 20:27:26.487897  2030 program_interpreter.cc:221] New Executor is Running.
I0521 20:27:26.499172  2030 analysis_predictor.cc:1850] CustomDevice is enabled
... ...
I0521 20:27:26.500521  2030 ir_params_sync_among_devices_pass.cc:142] Sync params from CPU to mlu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [save_optimized_model_pass]
--- Running analysis [ir_graph_to_program_pass]
I0521 20:27:26.504653  2030 analysis_predictor.cc:2032] ======= ir optimization completed =======
I0521 20:27:26.504727  2030 naive_executor.cc:200] ---  skip [feed], feed -> inputs
I0521 20:27:26.504953  2030 naive_executor.cc:200] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## Environment variables

| Name | Type | Desc | Default |
| ---- | ---- | ---- | ------- |
| PADDLE_MLU_ALLOW_TF32 | Bool | Whether to enable tf32 computation | True |
| CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE | Int | Whether to enlarge mem-pool of CNCL | 1 |
| CUSTOM_DEVICE_BLACK_LIST | String | op blacklist, force the operation to run in CPU mode | "" |
| FLAGS_allocator_strategy | ENUM | [paddlepaddle-doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/flags/memory_cn.html#flags-allocator-strategy) | auto_growth |
