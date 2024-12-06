# PaddlePaddle Custom Device Implementation for Ascend NPU

English | [简体中文](./README_cn.md)

Please refer to the following steps to compile, install and verify the custom device implementation for Ascend NPU.

## Ascend NPU System Requirements

| Type | Version     |
| --------- | -------- |
| Chip | Ascend 910B |
| CANN | [CANN 8.0.RC1](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software) |
| Driver | [23.0.3](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software) |

**Note**：please refer to [release/2.6](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/release/2.6/backends/npu/README_cn.md) for Ascend 910A support.

## Prepare environment and source code

```bash
# 1. pull PaddlePaddle Ascend NPU development docker image
# dockerfile of the image is in tools/dockerfile directory
# Ascend 910B - check with the output of 'lspci | grep d802'
registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py310
registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py310

# 2. refer to the following commands to start docker container
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-$(uname -m)-gcc84-py310 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice
```

## PaddlePaddle Installation and Verification

### Source Code Compile

```bash
# 1. go to ascend npu directory
cd backends/npu

# 2. please ensure the PaddlePaddle cpu whl package is already installed
# the development docker image NOT have PaddlePaddle cpu whl installed by default
# you may download and install the nightly built cpu whl package with links below
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/


# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_npu*.whl
```

## Verification

```bash
# 1. list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# output as following
['npu']

# 2. check installed custom npu version
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# output as following
version: 0.0.0
commit: 9bfc65a7f11072699d0c5af160cf7597720531ea
cann: 8.0.RC1

# 3. health check
python -c "import paddle; paddle.utils.run_check()"
# output as following
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## Train and Inference Demo

```bash
# demo for training, evaluation and inference
python tests/test_LeNet_MNIST.py

# training output as following
Epoch [1/2], Iter [01/14], reader_cost: 0.89279 s, batch_cost: 42.20599 s, ips: 97.04784 samples/s, eta: 0:19:41
Epoch [1/2], Iter [02/14], reader_cost: 0.44657 s, batch_cost: 21.10753 s, ips: 194.05393 samples/s, eta: 0:09:29
... ...
Epoch [2/2], Iter [14/14], reader_cost: 0.07168 s, batch_cost: 0.08018 s, ips: 51086.10163 samples/s, eta: 0:00:00
Epoch ID: 2, Epoch time: 1.36502 s, reader_cost: 1.00354 s, batch_cost: 1.12250 s, avg ips: 42009.72047 samples/s
Eval - Epoch ID: 2, Top1 accurary:: 0.81091, Top5 accurary:: 0.99036

# inference output as following
I0103 19:11:34.570551 43520 program_interpreter.cc:214] New Executor is Running.
I0103 19:11:34.589533 43520 analysis_predictor.cc:1684] CustomDevice is enabled
... ...
I0103 19:11:34.590348 43520 ir_params_sync_among_devices_pass.cc:144] Sync params from CPU to npu:0
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0103 19:11:34.592526 43520 analysis_predictor.cc:1867] ======= optimize end =======
I0103 19:11:34.592568 43520 naive_executor.cc:200] ---  skip [feed], feed -> inputs
I0103 19:11:34.592684 43520 naive_executor.cc:200] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
Output data size is 10
Output data shape is (1, 10)
```

## Environment Variables


| Subject     | Variable Name       | Type   | Description    | Default Value |
| -------- | -------------------------------- | ------ | --------------------------------- | ------------------------------------------------------------ |
| Debug     | CUSTOM_DEVICE_BLACK_LIST| String | Ops in back list will fallbacks to CPU  |  ""  |
| Debug     | FLAGS_npu_check_nan_inf | Bool   | check nan or inf of all npu kernels | False                                                       |
| Debug     | FLAGS_npu_blocking_run | Bool   | enable sync for all npu kernels | False                                                     |
| Profiling | FLAGS_npu_profiling_dir | String |   ACL profiling output dir     | "ascend_profiling"                                           |
| Profiling | FLAGS_npu_profiling_dtypes | Uint64 | ACL datatypes to profile | Refer to [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L31) |
| Profiling | FLAGS_npu_profiling_metrics | Uint64 | AI Core metric to profile  | Refer to [runtime.cc](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/runtime/runtime.cc#L36) |
| Performance | FLAGS_npu_storage_format         | Bool   | enable Conv/BN private ACL format | False                                                        |
| OP Compile | FLAGS_npu_jit_compile  | Bool   | enable NPU OP JIT compile  | True |
