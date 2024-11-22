# PaddleCustomDevice

简体中文 | [English](./README_en.md) | [日本語](./README_ja.md)

[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/hardware_support/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/PaddleCustomDevice/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)

『飞桨』自定义硬件接入实现。

## 使用指南

方案设计参考[Custom Device 接入方案介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_overview_cn.html)，开发指南请参考[新硬件接入示例](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/custom_device_docs/custom_device_example_cn.html)且示例代码位于 [CustomCPU](backends/custom_cpu/README_cn.md)。

## 硬件后端

飞桨自定义硬件接入支持如下硬件后端：

- [飞桨自定义接入硬件后端(昇腾NPU)](backends/npu/README_cn.md)
- [飞桨自定义接入硬件后端(寒武纪MLU)](backends/mlu/README_cn.md)
- [飞桨自定义接入硬件后端(英特尔GPU)](backends/intel_gpu/README.md)
- [飞桨自定义接入硬件后端(苹果MPS)](backends/mps/README.md)
- [飞桨自定义接入硬件后端(壁仞GPU)](backends/biren_gpu/README_cn.md)
- [飞桨自定义接入硬件后端(燧原GCU)](backends/gcu/README_cn.md)
- [飞桨自定义接入硬件后端(太初SDAA)](backends/sdaa/README_cn.md)

## 版权和许可证

PaddleCustomDevice由[Apache-2.0 license](LICENSE)提供。
