#!/usr/bin/env python3

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

"""
Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved.

Build and setup Intele_HPU custom ops.
"""

from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name="paddlenlp_ops",
    ext_modules=[
        CppExtension(
            sources=[
                "./src/index_copy.cc",
                "./src/fake_gpu_kernels.cc",
                "./llama_infer/fused_rms_qkv_rope.cc",
                "./llama_infer/fused_sdpa_proj.cc",
                "./llama_infer/fused_mlp.cc",
                "./llama_infer/fused_rms_mlp.cc",
                "./llama_infer/ref_pp_kernels.cc",
            ],
            include_dirs=[
                "../",
                "../build/third_party/install/onednn/include/",
                "../build/third_party/install/glog/include/",
                "../build/third_party/install/gflags/include/",
            ],
            library_dirs=[
                "../build/python/paddle_custom_device/",
                "/usr/lib/habanalabs/",
            ],
            libraries=[
                "paddle-intel-hpu",
                "Synapse",
            ],
        )
    ],
)
