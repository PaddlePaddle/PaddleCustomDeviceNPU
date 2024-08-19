/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void RandintKernel(const Context& dev_ctx,
                   const int low,
                   const int high,
                   const phi::IntArray& shape,
                   phi::DataType dtype UNUSED,
                   phi::DenseTensor* out) {
  out->Resize(common::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
  int64_t low_ = low;
  int64_t high_ = high;
  int64_t seed = 0;
  int64_t offset = 0;
  EXEC_NPU_CMD(aclnnInplaceRandom, dev_ctx, *out, low_, high_, seed, offset);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    randint, npu, ALL_LAYOUT, custom_kernel::RandintKernel, int, int64_t) {}
