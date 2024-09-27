// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void NMSKernel(const Context& dev_ctx,
               const DenseTensor& boxes,
               float threshold,
               DenseTensor* output) {
  PADDLE_GCU_KERNEL_TRACE("nms");
  // firstly, we need to set output shape to max size and alloc max size memory
  output->Resize({boxes.dims()[0]});
  auto out_data = dev_ctx.template Alloc<int64_t>(output);

  auto stream = static_cast<void*>(dev_ctx.stream());
  auto topscl_out = CreateTopsclTensor(*output);
  auto topscl_boxes = CreateTopsclTensor(boxes);

  LAUNCH_TOPSCLOP_WITH_RAW_TOPSCL_DEF(
      nms, dev_ctx, "nms", &topscl_out, topscl_boxes, threshold);
  // refresh the output tensor shape
  output->Resize(common::make_ddim(topscl_out.shape().dims()));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    nms, gcu, ALL_LAYOUT, custom_kernel::NMSKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
