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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
namespace {
std::vector<int64_t> GetAxis(const phi::DataLayout& from,
                             const phi::DataLayout& to) {
  if (from == phi::DataLayout::NCHW && to == phi::DataLayout::NHWC) {
    return {0, 2, 3, 1};
  } else if (from == phi::DataLayout::NHWC && to == phi::DataLayout::NCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unsupported layout transform."));
  }
}
}  // namespace

template <typename T, typename Context>
void TransferLayoutKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          int src_layout,
                          int dst_layout,
                          phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("transfer_layout");
  PADDLE_ENFORCE_NE(src_layout,
                    dst_layout,
                    phi::errors::PreconditionNotMet(
                        "No layout transform needed between same layout."));
  VLOG(6) << "TransDataLayout from " << static_cast<phi::DataLayout>(src_layout)
          << " -> " << static_cast<phi::DataLayout>(dst_layout);
  if (x.layout() == static_cast<phi::DataLayout>(dst_layout)) {
    VLOG(6) << "No need to transform, already is " << x.layout();
    custom_kernel::TensorCopy(dev_ctx, x, false, out);
    return;
  }
  if (LaunchAOTKernel()) {
    auto axis = GetAxis(x.layout(), static_cast<phi::DataLayout>(dst_layout));
    *out = custom_kernel::Transpose(dev_ctx, x, axis);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(transfer_layout,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TransferLayoutKernel,
                          int,
                          float,
                          phi::dtype::float16) {}
