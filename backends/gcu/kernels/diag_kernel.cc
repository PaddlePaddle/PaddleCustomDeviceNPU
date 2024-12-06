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
template <typename T, typename Context>
void DiagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int offset,
                float padding_value,
                DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("diag");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    auto rank = x.dims().size();
    PADDLE_ENFORCE_LE(
        rank, 2, phi::errors::InvalidArgument("x must be a 1D or 2D tensor"));
    if (rank == 1) {
      if (abs(padding_value) < 1e-6) {
        LAUNCH_TOPSATENOP(topsatenDiag, dev_ctx, *out, x, offset);
      } else {
        LAUNCH_TOPSATENOP(topsatenDiag, dev_ctx, *out, x, offset);

        phi::DenseTensor mask_tmp = custom_kernel::TensorEmpty(
            dev_ctx, {phi::DataType::BOOL, out->dims()});

        phi::DenseTensor cpu_tensor;
        phi::DenseTensorMeta cpu_meta = {phi::DataType::BOOL, out->dims()};
        cpu_tensor.set_meta(cpu_meta);
        bool* host_mask = dev_ctx.template HostAlloc<bool>(&cpu_tensor);
        for (size_t i = 0; i < mask_tmp.numel(); i++) {
          host_mask[i] = true;
        }
        int64_t stride_w = 1;
        int64_t stride_h = mask_tmp.dims()[0];
        bool* start =
            host_mask + (offset >= 0 ? offset * stride_w : -offset * stride_h);
        for (size_t i = 0; i < mask_tmp.numel(); i = i + stride_h + stride_w) {
          *(start + i) = false;
        }

        // copy mask to device
        TensorCopy(dev_ctx, cpu_tensor, true, &mask_tmp);

        // call mask_fill
        LAUNCH_TOPSATENOP(topsatenMasked_fill,
                          dev_ctx,
                          *out,
                          *out,
                          mask_tmp,
                          phi::Scalar(padding_value));
      }
    } else {
      LAUNCH_TOPSATENOP(topsatenDiag, dev_ctx, *out, x, offset);
    }
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(diag,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DiagKernel,
                          phi::dtype::float16,
                          int,
                          float,
                          double,
                          int64_t) {}
