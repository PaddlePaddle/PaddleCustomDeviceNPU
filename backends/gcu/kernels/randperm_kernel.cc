// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void RandpermRawKernel(const Context& dev_ctx,
                       int n,
                       phi::DataType dtype,
                       unsigned int seed,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("randperm_raw");
  ContextPinnedGuard<Context> ctx_pinned_guard(dev_ctx);
  VLOG(6) << "[HOST_KERNEL] Impl on host for randperm";
  VLOG(6) << "Enter RandpermRawKernel with n:" << n << ", seed:" << seed
          << ", dtype:" << phi::DataTypeToString(dtype);

  std::shared_ptr<std::mt19937_64> engine;

  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }

  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    T* out_data = dev_ctx.template HostAlloc<T>(out);
    for (int i = 0; i < n; ++i) {
      out_data[i] = static_cast<T>(i);
    }
    std::shuffle(out_data, out_data + n, *engine);
  } else {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(phi::make_ddim({n}));
    T* tmp_data = dev_ctx.template HostAlloc<T>(&tmp_tensor);
    for (int i = 0; i < n; ++i) {
      tmp_data[i] = static_cast<T>(i);
    }
    std::shuffle(tmp_data, tmp_data + n, *engine);
    TensorCopy(dev_ctx, tmp_tensor, false, out);
  }
}

template <typename T, typename Context>
void RandpermKernel(const Context& dev_ctx,
                    int n,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("randperm");
  custom_kernel::RandpermRawKernel<T, Context>(dev_ctx, n, dtype, 0, out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(randperm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RandpermKernel,
                          int64_t,
                          int,
                          float,
                          double) {}
