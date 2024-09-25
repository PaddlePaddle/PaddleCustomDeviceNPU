// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

template <typename T, typename Context>
void AssignKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, x, false, out);
}

template <typename T, typename Context>
void AssignRawKernel(const Context& dev_ctx,
                     const paddle::optional<phi::DenseTensor>& x,
                     phi::DenseTensor* out) {
  if (x) {
    if (!x->initialized()) {
      return;
    }
    auto& x_tensor = *x.get_ptr();
    custom_kernel::AssignKernel<T, Context>(dev_ctx, x_tensor, out);
  }
}

template <typename T, typename Context>
void AssignArrayKernel(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& x,
                       std::vector<phi::DenseTensor*> out) {
  for (size_t i = 0; i < x.size(); ++i) {
    custom_kernel::AssignKernel<T, Context>(dev_ctx, *x[i], out.at(i));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(assign,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::AssignKernel,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          bool) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(assign_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::AssignRawKernel,
                          int,
                          phi::dtype::float16,
                          float,
                          double,
                          bool) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(assign_array,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::AssignArrayKernel,
                          int,
                          float,
                          double,
                          bool) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
