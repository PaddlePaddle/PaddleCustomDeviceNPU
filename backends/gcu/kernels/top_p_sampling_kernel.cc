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

#include "custom_op/custom_op_common.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/extension.h"

namespace custom_kernel {
template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& ps,
                        const paddle::optional<phi::DenseTensor>& threshold,
                        int random_seed,
                        phi::DenseTensor* out,
                        phi::DenseTensor* ids) {
  PADDLE_GCU_KERNEL_TRACE("top_p_sampling");
  auto probs = custom_op_common::CreateTensorFromDenseTensor(x);
  auto top_p = custom_op_common::CreateTensorFromDenseTensor(ps);
  auto sample_out = custom_op_common::TopPSampling(probs, top_p);
  *out = custom_op_common::CreateDenseTensorFromTernsor(sample_out[0]);
  *ids = custom_op_common::CreateDenseTensorFromTernsor(sample_out[1]);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(top_p_sampling,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TopPSamplingKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
