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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& ps,
                        const paddle::optional<phi::DenseTensor>& threshold,
                        const paddle::optional<phi::DenseTensor>& topp_seed,
                        int seed,
                        int k,
                        const std::string& mode,
                        phi::DenseTensor* out,
                        phi::DenseTensor* ids,
                        phi::DenseTensor* topk_scores,
                        phi::DenseTensor* topk_ids) {
  PADDLE_THROW(phi::errors::Unimplemented("Only supports model export"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(top_p_sampling,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TopPSamplingKernel,
                          float,
                          phi::float16) {}
