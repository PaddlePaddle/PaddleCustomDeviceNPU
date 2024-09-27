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
void IndexPutKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<const phi::DenseTensor*>& indices,
                    const phi::DenseTensor& value,
                    bool accumulate,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("index_put");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    std::vector<phi::DenseTensor> input_indices;
    for (const auto& index : indices) {
      input_indices.emplace_back(MaybeCreateOrTrans64To32bits(dev_ctx, *index));
    }
    std::vector<topsatenTensor> indices_tensors;
    for (const auto& tensor : input_indices) {
      indices_tensors.emplace_back(CreateTopsatenTensor(tensor));
    }
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_value = MaybeCreateOrTrans64To32bits(dev_ctx, value);
    auto input_tensor = CreateTopsatenTensor(input_x);
    auto value_tensor = CreateTopsatenTensor(input_value);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    auto out_tensor = CreateTopsatenTensor(output);
    std::string abstract_info =
        custom_kernel::GetAbstractInfo("topsatenIndexPut",
                                       output,
                                       input_x,
                                       input_indices,
                                       input_value,
                                       accumulate);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenIndexPut,
                                        dev_ctx,
                                        abstract_info,
                                        out_tensor,
                                        input_tensor,
                                        indices_tensors,
                                        value_tensor,
                                        accumulate);
    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_put,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::IndexPutKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
