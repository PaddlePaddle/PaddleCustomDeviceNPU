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
void SquaredL2NormKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("squared_l2_norm");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    auto square = TensorEmpty(dev_ctx, x.meta());
    LAUNCH_TOPSATENOP(topsatenSquare, dev_ctx, square, x);
    std::vector<int64_t> reduce_axis(x.dims().size(), 0);
    std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    bool keep_dim = false;
    LAUNCH_TOPSATENOP(topsatenSum,
                      dev_ctx,
                      *out,
                      square,
                      reduce_axis,
                      keep_dim,
                      out->dtype());
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "squared_l2_norm",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squared_l2_norm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SquaredL2NormKernel,
                          float,
                          phi::dtype::float16) {}
