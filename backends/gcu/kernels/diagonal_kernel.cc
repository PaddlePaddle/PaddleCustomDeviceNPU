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
void DiagonalKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("diagonal");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    phi::DenseTensor out_tmp;
    out_tmp.set_meta(out->meta());

    auto out_tmp_tensor = CreateTopsatenTensorWithoutInitialized(out_tmp);
    auto x_tensor = CreateTopsatenTensor(x);

    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "Diagonal_topsatenDiagonal", out_tmp, x, offset, axis1, axis2);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenDiagonal,
                                        dev_ctx,
                                        abstract_info,
                                        out_tmp_tensor,
                                        x_tensor,
                                        offset,
                                        axis1,
                                        axis2);

    auto out_tensor = CreateTopsatenTensor(*out);
    abstract_info = custom_kernel::GetAbstractInfo(
        "Diagonal_topsatenCopy", out_tensor, false);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenCopy,
                                        dev_ctx,
                                        abstract_info,
                                        out_tensor,
                                        out_tmp_tensor,
                                        false);
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(diagonal,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DiagonalKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}
