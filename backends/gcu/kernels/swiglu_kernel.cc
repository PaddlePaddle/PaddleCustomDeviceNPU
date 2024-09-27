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
void SwiGLUKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const paddle::optional<phi::DenseTensor>& y,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("swiglu");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    auto input_x = x;
    if (y) {
      PADDLE_ENFORCE_EQ(y.get().dims(),
                        x.dims(),
                        phi::errors::InvalidArgument(
                            "The shape of Input(Y):[%s] must be equal "
                            "to the shape of Input(X):[%s].",
                            y.get().dims(),
                            x.dims()));
      auto meta = x.meta();
      auto rank = meta.dims.size();
      meta.dims[rank - 1] *= 2;
      meta.strides = meta.calc_strides(meta.dims);
      phi::DenseTensor concat_output = TensorEmpty(dev_ctx, meta);
      std::vector<topsatenTensor> in_tensors = {CreateTopsatenTensor(x),
                                                CreateTopsatenTensor(y.get())};
      auto out_tensor = CreateTopsatenTensor(concat_output);
      int64_t axis = rank - 1;
      std::vector<phi::DenseTensor> concat_ins = {x, y.get()};
      std::string abstract_info = custom_kernel::GetAbstractInfo(
          "topsatenCat", concat_output, concat_ins, axis);
      LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(
          topsatenCat, dev_ctx, abstract_info, out_tensor, in_tensors, axis);
      input_x = concat_output;
    }
    LAUNCH_TOPSATENOP(topsvllmSiluAndMul, dev_ctx, *out, input_x);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(swiglu,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SwiGLUKernel,
                          float,
                          phi::dtype::float16) {}
