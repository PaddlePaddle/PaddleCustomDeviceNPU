// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  std::vector<int32_t> axes(axes_int_array.GetData().begin(),
                            axes_int_array.GetData().end());

  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);

  TensorCopy(dev_ctx, x, false, out);

  out->Resize(out_dims);
}

template <typename T, typename Context>
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::IntArray& axes_int_array,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  custom_kernel::SqueezeKernel<T, Context>(dev_ctx, x, axes_int_array, out);
}

template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& dout,
                       const phi::IntArray& axes_int_array,
                       phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();

  auto x_dims = dx->dims();

  TensorCopy(dev_ctx, dout, false, dx);
  dx->Resize(x_dims);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squeeze,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_with_xshape,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeWithXShapeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
