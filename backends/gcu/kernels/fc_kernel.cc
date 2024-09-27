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
namespace {
void AdjustStrides(phi::DenseTensor& tensor) {  // NOLINT
  size_t rank = tensor.dims().size();
  if (rank <= 1) {
    return;
  }
  auto meta = tensor.meta();
  meta.strides = meta.calc_strides(meta.dims);
  std::swap(meta.dims[rank - 1], meta.dims[rank - 2]);
  std::swap(meta.strides[rank - 1], meta.strides[rank - 2]);
  tensor.set_meta(meta);
}
}  // namespace

template <typename T, typename Context>
void FCKernel(const Context& dev_ctx,
              const phi::DenseTensor& input,
              const phi::DenseTensor& w,
              const paddle::optional<phi::DenseTensor>& bias,
              const int in_num_col_dims,
              const std::string& activation_type,
              const bool padding_weights,
              phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("fc");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);

    const phi::DenseTensor x_matrix =
        input.dims().size() > 2 ? phi::ReshapeToMatrix(input, in_num_col_dims)
                                : input;

    auto out_dim = out->dims();
    if (out_dim.size() != 2) {
      out->Resize({x_matrix.dims()[0], w.dims().at(1)});
    }

    auto w_trans = w;
    AdjustStrides(w_trans);

    phi::DenseTensor fc_bias;
    if (bias) {
      fc_bias = bias.get();
    } else {
      auto meta =
          phi::DenseTensorMeta(input.dtype(), phi::make_ddim({w.dims().at(0)}));
      fc_bias = TensorZeros(dev_ctx, meta);
    }
    LAUNCH_TOPSATENOP(
        topsatenLinear, dev_ctx, *out, x_matrix, w_trans, fc_bias);
    if (out_dim.size() != 2) {
      out->Resize(out_dim);
    }

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    fc, gcu, ALL_LAYOUT, custom_kernel::FCKernel, float, phi::dtype::float16) {}
