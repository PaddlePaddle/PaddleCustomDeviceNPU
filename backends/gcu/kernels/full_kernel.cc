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
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("full");
  if (LaunchAOTKernel()) {
    auto shape_vec = shape.GetData();
    if (shape_vec.empty()) {
      shape_vec = {1};
    }
    auto out_dim = phi::make_ddim(shape_vec);
    out->ResizeAndAllocate(out_dim);
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor output(*out);
    if (out->dtype() == phi::DataType::BOOL ||
        out->dtype() == phi::DataType::INT32 ||
        out->dtype() == phi::DataType::INT64 ||
        out->dtype() == phi::DataType::FLOAT64) {
      auto meta = out->meta();
      meta.dtype = phi::DataType::FLOAT32;
      output.set_meta(meta);
      dev_ctx.template Alloc<float>(&output);
    }
    // topsatenFull not support bool or int32 yet.
    LAUNCH_TOPSATENOP(topsatenFull, dev_ctx, output, shape_vec, val);
    if (out->dtype() == phi::DataType::BOOL ||
        out->dtype() == phi::DataType::INT32 ||
        out->dtype() == phi::DataType::INT64 ||
        out->dtype() == phi::DataType::FLOAT64) {
      custom_kernel::Cast(dev_ctx, output, out->dtype(), out);
    }

  } else {  // kernel impl base on JIT
    auto shape_vec = shape.GetData();
    auto out_dim = phi::make_ddim(shape_vec);
    out->ResizeAndAllocate(out_dim);
    dev_ctx.template Alloc<T>(out);
    FillGcuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(val.to<T>()));
    out->Resize(out_dim);
  }
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("full_like");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    std::vector<int64_t> shape_vec = phi::vectorize(x.dims());
    phi::IntArray out_shape(shape_vec);
    custom_kernel::FullKernel<T, Context>(dev_ctx, out_shape, val, dtype, out);

  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    float value = val.to<float>();

    GcuAttributeMap attrs;
    attrs["dtype"] = static_cast<int>(dtype);
    attrs["value"] = value;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "fill_any_like",
              dev_ctx);
  }
}

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const std::vector<int>& shape,
                             const phi::Scalar& val,
                             phi::DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             phi::DenseTensor* out) {
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);

    std::vector<int> out_shpae = shape;
    if (x_batch_size_dim == 0 && x.dims().size() > 0) {
      out_shpae[out_batch_size_dim] = x.dims().at(0);
    }

    custom_kernel::FullKernel<T, Context>(dev_ctx, out_shpae, val, dtype, out);
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(full,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(full_batch_size_like,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FullBatchSizeLikeKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
