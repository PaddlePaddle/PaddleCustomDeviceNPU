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
void OneHotKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& num_classes_s,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("one_hot");
  int64_t depth = num_classes_s.to<int>();
  auto out_dims = out->dims();
  if (out_dims.size() > 0 && out_dims[out_dims.size() - 1] == -1) {
    out_dims[out_dims.size() - 1] = depth;
    out->Resize(out_dims);
  }
  dev_ctx.template Alloc<float>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor aten_out;
    auto meta = out->meta();
    meta.dtype = phi::DataType::INT32;
    aten_out.set_meta(meta);
    dev_ctx.template Alloc<int32_t>(&aten_out);
    LAUNCH_TOPSATENOP(topsatenOneHot, dev_ctx, aten_out, x, depth);
    custom_kernel::Cast(dev_ctx, aten_out, out->dtype(), out);

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
    attrs["depth"] = depth;
    attrs["dtype"] = static_cast<int>(phi::DataType::FLOAT32);

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "one_hot_v2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void OneHotV2Kernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& num_classes_s,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("one_hot_v2");
  custom_kernel::OneHotKernel<T, Context>(dev_ctx, x, num_classes_s, out);
}

template <typename T, typename Context>
void OneHotRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& num_classes_s,
                     phi::DataType dtype,
                     bool allow_out_of_range UNUSED,
                     phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("one_hot_raw");
  custom_kernel::OneHotKernel<T, Context>(dev_ctx, x, num_classes_s, out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    one_hot, gcu, ALL_LAYOUT, custom_kernel::OneHotKernel, int32_t, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_PLUGIN_KERNEL(one_hot_v2,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::OneHotV2Kernel,
                          int32_t,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}

PD_REGISTER_PLUGIN_KERNEL(one_hot_raw,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::OneHotRawKernel,
                          int32_t,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
