/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  phi::DenseTensor out_tmp;
  out_tmp.Resize(out->dims());
  dev_ctx.template Alloc<int32_t>(&out_tmp);
  dev_ctx.Alloc(out, out->dtype());
  auto stream = dev_ctx.stream();

  NpuOpRunner runner;
  runner.SetType("ArgMin")
      .AddInput(x)
      .AddInput(dev_ctx, std::vector<int64_t>({axis.to<int64_t>()}))
      .AddOutput(out_tmp)
      .AddAttr("dtype", dtype);

  runner.Run(stream);
  dev_ctx.Wait();

  if (dtype == 2) {
    TensorCopy(dev_ctx, out_tmp, true, out);
  } else if (dtype == 3) {
    const auto& cast_runner =
        NpuOpRunner("Cast", {out_tmp}, {*out}, {{"dst_type", ACL_INT64}});
    cast_runner.Run(stream);
  }
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  dev_ctx.Alloc(out, out->dtype());
  auto stream = dev_ctx.stream();

  phi::DenseTensor transformed_x;
  // TODO(songkai05): CANN512 doesn't support double dtype for ArgMax NPU op,
  // we cast double to float32 to support double dtype for now.
  if (x.dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, x.dims()};
    transformed_x.set_meta(meta);
    dev_ctx.template Alloc<float>(&transformed_x);
    const auto& cast_runner =
        NpuOpRunner("Cast", {x}, {transformed_x}, {{"dst_type", ACL_FLOAT}});
    cast_runner.Run(stream);
  } else {
    transformed_x = x;
  }
  if (flatten) {
    transformed_x.Resize(phi::make_ddim({x.numel()}));
  }

  std::vector<int64_t> axis_v;
  axis_v.push_back(axis.to<int64_t>());

  int out_dtype;
  if (dtype == 2) {
    out_dtype = static_cast<int>(phi::DataType::INT32);
  } else if (dtype == 3) {
    out_dtype = static_cast<int>(phi::DataType::INT64);
  }

  NpuOpRunner runner;
  runner.SetType("ArgMaxV2")
      .AddInput(transformed_x)
      .AddInput(dev_ctx, std::move(axis_v))
      .AddOutput(*out)
      .AddAttrDataType("dtype", out_dtype)
      .Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argmin,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(argmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
