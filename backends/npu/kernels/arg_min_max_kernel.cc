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
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  dev_ctx.Alloc(out, out->dtype());
  auto stream = dev_ctx.stream();

  if (dtype == phi::DataType::INT32) {
    NpuOpRunner runner;
    runner.SetType("ArgMin")
        .AddInput(x)
        .AddInput(dev_ctx, std::vector<int64_t>({axis.to<int64_t>()}))
        .AddOutput(*out)
        .AddAttr("dtype", ConvertToNpuDtype(dtype));
    runner.Run(stream);
  } else if (dtype == phi::DataType::INT64) {
    // TODO(songkai05): core dump happend when the dtype of CANN op ArgMin's
    // output is int64, so we compute the int32 result and cast it to int64 when
    // param dtype is INT64 temporarily.
    NPUAttributeMap attrs = {{"dtype", ConvertToNpuDtype(dtype)}};

    auto op_runner = [](const std::vector<phi::DenseTensor>& inputs,
                        const std::vector<phi::DenseTensor>& outputs,
                        const NPUAttributeMap& attrs,
                        const phi::CustomContext& dev_ctx,
                        const auto& host_vecs) {
      NpuOpRunner runner;
      runner.SetType("ArgMin")
          .AddInput(inputs[0])
          .AddInput(dev_ctx, std::move(host_vecs[0]))
          .AddOutput(outputs[0])
          .AddAttrs(attrs);
      runner.Run(dev_ctx.stream());
    };

    NpuOpRunner::TypeAdapter<int64_t>(
        {x},
        {*out},
        attrs,
        dev_ctx,
        op_runner,
        {x.dtype()},
        {phi::DataType::INT32},
        {std::vector<int64_t>({axis.to<int64_t>()})});
  }
}

template <typename T, typename Context>
void AclopArgMaxKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::Scalar& axis,
                       bool keepdims,
                       bool flatten,
                       phi::DataType dtype,
                       phi::DenseTensor* out) {
  dev_ctx.Alloc(out, out->dtype());
  auto stream = dev_ctx.stream();

  phi::DenseTensor transformed_x;
  // TODO(songkai05): CANN512 doesn't support double dtype for ArgMax NPU op,
  // we cast double to float32 to support double dtype for now.
  if (x.dtype() == phi::DataType::FLOAT64 ||
      x.dtype() == phi::DataType::INT32) {
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

  NpuOpRunner runner;
  runner.SetType("ArgMaxV2")
      .AddInput(transformed_x)
      .AddInput(dev_ctx, std::move(axis_v))
      .AddOutput(*out)
      .AddAttrDataType("dtype", static_cast<int>(dtype))
      .Run(stream);
}

template <typename T, typename Context>
void AclnnArgMaxKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::Scalar& axis,
                       bool keepdims,
                       bool flatten,
                       phi::DataType dtype,
                       phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  phi::DenseTensor transformed_x;
  if (x.dtype() == phi::DataType::FLOAT64 ||
      x.dtype() == phi::DataType::INT32) {
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, x.dims()};
    transformed_x.set_meta(meta);
    dev_ctx.template Alloc<float>(&transformed_x);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::FLOAT32, &transformed_x);
  } else {
    transformed_x = x;
  }
  int64_t axis_v = axis.to<int64_t>();
  phi::DenseTensor cast_out;
  phi::DenseTensorMeta out_meta = {out->dtype(), out->dims()};
  cast_out.set_meta(out_meta);
  dev_ctx.Alloc(out, out->dtype());

  if (flatten) {
    transformed_x.Resize(phi::make_ddim({x.numel()}));
    out->Resize(phi::make_ddim({x.numel()}));
  } else {
    std::vector<int> out_shape;
    auto in_dims = x.dims();
    auto vec_in_dims = phi::vectorize<int>(in_dims);
    int dim_v;
    if (axis_v == -1) {
      dim_v = vec_in_dims.size() - 1;
    } else {
      dim_v = axis_v;
    }
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      if (i == dim_v) {
        if (keepdims) {
          out_shape.push_back(1);
        }
        continue;
      }
      out_shape.push_back(vec_in_dims[i]);
    }
    out->Resize(phi::make_ddim(out_shape));
  }
  dev_ctx.template Alloc<int64_t>(&cast_out);
  EXEC_NPU_CMD(aclnnArgMax, dev_ctx, transformed_x, axis_v, keepdims, cast_out);
  custom_kernel::CastKernel<T, Context>(dev_ctx, cast_out, dtype, out);
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnArgMax,
                   (custom_kernel::AclopArgMaxKernel<T, Context>(
                       dev_ctx, x, axis, keepdims, flatten, dtype, out)));
  dev_ctx.template Alloc<T>(out);
  AclnnArgMaxKernel<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argmin,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(argmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
