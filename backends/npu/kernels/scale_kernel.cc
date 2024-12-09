// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void FillKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& val);

template <typename T, typename Context>
void AclopScaleKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::Scalar& in_scale,
                      const phi::Scalar& in_bias,
                      bool bias_after_scale,
                      phi::DenseTensor* out) {
  auto scale = in_scale.to<float>();
  auto bias = in_bias.to<float>();
  auto stream = dev_ctx.stream();
  float power = 1.0;
  VLOG(4) << "scale:" << scale << ", bias:" << bias
          << " ,bias_after_scale:" << bias_after_scale;
  dev_ctx.template Alloc<T>(out);
  if (std::isinf(scale) || std::isnan(scale)) {
    FillNpuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(scale));
    return;
  }
  if (!bias_after_scale) {
    bias *= scale;
  }

  NPUAttributeMap attrs = {{"power", power}, {"scale", scale}, {"shift", bias}};

  auto op_func1 = [](const std::vector<phi::DenseTensor>& inputs,
                     const std::vector<phi::DenseTensor>& outputs,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext& dev_ctx) {
    const auto& muls_runner = NpuOpRunner(
        "Muls", {inputs[0]}, {outputs[0]}, {{"value", attrs.at("scale")}});
    muls_runner.Run(dev_ctx.stream());

    const auto& adds_runner = NpuOpRunner(
        "Adds", {outputs[0]}, {outputs[0]}, {{"value", attrs.at("shift")}});
    adds_runner.Run(dev_ctx.stream());
  };

  auto op_func2 = [](const std::vector<phi::DenseTensor>& inputs,
                     const std::vector<phi::DenseTensor>& outputs,
                     const NPUAttributeMap& attrs,
                     const phi::CustomContext& dev_ctx) {
    const auto& power_runner =
        NpuOpRunner("Power", {inputs[0]}, {outputs[0]}, attrs);
    power_runner.Run(dev_ctx.stream());
  };

  if (x.dtype() == phi::DataType::INT32 || x.dtype() == phi::DataType::INT64) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attrs,
                             dev_ctx,
                             op_func1,
                             {phi::DataType::INT32},
                             {phi::DataType::INT32});
  } else {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attrs,
                             dev_ctx,
                             op_func2,
                             {phi::DataType::FLOAT32},
                             {phi::DataType::FLOAT32});
  }
}

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 const phi::Scalar& in_bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  if (!FLAGS_npu_scale_aclnn) {
    return custom_kernel::AclopScaleKernel<T, Context>(
        dev_ctx, x, in_scale, in_bias, bias_after_scale, out);
  }

  DO_COMPATIBILITY(aclnnScale,
                   (custom_kernel::AclopScaleKernel<T, Context>(
                       dev_ctx, x, in_scale, in_bias, bias_after_scale, out)));

  auto scale = in_scale.to<float>();
  auto bias = in_bias.to<float>();
  float alpha = 1.0;
  VLOG(4) << "scale:" << scale << ", bias:" << bias
          << " ,bias_after_scale:" << bias_after_scale;

  if (std::isinf(scale) || std::isnan(scale)) {
    FillNpuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(scale));
    return;
  }

  if (!bias_after_scale) {
    bias *= scale;
  }

  if (x.dtype() == phi::DataType::INT64 || x.dtype() == phi::DataType::INT32) {
    phi::Scalar scale_scalar = scale;
    phi::Scalar bias_scalar = bias;
    phi::Scalar alpha_scalar = alpha;

    phi::DenseTensor cast_x;
    if (x.dtype() == phi::DataType::INT32) {
      cast_x = x;
    } else {
      cast_x.Resize(x.dims());
      dev_ctx.Alloc(&cast_x, phi::DataType::INT32);
      custom_kernel::CastKernel<T, Context>(
          dev_ctx, x, cast_x.dtype(), &cast_x);
    }

    phi::DenseTensor mid_out;
    mid_out.Resize(out->dims());
    dev_ctx.Alloc(&mid_out, phi::DataType::FLOAT32);

    EXEC_NPU_CMD(aclnnMuls, dev_ctx, cast_x, scale_scalar, mid_out);
    EXEC_NPU_CMD(
        aclnnAdds, dev_ctx, mid_out, bias_scalar, alpha_scalar, mid_out);

    dev_ctx.template Alloc<T>(out);
    custom_kernel::CastKernel<T, Context>(dev_ctx, mid_out, out->dtype(), out);

  } else if (x.dtype() == phi::DataType::FLOAT64) {
    phi::Scalar scale_scalar = scale;
    phi::Scalar bias_scalar = bias;
    phi::Scalar alpha_scalar = alpha;

    phi::DenseTensor cast_x;
    cast_x.Resize(x.dims());
    dev_ctx.Alloc(&cast_x, phi::DataType::FLOAT32);
    custom_kernel::CastKernel<T, Context>(dev_ctx, x, cast_x.dtype(), &cast_x);

    phi::DenseTensor mid_out;
    mid_out.Resize(out->dims());
    dev_ctx.Alloc(&mid_out, phi::DataType::FLOAT32);

    EXEC_NPU_CMD(aclnnMuls, dev_ctx, cast_x, scale_scalar, mid_out);
    EXEC_NPU_CMD(
        aclnnAdds, dev_ctx, mid_out, bias_scalar, alpha_scalar, mid_out);

    dev_ctx.template Alloc<T>(out);
    custom_kernel::CastKernel<T, Context>(dev_ctx, mid_out, out->dtype(), out);

  } else {
    phi::DenseTensor scale_tensor, bias_tensor, x_tmp;

    if (x.dims().size() == 0) {
      x_tmp.set_meta(x.meta());
      dev_ctx.template Alloc<T>(&x_tmp);
      TensorCopy(dev_ctx, x, false, &x_tmp);
      x_tmp.Resize({1});
    } else {
      x_tmp = x;
    }

    scale_tensor.set_meta(x_tmp.meta());
    dev_ctx.template Alloc<T>(&scale_tensor);

    aclDataType acl_data_type = ConvertToNpuDtype(x_tmp.dtype());
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);

    auto scale_scalar = static_cast<T>(scale);
    aclScalar* acl_scale_scalar = aclCreateScalar(&scale_scalar, acl_data_type);
    EXEC_NPU_CMD(
        aclnnInplaceFillScalar, dev_ctx, scale_tensor, acl_scale_scalar);

    if (bias != 0) {
      bias_tensor.set_meta(x_tmp.meta());
      dev_ctx.template Alloc<T>(&bias_tensor);

      auto bias_scalar = static_cast<T>(bias);
      aclScalar* acl_bias_scalar = aclCreateScalar(&bias_scalar, acl_data_type);
      EXEC_NPU_CMD(
          aclnnInplaceFillScalar, dev_ctx, bias_tensor, acl_bias_scalar);
    }

    int64_t axis = 0;
    int64_t num_axes = -1;
    bool scale_from_blob = false;
    dev_ctx.template Alloc<T>(out);

    if (x.dims().size() == 0) {
      phi::DenseTensor mid_out;
      mid_out.set_meta(x_tmp.meta());
      dev_ctx.template Alloc<T>(&mid_out);

      EXEC_NPU_CMD(aclnnScale,
                   dev_ctx,
                   x_tmp,
                   scale_tensor,
                   bias_tensor,
                   axis,
                   num_axes,
                   scale_from_blob,
                   mid_out);

      auto out_dims = out->dims();
      TensorCopy(dev_ctx, mid_out, false, out);
      out->Resize(out_dims);
    } else {
      EXEC_NPU_CMD(aclnnScale,
                   dev_ctx,
                   x_tmp,
                   scale_tensor,
                   bias_tensor,
                   axis,
                   num_axes,
                   scale_from_blob,
                   *out);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          int,
                          int64_t) {}
