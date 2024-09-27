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
void InterpolateKernel(
    const std::string& Interpolated_type,
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  auto input = x;
  auto input_dims = input.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4UL,
      phi::errors::External(
          "GCU Interpolate Kernel only support 4-D Tensor so far as now."));

  phi::DataLayout input_layout = StringToDataLayout(data_layout);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input_dims, input_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_h = -1;
  float scale_w = -1;
  std::vector<float> new_scale(scale);
  // Priority: size_tensor > out_size > scale_tensor > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    auto tensors = size_tensor.get();
    auto output_h = get_new_data_from_tensor<int>(ctx, tensors[0]);
    auto output_w = get_new_data_from_tensor<int>(ctx, tensors[1]);
    out_h = output_h[0];
    out_w = output_w[0];
  } else if (out_size) {
    auto size_data = get_new_data_from_tensor<int>(ctx, out_size.get_ptr());
    out_h = size_data[0];
    out_w = size_data[1];
  } else if (scale_tensor) {
    auto scale_data =
        get_new_data_from_tensor<float>(ctx, scale_tensor.get_ptr());
    if (scale_data.size() > 1) {
      scale_h = scale_data[0];
      scale_w = scale_data[1];
    } else {
      scale_h = scale_data[0];
      scale_w = scale_data[0];
    }
    PADDLE_ENFORCE_GT(scale_h,
                      0,
                      phi::errors::InvalidArgument(
                          "The scale_h in input 'scale_tensor' Tensor of "
                          "Operator(interpolate) "
                          "should be greater than 0, but received value is %d.",
                          scale_h));
    PADDLE_ENFORCE_GT(scale_w,
                      0,
                      phi::errors::InvalidArgument(
                          "The scale_w in input 'scale_tensor' Tensor of "
                          "Operator(interpolate) "
                          "should be greater than 0, but received value is %d.",
                          scale_w));
    new_scale = {scale_h, scale_w};
  } else {
    if (scale.size() > 1) {
      scale_h = scale[0];
      scale_w = scale[1];
      PADDLE_ENFORCE_GT(
          scale_w,
          0,
          phi::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_GT(
          scale_h,
          0,
          phi::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    }
  }
  if (scale_h > 0. && scale_w > 0.) {
    out_h = static_cast<int>(in_h * scale_h);
    out_w = static_cast<int>(in_w * scale_w);
  }

  PADDLE_ENFORCE_GT(out_h,
                    0,
                    phi::errors::InvalidArgument("out_h  of Op(interpolate) "
                                                 "should be greater than 0."));
  PADDLE_ENFORCE_GT(out_w,
                    0,
                    phi::errors::InvalidArgument("out_w  of Op(interpolate) "
                                                 "should be greater than 0."));
  phi::DDim dim_out;
  if (input_layout == phi::DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }

  phi::DenseTensorMeta out_meta(output->dtype(), dim_out);
  output->set_meta(out_meta);
  ctx.template Alloc<T>(output);
  if (LaunchAOTKernel()) {
    phi::DenseTensor output_perm = *output;
    if (DataPdCustomNHWC(x)) {
      PADDLE_ENFORCE_EQ(data_layout,
                        "NCHW",
                        phi::errors::InvalidArgument(
                            "Layout of kernel attr should be NCHW."));
      PdCustomNHWCRepresentAsAtenNHWC(input);
      PdCustomNHWCRepresentAsAtenNHWC(output_perm, true);
    } else {
      if (data_layout == "NHWC") {
        PermutedShapeAndStrides(
            input, layout_trans::kNHWC_to_NCHW, common::DataLayout::kNCHW);
        PermutedShapeAndStrides(output_perm,
                                layout_trans::kNHWC_to_NCHW,
                                common::DataLayout::kNCHW);
      }
    }
    auto aten_x = CreateTopsatenTensor(input);
    auto aten_out = CreateTopsatenTensor(output_perm);
    std::vector<int64_t> output_size{out_h, out_w};
    topsatenSize_t aten_output_size{output_size.data(),
                                    static_cast<int64_t>(output_size.size())};
    topsatenScalar_t aten_scales_h;
    aten_scales_h.dtype = TOPSATEN_DATA_NONE;
    aten_scales_h.fval = scale_h;
    topsatenScalar_t aten_scales_w;
    aten_scales_w.dtype = TOPSATEN_DATA_NONE;
    aten_scales_w.fval = scale_w;

    if (interp_method == "nearest") {
      std::string abstract_info =
          custom_kernel::GetAbstractInfo("topsatenUpsampleNearest2d",
                                         output_perm,
                                         input,
                                         output_size,
                                         scale_h,
                                         scale_w);
      LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenUpsampleNearest2d,
                                          ctx,
                                          abstract_info,
                                          aten_out,
                                          aten_x,
                                          aten_output_size,
                                          aten_scales_h,
                                          aten_scales_w);

    } else if (interp_method == "bilinear") {
      PADDLE_ENFORCE_EQ((!align_corners && align_mode == 1),
                        false,
                        phi::errors::Unimplemented(
                            "Interpolate bilinear AOT kernel is unimplemented "
                            "for align_corners(false) and align_mode(1)"));
      std::string abstract_info =
          custom_kernel::GetAbstractInfo("topsatenUpsampleBilinear2d",
                                         output_perm,
                                         input,
                                         output_size,
                                         scale_h,
                                         scale_w);
      LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenUpsampleBilinear2d,
                                          ctx,
                                          abstract_info,
                                          aten_out,
                                          aten_x,
                                          aten_output_size,
                                          align_corners,
                                          aten_scales_h,
                                          aten_scales_w);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Interpolate AOT kernel is unimplemented mode[%s]",
          interp_method.c_str()));
    }

    if (DataPdCustomNHWC(x)) {
      AtenNHWCRepresentAsPdCustomNHWC(output_perm);
      AtenNHWCRepresentAsPdCustomNHWC(*output, true);
    }

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Output"] = {"output"};

    TensorValueMap outputs;
    outputs["Output"] = {output};

    GcuAttributeMap attrs;
    attrs["data_layout"] = data_layout;
    attrs["out_d"] = out_d;
    attrs["out_h"] = out_h;
    attrs["out_w"] = out_w;
    attrs["scale"] = new_scale;
    attrs["interp_method"] = interp_method;
    attrs["align_corners"] = align_corners;
    attrs["align_mode"] = align_mode;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              Interpolated_type,
              ctx);
  }
}

template <typename T, typename Context>
void InterpolateGradKernel(
    const std::string& Interpolated_grad_type,
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  if (out_size) {
    input_names["OutSize"] = {"out_size"};
    inputs["OutSize"] = {const_cast<DenseTensor*>(out_size.get_ptr())};
  }
  if (scale_tensor) {
    input_names["Scale"] = {"scale"};
    inputs["Scale"] = {const_cast<DenseTensor*>(scale_tensor.get_ptr())};
  }
  if (size_tensor) {
    auto tensors = size_tensor.get();
    std::vector<std::string> in_names;
    in_names.reserve(tensors.size());
    std::vector<phi::DenseTensor*> in_tensors;
    in_tensors.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      in_names.emplace_back(std::string("size_tensor_") + std::to_string(i));
      in_tensors.emplace_back(const_cast<DenseTensor*>(tensors[i]));
    }
    input_names["SizeTensor"] = in_names;
    inputs["SizeTensor"] = in_tensors;
  }

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"x_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {x_grad};

  GcuAttributeMap attrs;
  attrs["data_layout"] = data_layout;
  attrs["out_d"] = out_d;
  attrs["out_h"] = out_h;
  attrs["out_w"] = out_w;
  attrs["scale"] = scale;
  attrs["interp_method"] = interp_method;
  attrs["align_corners"] = align_corners;
  attrs["align_mode"] = align_mode;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            Interpolated_grad_type,
            ctx);
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  PADDLE_GCU_KERNEL_TRACE("bilinear_interp");
  InterpolateKernel<T, Context>("bilinear_interp_v2",
                                ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void BilinearInterpGradKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("bilinear_interp_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    InterpolateGradKernel<T, Context>("bilinear_interp_v2_grad",
                                      ctx,
                                      x,
                                      out_size,
                                      size_tensor,
                                      scale_tensor,
                                      out_grad,
                                      data_layout,
                                      out_d,
                                      out_h,
                                      out_w,
                                      scale,
                                      interp_method,
                                      align_corners,
                                      align_mode,
                                      x_grad);
  }
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  PADDLE_GCU_KERNEL_TRACE("nearest_interp");
  InterpolateKernel<T, Context>("nearest_interp_v2",
                                ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpGradKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("nearest_interp_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    InterpolateGradKernel<T, Context>("nearest_interp_v2_grad",
                                      ctx,
                                      x,
                                      out_size,
                                      size_tensor,
                                      scale_tensor,
                                      out_grad,
                                      data_layout,
                                      out_d,
                                      out_h,
                                      out_w,
                                      scale,
                                      interp_method,
                                      align_corners,
                                      align_mode,
                                      x_grad);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
