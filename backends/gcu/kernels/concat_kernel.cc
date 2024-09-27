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
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("concat");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    bool use_nhwc = false;
    std::vector<phi::DenseTensor> input_tensors;
    for (const auto& in : ins) {
      input_tensors.emplace_back(MaybeCreateOrTrans64To32bits(dev_ctx, *in));
      if (EnableTransposeOptimize() && (!use_nhwc) &&
          in->layout() == common::DataLayout::kNHWC) {
        use_nhwc = true;
      }
    }
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    if (use_nhwc) {
      PdCustomNHWCRepresentAsAtenNHWC(output, true);
    }

    auto out_tensor = CreateTopsatenTensor(output);

    std::vector<topsatenTensor> in_tensors;
    for (auto& tensor : input_tensors) {
      if (use_nhwc) {
        if (tensor.layout() == common::DataLayout::kNHWC) {
          PdCustomNHWCRepresentAsAtenNHWC(tensor);
        } else {
          tensor = NCHWTransToAtenNHWC(dev_ctx, tensor);
        }
      }
      in_tensors.emplace_back(CreateTopsatenTensor(tensor));
    }
    int64_t dim = axis_scalar.to<int64_t>();
    if (dim < 0 && !ins.empty()) {
      dim += ins[0]->dims().size();
    }

    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "topsatenCat", output, input_tensors, dim);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(
        topsatenCat, dev_ctx, abstract_info, out_tensor, in_tensors, dim);

    if (use_nhwc) {
      AtenNHWCRepresentAsPdCustomNHWC(output);
    }

    MaybeTransResult(dev_ctx, output, out);
    if (use_nhwc) {
      AtenNHWCRepresentAsPdCustomNHWC(output);
    }
    VLOG(6) << "Transpose debug, ConcatKernel output:"
            << custom_kernel::TensorDetailsToString(*out);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    TensorValueMap inputs;
    std::vector<std::string> names;
    names.reserve(ins.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
      names.emplace_back(std::string("x_") + std::to_string(i));
      values.emplace_back(const_cast<DenseTensor*>(ins[i]));
    }
    input_names["X"] = names;
    inputs["X"] = values;

    TensorNameMap output_names;
    TensorValueMap outputs;

    output_names["Out"] = {"out"};
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["axis"] = axis_scalar.to<int>();

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "concat", dev_ctx);
  }
}

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& ins,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("concat_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    TensorValueMap inputs;
    {
      std::vector<std::string> names;
      names.reserve(ins.size());
      std::vector<phi::DenseTensor*> values;
      values.reserve(ins.size());
      for (size_t i = 0; i < ins.size(); ++i) {
        names.emplace_back(std::string("x_") + std::to_string(i));
        values.emplace_back(const_cast<DenseTensor*>(ins[i]));
      }
      input_names["X"] = names;
      inputs["X"] = values;
    }

    input_names[GradVarName("Out")] = {"dout"};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    {
      std::vector<std::string> names;
      names.reserve(outs.size());
      std::vector<phi::DenseTensor*> values;
      values.reserve(outs.size());
      for (size_t i = 0; i < outs.size(); ++i) {
        if ((outs[i] != nullptr) && (outs[i]->numel() != 0UL)) {
          dev_ctx.template Alloc<T>(outs[i]);
          names.emplace_back(
              GradVarName(std::string("x_") + std::to_string(i)));
          values.emplace_back(outs[i]);
        }
      }
      output_names[GradVarName("X")] = names;
      outputs[GradVarName("X")] = values;
    }

    GcuAttributeMap attrs;
    attrs["axis"] = axis_scalar.to<int>();

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "concat_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          bool,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t,
                          uint8_t) {}

PD_REGISTER_PLUGIN_KERNEL(concat_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatGradKernel,
                          bool,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t,
                          uint8_t) {}
