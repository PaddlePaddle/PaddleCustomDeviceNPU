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
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("masked_select");
  if (LaunchAOTKernel()) {
    // topsatenMaskedSelect does not refresh the meta information of output.
    THROW_AOT_UNIMPLEMENTED();

    auto x_tensor = CreateTopsatenTensor(x);
    auto mask_tensor = CreateTopsatenTensor(mask);
    auto out_tensor = CreateTopsatenTensorWithoutInitialized(*out);
    std::vector<int64_t> tensor_strides = {1};
    auto strides = topsatenSize_t{tensor_strides.data(), 1};
    out_tensor.SetTensorStrides(strides);
    std::string abstract_info =
        custom_kernel::GetAbstractInfo("topsatenMaskedSelect", *out, x, mask);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenMaskedSelect,
                                        dev_ctx,
                                        abstract_info,
                                        out_tensor,
                                        x_tensor,
                                        mask_tensor);
    topsatenSize_t aten_out_shape = out_tensor.GetTensorShape();
    std::vector<int64_t> out_shape(aten_out_shape.data,
                                   aten_out_shape.data + aten_out_shape.len);
    out->Resize(common::make_ddim(out_shape));

    VLOG(6) << "MaskedSelectKernel, out tensor shape:" << out->dims();
    dev_ctx.template Alloc<T>(out);
    C_Device_st device;
    device.id = out->place().GetDeviceId();
    C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());
    auto bytes_size = out->numel() * phi::SizeOf(out->dtype());
    AsyncMemCpyD2D(&device,
                   stream,
                   out->data(),
                   static_cast<void*>(out_tensor.GetTensorData()),
                   bytes_size);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Mask"] = {"mask"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Mask"] = {const_cast<DenseTensor*>(&mask)};

    TensorNameMap output_names;
    output_names["Y"] = {"out"};

    TensorValueMap outputs;
    outputs["Y"] = {out};

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              {},
              "masked_select",
              dev_ctx);
  }
}

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& mask,
                            const phi::DenseTensor& out_grad,
                            phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("masked_select_grad");

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(x_grad);
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Mask"] = {"mask"};
    input_names[GradVarName("Y")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Mask"] = {const_cast<DenseTensor*>(&mask)};
    inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {x_grad};

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              {},
              "masked_select_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(masked_select,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MaskedSelectKernel,
                          phi::dtype::float16,
                          float,
                          int) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}

// PD_REGISTER_PLUGIN_KERNEL(masked_select_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::MaskedSelectGradKernel,
//                           phi::dtype::float16,
//                           float,
//                           int,
//                           int64_t) {
//   kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
// }
