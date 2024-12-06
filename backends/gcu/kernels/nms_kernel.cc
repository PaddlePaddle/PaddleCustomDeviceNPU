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

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void NMSKernel(const Context& dev_ctx,
               const DenseTensor& boxes,
               float threshold,
               DenseTensor* output) {
  PADDLE_GCU_KERNEL_TRACE("nms");
  // firstly, we need to set output shape to max size and alloc max size memory
  output->Resize({boxes.dims()[0]});
  dev_ctx.template Alloc<int64_t>(output);

  if (LaunchAOTKernel()) {
    auto boxes_num = boxes.dims().at(0);

    phi::DenseTensor cpu_tensor;
    phi::DenseTensorMeta scores_meta = {phi::DataType::FLOAT32,
                                        phi::make_ddim({boxes_num})};
    cpu_tensor.set_meta(scores_meta);
    float* host_mask = dev_ctx.template HostAlloc<float>(&cpu_tensor);
    for (size_t i = 0; i < boxes_num; i++) {
      host_mask[i] = boxes_num - i;
    }

    phi::DenseTensor scores_tensor =
        custom_kernel::TensorEmpty(dev_ctx, scores_meta);

    // copy mask to device
    TensorCopy(dev_ctx, cpu_tensor, true, &scores_tensor);

    PADDLE_ENFORCE_EQ(output->initialized(),
                      true,
                      phi::errors::InvalidArgument(
                          "The output tensor should has been init."));
    PADDLE_ENFORCE_EQ(
        output->dims().size(),
        1,
        phi::errors::InvalidArgument("The output tensor should be 1D."));
    PADDLE_ENFORCE_EQ(
        boxes.dims().at(0),
        output->dims().at(0),
        phi::errors::InvalidArgument("The dim0 of boxes and the dim output "
                                     "should be equal."));

    DenseTensor out_imp = *output;
    if (output->dtype() != phi::DataType::INT32) {
      phi::DenseTensorMeta int32_meta = {phi::DataType::INT32,
                                         phi::make_ddim({boxes_num})};
      out_imp = custom_kernel::TensorEmpty(dev_ctx, int32_meta);
      dev_ctx.template Alloc<int32_t>(&out_imp);
    }

    // call topsatenNms
    auto outimp_aten = CreateTopsatenTensor(out_imp);
    auto boxes_aten = CreateTopsatenTensor(boxes);
    auto scores_aten = CreateTopsatenTensor(scores_tensor);

    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "NMS_topsatenNms", out_imp, boxes, scores_tensor, threshold);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenNms,
                                        dev_ctx,
                                        abstract_info,
                                        outimp_aten,
                                        boxes_aten,
                                        scores_aten,
                                        threshold);

    auto outimp_aten_shape = outimp_aten.GetTensorShape();
    std::vector<int64_t> output_shape(outimp_aten_shape.len, 0);
    for (int i = 0; i < outimp_aten_shape.len; i++) {
      output_shape[i] = outimp_aten_shape.data[i];
    }

    // refresh the output tensor shape
    out_imp.Resize(phi::make_ddim(output_shape));
    output->Resize(phi::make_ddim(output_shape));

    if (output->dtype() != out_imp.dtype()) {
      auto output_aten = CreateTopsatenTensor(*output);

      topsatenDataType_t topsaten_dtype =
          DataTypeToTopsatenDataType(output->dtype());
      topsatenMemoryFormat_t topsaten_format = TOPSATEN_MEMORY_PRESERVE;
      std::string abstract_info = custom_kernel::GetAbstractInfo(
          "NMS_topsatenTo", *output, out_imp, out_imp.dtype(), output->dtype());
      LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenTo,
                                          dev_ctx,
                                          abstract_info,
                                          output_aten,
                                          outimp_aten,
                                          topsaten_dtype,
                                          false,
                                          true,
                                          topsaten_format);
    }
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    nms, gcu, ALL_LAYOUT, custom_kernel::NMSKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
