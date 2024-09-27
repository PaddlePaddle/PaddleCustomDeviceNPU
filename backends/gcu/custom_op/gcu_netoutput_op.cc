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

#include "custom_op/custom_op_common.h"

std::vector<std::vector<int64_t>> GcuNetOutputInferShape(
    const std::vector<int64_t>& x_shape, int origin_out_dtype) {
  return {x_shape};
}

std::vector<paddle::Tensor> GcuNetOutput(const paddle::Tensor& x,
                                         int origin_out_dtype) {
  PADDLE_GCU_KERNEL_TRACE("gcu_netoutput");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: gcu_netoutput";
  VLOG(3) << "gcu_netoutput, x dtype:" << phi::DataTypeToString(x.dtype());

  auto x_dims = x.dims();
  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  phi::DenseTensor trans_out = *x_tensor;
  if (custom_kernel::DataPdCustomNHWC(*x_tensor)) {
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        4,
        phi::errors::InvalidArgument(
            "Only support 4D tensor, but get x rank %d.", x_dims.size()));

    trans_out = custom_kernel::PdCustomNHWCTransToNCHW(*dev_ctx, *x_tensor);
  }

  phi::DataType out_dtype = phi::TransToPhiDataType(origin_out_dtype);
  VLOG(3) << "gcu_netoutput, x_tensor dtype:"
          << phi::DataTypeToString(x_tensor->dtype())
          << ", trans_out dtype:" << phi::DataTypeToString(trans_out.dtype())
          << ", dst dtype:" << phi::DataTypeToString(out_dtype)
          << ", native attr dtype:" << origin_out_dtype;
  auto out_pinned =
      custom_kernel::CastOrCopyToPinnedMemory(*dev_ctx, trans_out, out_dtype);
  auto out = custom_op_common::CreateTensorFromDenseTensor(out_pinned);
  //   dev_ctx->Wait();
  return {out};
}

std::vector<paddle::DataType> GcuNetOutputInferDtype(
    const paddle::DataType& x_dtype, int origin_out_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(gcu_netoutput)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"origin_out_dtype: int"})
    .SetKernelFn(PD_KERNEL(GcuNetOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(GcuNetOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GcuNetOutputInferDtype));
