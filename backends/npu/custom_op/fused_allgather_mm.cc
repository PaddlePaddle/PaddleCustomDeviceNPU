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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

static inline aclTensor* Create_Acltensor(const phi::DenseTensor& paddletensor,
                                          const bool transpose) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  auto tensor_dtype = paddletensor.dtype();
  auto acl_data_type = ConvertToNpuDtype(tensor_dtype);
  const auto dimNum =
      paddletensor.dims().size() == 0 ? 1 : paddletensor.dims().size();
  std::vector<int64_t> storageDims(dimNum - 1);
  storageDims.push_back(paddletensor.numel() * sizeof(tensor_dtype));
  aclFormat format = ACL_FORMAT_ND;
  switch (dimNum) {
    break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  auto shape = phi::vectorize(paddletensor.dims());
  auto strides = shape;
  if (!strides.empty()) {
    strides.erase(strides.begin());
  }
  strides.push_back(1);
  for (int i = static_cast<int>(strides.size()) - 2; i >= 0; i--) {
    strides[i] = strides[i] * strides[i + 1];
  }
  if (transpose) {
    std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
    std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
  }
  auto acl_tensor = aclCreateTensor(shape.data(),
                                    dimNum,
                                    acl_data_type,
                                    strides.data(),
                                    0,
                                    format,
                                    shape.data(),
                                    dimNum,
                                    const_cast<void*>(paddletensor.data()));
  return acl_tensor;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int CreateZeroDimAclTensor(const std::vector<float>& hostData,
                           const std::vector<int64_t>& shape,
                           void** deviceAddr,
                           aclDataType dataType,
                           aclTensor** tensor) {
  auto size = GetShapeSize(shape) * 2;
  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(),
                            shape.size(),
                            dataType,
                            strides.data(),
                            0,
                            aclFormat::ACL_FORMAT_ND,
                            shape.data(),
                            shape.size(),
                            *deviceAddr);
  return 0;
}

const phi::DDim get_output_size_gather_mm(const paddle::Tensor& x1,
                                          const paddle::Tensor& x2,
                                          int64_t world_size,
                                          int64_t gather_index,
                                          const bool transpose_y) {
  auto out_x = gather_index == 0 ? x1.dims()[0] * world_size : x1.dims()[0];
  if (transpose_y) {
    auto out_y = x2.dims()[0];
    return {out_x, out_y};
  } else {
    auto out_y = x2.dims()[1];
    return {out_x, out_y};
  }
}

const phi::DDim get_output_size_gather(const paddle::Tensor& x1,
                                       const paddle::Tensor& x2,
                                       int64_t world_size,
                                       int64_t gather_index,
                                       const bool transpose_y) {
  if (gather_index == 0) {
    return {x1.dims()[0] * world_size, x1.dims()[1]};
  } else {
    if (transpose_y) {
      return {x2.dims()[1] * world_size, x2.dims()[0]};
    } else {
      return {x2.dims()[0] * world_size, x2.dims()[1]};
    }
  }
}

std::vector<paddle::Tensor> npu_allgather_mm(
    const paddle::Tensor& x1,
    const paddle::Tensor& x2,
    const paddle::optional<paddle::Tensor>& bias,
    std::string hcom,
    int64_t world_size,
    int64_t gather_index,
    bool gather_output,
    int64_t comm_turn,
    const bool transpose_y) {
  PD_CHECK(
      x1.dims().size() == 2 && x2.dims().size() == 2,
      "Both inputs of mm are required to be 2D, but the actual inputs are ",
      x1.dims().size(),
      "D and ",
      x2.dims().size(),
      "D");
  if (transpose_y) {
    PD_CHECK(x1.dims()[1] == x2.dims()[1],
             "The K-axis in the two inputs of Matmul must be equal, but in "
             "reality, the K-axis of x1 is ",
             x1.dims()[1],
             " and the K-axis of x2 is ",
             x2.dims()[1]);
  } else {
    PD_CHECK(x1.dims()[1] == x2.dims()[0],
             "The K-axis in the two inputs of Matmul must be equal, but in "
             "reality, the K-axis of x1 is ",
             x1.dims()[1],
             " and the K-axis of x2 is ",
             x2.dims()[0]);
  }

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x1.place()));
  auto out_gather_mm_size =
      get_output_size_gather_mm(x1, x2, world_size, gather_index, transpose_y);
  auto out_gather_size =
      get_output_size_gather(x1, x2, world_size, gather_index, transpose_y);

  std::shared_ptr<phi::DenseTensor> out_gather_mm =
      std::make_shared<phi::DenseTensor>();
  out_gather_mm->Resize(out_gather_mm_size);
  dev_ctx->Alloc(out_gather_mm.get(), x1.dtype());

  std::shared_ptr<phi::DenseTensor> out_gather =
      std::make_shared<phi::DenseTensor>();

  phi::DenseTensor* bias_real = nullptr;
  if (bias) {
    auto bias_ptr = *(bias.get_ptr());
    bias_real = static_cast<phi::DenseTensor*>(bias_ptr.impl().get());
  } else {
    bias_real = new phi::DenseTensor();
  }

  auto x1_tensor = *(static_cast<const phi::DenseTensor*>(x1.impl().get()));
  auto x2_tensor = *(static_cast<const phi::DenseTensor*>(x2.impl().get()));

  aclTensor* y_acltensor = Create_Acltensor(x2_tensor, transpose_y);
  char* hcom_ptr = const_cast<char*>(hcom.data());
  aclTensor* out_gather_zerotensor = nullptr;
  if (gather_output) {
    out_gather->Resize(out_gather_size);
    dev_ctx->Alloc(out_gather.get(), x1.dtype());
#if (CANN_VERSION_CODE >= 700000)
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    EXEC_NPU_CMD(aclnnAllGatherMatmul,
                 *dev_ctx,
                 x1_tensor,
                 y_acltensor,
                 *bias_real,
                 hcom_ptr,
                 gather_index,
                 comm_turn,
                 stream_mode,
                 *out_gather_mm,
                 *out_gather);
#else
    PADDLE_THROW(::phi::errors::Unimplemented(
        "current cann version doed not support this kernel"));
#endif
  } else {
    // custom写法要求 必须要输出两个tensor，此处虽然不用但也要申请内存
    out_gather->Resize({1});
    dev_ctx->Alloc(out_gather.get(), x1.dtype());
    // 这里随便给一个整数就行，方便计算，不会额外占用内存空间
    std::vector<int64_t> selfShape = {4, 0};
    std::vector<float> selfHostData = {0};
    void* acl_tensor = nullptr;
    auto datatype = x1_tensor.dtype();
    auto acl_datatype = ConvertToNpuDtype(datatype);
    // 此处直接调aclnn接口创建0维acltensor，在EXEC_NPU_CMD处透传
    auto ret = CreateZeroDimAclTensor(selfHostData,
                                      selfShape,
                                      &acl_tensor,
                                      acl_datatype,
                                      &out_gather_zerotensor);
#if (CANN_VERSION_CODE >= 700000)
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    EXEC_NPU_CMD(aclnnAllGatherMatmul,
                 *dev_ctx,
                 x1_tensor,
                 y_acltensor,
                 *bias_real,
                 hcom_ptr,
                 gather_index,
                 comm_turn,
                 stream_mode,
                 *out_gather_mm,
                 out_gather_zerotensor);
#else
    PADDLE_THROW(::phi::errors::Unimplemented(
        "current cann version doed not support this kernel"));
#endif
  }
  return {paddle::Tensor(out_gather_mm), paddle::Tensor(out_gather)};
}

std::vector<std::vector<int64_t>> FusedAllgatherMMInferShape(
    std::vector<int64_t> x1_shape, std::vector<int64_t> x2_shape) {
  return {x1_shape, x2_shape};
}

PD_BUILD_OP(fused_allgather_mm)
    .Inputs({"x1", "x2", paddle::Optional("bias")})
    .Outputs({"out_gather_mm", "out_gather"})
    .Attrs({"hcom:std::string",
            "world_size:int64_t",
            "gather_index:int64_t",
            "gather_output:bool",
            "comm_turn:int64_t",
            "transpose_y:bool"})
    .SetKernelFn(PD_KERNEL(npu_allgather_mm))
    .SetInferShapeFn(PD_INFER_SHAPE(
        FusedAllgatherMMInferShape));  // neccessary if the op has muti_inputs
