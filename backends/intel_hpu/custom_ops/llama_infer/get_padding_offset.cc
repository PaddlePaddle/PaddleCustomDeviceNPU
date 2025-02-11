// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len) {
  /*
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          input_ids.place()));
  std::shared_ptr<phi::DenseTensor> padding_offset =
      std::make_shared<phi::DenseTensor>();
  padding_offset->Resize(input_ids.dims());
  dev_ctx->Alloc(padding_offset.get(), paddle::DataType::INT32);
  */
  return {input_ids, cum_offsets, token_num};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& token_num_shape,
    const std::vector<int64_t>& seq_len_shape) {
  return {input_ids_shape, {-1}, {-1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& token_num_dtype,
    const paddle::DataType& seq_len_dtype) {
  return {input_ids_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({paddle::Optional("x_remove_padding"),
              paddle::Optional("cum_offsets_out"),
              paddle::Optional("padding_offset")})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));
