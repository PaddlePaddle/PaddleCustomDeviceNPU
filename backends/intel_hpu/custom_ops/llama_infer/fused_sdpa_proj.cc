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

namespace custom_kernel {

class FusedSdpaProj : public HpuOperator {
 public:
  explicit FusedSdpaProj(synDataType dtype)
      : HpuOperator("fused_sdpa_proj_", false), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, ns_Sdpa::ParamsV2 params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    synStatus status = synFail;

    std::string name_sdpa = guid_ + "sdpa";
    std::string name_trans = guid_ + "transpose";
    std::string name_reshape = guid_ + "reshape";
    std::string name_gemm = guid_ + "gemm";

    std::string guid_sdpa = "sdpa_recomp_fwd_";
    std::string guid_reshape = "reshape";
    std::string guid_trans = "transpose";
    std::string guid_gemm = "gemm";

    if (dtype_ == syn_type_fp16) {
      guid_sdpa = guid_sdpa + "f16";
    } else if (dtype_ == syn_type_bf16) {
      guid_sdpa = guid_sdpa + "bf16";
    }

    std::vector<synTensor> attn_inputs;
    // params.is_causal = true; ==> input[3] is not used
    // input[3] is in use ==> params.is_causal = false;
    for (size_t i = 0; i < 4; i++) {
      attn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                         inputs[i].type,
                                         inputs[i].dims,
                                         true,
                                         inputs[i].name));
    }
    std::vector<synTensor> attn_outputs;
    auto attn = createTensor(
        inputs[0].dims.size(), inputs[0].type, inputs[0].dims, false, "attn");
    attn_outputs.push_back(attn);

    status = synNodeCreate(graphHandle_,
                           attn_inputs.data(),
                           attn_outputs.data(),
                           attn_inputs.size(),
                           attn_outputs.size(),
                           &params,
                           sizeof(params),
                           guid_sdpa.c_str(),
                           name_sdpa.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate () failed = ",
             status);

    std::vector<int64_t> q_dims = std::vector<int64_t>(inputs[0].dims);
    std::vector<int64_t> qt_dims(q_dims.cbegin(), q_dims.cend());
    int rank = q_dims.size();
    qt_dims[rank - 3] = q_dims[rank - 2];
    qt_dims[rank - 2] = q_dims[rank - 3];

    std::vector<int> axis = {0, 2, 1, 3};
    synTransposeParams trans_params;
    for (size_t i = 0; i < axis.size(); i++) {
      trans_params.permutation[i] =
          static_cast<TransposePermutationDim>(axis[i]);
    }
    trans_params.tensorDim = rank;

    std::vector<synTensor> attn_out_transpose;
    auto attn_t =
        createTensor(qt_dims.size(), inputs[0].type, qt_dims, false, "attn_t");
    attn_out_transpose.push_back(attn_t);

    status = synNodeCreate(graphHandle_,
                           attn_outputs.data(),
                           attn_out_transpose.data(),
                           1,
                           1,
                           &trans_params,
                           sizeof(trans_params),
                           guid_trans.c_str(),
                           name_trans.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate () failed = ",
             status);

    std::vector<int64_t> attn_reshape;
    attn_reshape.push_back(qt_dims[0]);
    attn_reshape.push_back(qt_dims[1]);
    attn_reshape.push_back(qt_dims[2] * qt_dims[3]);

    std::vector<synTensor> attn_out_reshape;
    auto attn_r = createTensor(
        attn_reshape.size(), inputs[0].type, attn_reshape, false, "attn_r");
    attn_out_reshape.push_back(attn_r);

    status = synNodeCreate(graphHandle_,
                           attn_out_transpose.data(),
                           attn_out_reshape.data(),
                           1,
                           1,
                           nullptr,
                           0,
                           guid_reshape.c_str(),
                           name_reshape.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate () failed = ",
             status);

    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(attn_r);
    mul_inputs.push_back(createTensor(inputs[4].dims.size(),
                                      inputs[4].type,
                                      inputs[4].dims,
                                      true,
                                      inputs[4].name));
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));
    synGEMMParams gemm_params;
    gemm_params.transpose_a = false;
    gemm_params.transpose_b = false;
    status = synNodeCreate(graphHandle_,
                           mul_inputs.data(),
                           mul_outputs.data(),
                           2,
                           1,
                           &gemm_params,
                           sizeof(gemm_params),
                           guid_gemm.c_str(),
                           name_gemm.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate () failed = ",
             status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedSdpaProjKernel(const Context& dev_ctx,
                         const phi::DenseTensor& query_states,
                         const phi::DenseTensor& key_states,
                         const phi::DenseTensor& value_states,
                         const phi::DenseTensor& attn_mask,
                         const phi::DenseTensor& linear_weights,
                         phi::DenseTensor* out_linear,
                         const phi::Scalar& scaling_factor) {
  ConvertTensors ct;
  ct.Add(query_states);
  ct.Add(key_states);
  std::vector<DIMS> in_out_dims = ct.GetDims();
  ct.Add(value_states);
  ct.Add(attn_mask);
  ct.Add(linear_weights);
  ct.Add(out_linear, false);
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(), out_dims.begin(), out_dims.end());

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("fused_sdpa_proj_", in_out_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    ns_Sdpa::ParamsV2 params;
    memset(reinterpret_cast<void*>(&params), 0x00, sizeof(ns_Sdpa::ParamsV2));
    params.scale = scaling_factor.to<float>();
    params.is_causal = false;  // true;
    params.dropout.ratio = 0.0;
    params.dropout.disableMaskOut = false;
    params.is_inference = true;
    params.softmax_mode = SDPA_DEFAULT_SOFTMAX;

    FusedSdpaProj op(op_info.datatype_);
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

template <typename Context>
void CallFusedSdpaProjKernel(const Context& dev_ctx,
                             const phi::DenseTensor& query_states,
                             const phi::DenseTensor& key_states,
                             const phi::DenseTensor& value_states,
                             const phi::DenseTensor& attn_mask,
                             const phi::DenseTensor& linear_weights,
                             phi::DenseTensor* out_linear,
                             const phi::Scalar& scaling_factor) {
  if (query_states.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedSdpaProjKernel<phi::dtype::float16>(dev_ctx,
                                                            query_states,
                                                            key_states,
                                                            value_states,
                                                            attn_mask,
                                                            linear_weights,
                                                            out_linear,
                                                            scaling_factor);
  } else if (query_states.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedSdpaProjKernel<phi::dtype::bfloat16>(dev_ctx,
                                                             query_states,
                                                             key_states,
                                                             value_states,
                                                             attn_mask,
                                                             linear_weights,
                                                             out_linear,
                                                             scaling_factor);
  } else {
    throw std::runtime_error("Unsupported data type for FusedSdpaProjKernel");
  }
}

std::vector<paddle::Tensor> FusedSdpaProj(const paddle::Tensor& query_states,
                                          const paddle::Tensor& key_states,
                                          const paddle::Tensor& value_states,
                                          const paddle::Tensor& attn_mask,
                                          const paddle::Tensor& linear_weights,
                                          float scaling_factor) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          query_states.place()));
  auto query_states_tensor =
      static_cast<const phi::DenseTensor*>(query_states.impl().get());
  auto key_states_tensor =
      static_cast<const phi::DenseTensor*>(key_states.impl().get());
  auto value_states_tensor =
      static_cast<const phi::DenseTensor*>(value_states.impl().get());
  auto attn_mask_tensor =
      static_cast<const phi::DenseTensor*>(attn_mask.impl().get());
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());

  // allocate memory on device.
  int64_t bsz = query_states.dims()[0];
  int64_t num_head = query_states.dims()[1];
  int64_t seq_len = query_states.dims()[2];
  int64_t head_dim = query_states.dims()[3];
  int hidden_size = num_head * head_dim;

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({bsz, seq_len, hidden_size}));
  dev_ctx->Alloc(out_linear.get(), query_states_tensor->dtype());

  CallFusedSdpaProjKernel(*dev_ctx,
                          *query_states_tensor,
                          *key_states_tensor,
                          *value_states_tensor,
                          *attn_mask_tensor,
                          *linear_weights_tensor,
                          out_linear.get(),
                          phi::Scalar(scaling_factor));
  return {paddle::Tensor(out_linear)};
}

std::vector<std::vector<int64_t>> FusedSdpaProjShape(
    const std::vector<int64_t>& query_states_shape,
    const std::vector<int64_t>& key_states_shape,
    const std::vector<int64_t>& value_states_shape,
    const std::vector<int64_t>& attn_mask_shape,
    const std::vector<int64_t>& linear_weights_shape) {
  int64_t bsz = query_states_shape[0];
  int64_t num_head = query_states_shape[1];
  int64_t seq_len = query_states_shape[2];
  int64_t head_dim = query_states_shape[3];
  int hidden_size = num_head * head_dim;
  return {{bsz, seq_len, hidden_size}};
}

std::vector<paddle::DataType> FusedSdpaProjDtype(
    const paddle::DataType& query_states_dtype,
    const paddle::DataType& key_states_dtype,
    const paddle::DataType& value_states_dtype,
    const paddle::DataType& attn_mask_dtype,
    const paddle::DataType& linear_weights_dtype) {
  return {query_states_dtype};
}

PD_BUILD_OP(fused_sdpa_proj)
    .Inputs({"query_states",
             "key_states",
             "value_states",
             "attn_mask",
             "linear_weights"})
    .Outputs({"out_linear"})
    .Attrs({"scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(FusedSdpaProj))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedSdpaProjShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedSdpaProjDtype));
