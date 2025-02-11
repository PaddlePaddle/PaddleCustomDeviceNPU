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

class FusedSdpaProjV2 : public HpuOperator {
 public:
  explicit FusedSdpaProjV2(synDataType dtype)
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

    std::vector<synTensor> kv_inputs;
    kv_inputs.push_back(createTensor(inputs[1].dims.size(),
                                     inputs[1].type,
                                     inputs[1].dims,
                                     true,
                                     inputs[1].name));
    auto k_v_dims = inputs[1].dims;
    k_v_dims[0] = 1;

    synSliceParamsV2 sliceParams;
    for (uint64_t i = 0; i < k_v_dims.size(); i++) {
      sliceParams.axes[i] = i;
      sliceParams.steps[i] = 1;
      sliceParams.starts[i] = 0;
      sliceParams.ends[i] = k_v_dims[k_v_dims.size() - 1 - i];
    }

    std::string slice_guid = "slice";
    std::string slice_name = guid_ + "slice";
    std::string slice_name_k = slice_name + "_key";

    std::vector<synTensor> k_slice;
    auto k_split =
        createTensor(k_v_dims.size(), dtype_, k_v_dims, false, "k_split");
    k_slice.push_back(k_split);

    status = synNodeCreate(graphHandle_,
                           kv_inputs.data(),
                           k_slice.data(),
                           kv_inputs.size(),
                           k_slice.size(),
                           &sliceParams,
                           sizeof(sliceParams),
                           slice_guid.c_str(),
                           slice_name_k.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate (slice/k) failed = ",
        status);

    std::vector<synTensor> v_slice;
    auto v_split =
        createTensor(k_v_dims.size(), dtype_, k_v_dims, false, "v_split");
    v_slice.push_back(v_split);
    sliceParams.starts[k_v_dims.size() - 1] = 1;
    sliceParams.ends[k_v_dims.size() - 1] = 2;
    std::string slice_name_v = slice_name + "_value";
    status = synNodeCreate(graphHandle_,
                           kv_inputs.data(),
                           v_slice.data(),
                           kv_inputs.size(),
                           v_slice.size(),
                           &sliceParams,
                           sizeof(sliceParams),
                           slice_guid.c_str(),
                           slice_name_v.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate (slice/v) failed = ",
        status);

    synSqueezeParams squeezeParams;
    squeezeParams.axis = 4;
    std::string squeeze_guid = "squeeze";
    std::string squeeze_name = guid_ + "squeeze";

    k_v_dims.erase(k_v_dims.begin());

    std::vector<synTensor> key_squeezed;
    auto key_states =
        createTensor(k_v_dims.size(), dtype_, k_v_dims, false, "key_states");

    key_squeezed.push_back(key_states);
    std::string squeeze_name_key = squeeze_name + "_key";
    status = synNodeCreate(graphHandle_,
                           k_slice.data(),
                           key_squeezed.data(),
                           1,
                           1,
                           &squeezeParams,
                           sizeof(squeezeParams),
                           squeeze_guid.c_str(),
                           squeeze_name_key.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate (squeeze/key) failed = ",
        status);

    std::vector<synTensor> value_squeezed;
    auto value_states =
        createTensor(k_v_dims.size(), dtype_, k_v_dims, false, "value_states");
    value_squeezed.push_back(value_states);
    std::string squeeze_name_value = squeeze_name + "_value";
    status = synNodeCreate(graphHandle_,
                           v_slice.data(),
                           value_squeezed.data(),
                           1,
                           1,
                           &squeezeParams,
                           sizeof(squeezeParams),
                           squeeze_guid.c_str(),
                           squeeze_name_value.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate (squeeze/value) "
             "failed = ",
             status);

    std::vector<synTensor> attn_inputs;
    attn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                       inputs[0].type,
                                       inputs[0].dims,
                                       true,
                                       inputs[0].name));
    attn_inputs.push_back(key_states);
    attn_inputs.push_back(value_states);
    // params.is_causal = true; ==> input[2] is not used
    // input[2] is in use ==> params.is_causal = false;
    attn_inputs.push_back(createTensor(inputs[2].dims.size(),
                                       inputs[2].type,
                                       inputs[2].dims,
                                       true,
                                       inputs[2].name));
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
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate (sdpa) failed = ",
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
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] FusedSdpaProjKernel synNodeCreate (transpose) failed = ",
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
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate (reshape) failed = ",
             status);

    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(attn_r);
    mul_inputs.push_back(createTensor(inputs[3].dims.size(),
                                      inputs[3].type,
                                      inputs[3].dims,
                                      true,
                                      inputs[3].name));
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
             "[RUNTIME] FusedSdpaProjKernel synNodeCreate (matmul) failed = ",
             status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedSdpaProjKernelV2(const Context& dev_ctx,
                           const phi::DenseTensor& query_states,
                           const phi::DenseTensor& key_value_states,
                           const phi::DenseTensor& attn_mask,
                           const phi::DenseTensor& linear_weights,
                           phi::DenseTensor* out_linear,
                           const phi::Scalar& scaling_factor) {
  ConvertTensors ct;
  ct.Add(query_states);
  ct.Add(key_value_states);
  std::vector<DIMS> in_out_dims = ct.GetDims();
  ct.Add(attn_mask);
  ct.Add(linear_weights);
  ct.Add(out_linear, false);
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(), out_dims.begin(), out_dims.end());

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(
      "fused_sdpa_proj_v2_fwd_", in_out_dims, nullptr);
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

    FusedSdpaProjV2 op(op_info.datatype_);
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
void CallFusedSdpaProjKernelV2(const Context& dev_ctx,
                               const phi::DenseTensor& query_states,
                               const phi::DenseTensor& key_value_states,
                               const phi::DenseTensor& attn_mask,
                               const phi::DenseTensor& linear_weights,
                               phi::DenseTensor* out_linear,
                               const phi::Scalar& scaling_factor) {
  if (query_states.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedSdpaProjKernelV2<phi::dtype::float16>(dev_ctx,
                                                              query_states,
                                                              key_value_states,
                                                              attn_mask,
                                                              linear_weights,
                                                              out_linear,
                                                              scaling_factor);
  } else if (query_states.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedSdpaProjKernelV2<phi::dtype::bfloat16>(dev_ctx,
                                                               query_states,
                                                               key_value_states,
                                                               attn_mask,
                                                               linear_weights,
                                                               out_linear,
                                                               scaling_factor);
  } else {
    throw std::runtime_error("Unsupported data type for FusedSdpaProjKernel");
  }
}

std::vector<paddle::Tensor> FusedSdpaProjV2(
    const paddle::Tensor& query_states,
    const paddle::Tensor& key_value_states,
    const paddle::Tensor& attn_mask,
    const paddle::Tensor& linear_weights,
    float scaling_factor) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          query_states.place()));
  auto query_states_tensor =
      static_cast<const phi::DenseTensor*>(query_states.impl().get());
  auto key_value_states_tensor =
      static_cast<const phi::DenseTensor*>(key_value_states.impl().get());
  auto attn_mask_tensor =
      static_cast<const phi::DenseTensor*>(attn_mask.impl().get());
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());

  // allocate memory on device.
  int64_t bsz = query_states.dims()[0];
  int64_t seq_len = query_states.dims()[2];
  int hidden_size = linear_weights.dims()[1];

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({bsz, seq_len, hidden_size}));
  dev_ctx->Alloc(out_linear.get(), query_states_tensor->dtype());

  CallFusedSdpaProjKernelV2(*dev_ctx,
                            *query_states_tensor,
                            *key_value_states_tensor,
                            *attn_mask_tensor,
                            *linear_weights_tensor,
                            out_linear.get(),
                            phi::Scalar(scaling_factor));
  return {paddle::Tensor(out_linear)};
}

std::vector<std::vector<int64_t>> FusedSdpaProjV2Shape(
    const std::vector<int64_t>& query_states_shape,
    const std::vector<int64_t>& key_value_states_shape,
    const std::vector<int64_t>& attn_mask_shape,
    const std::vector<int64_t>& linear_weights_shape) {
  int64_t bsz = query_states_shape[0];
  int64_t seq_len = query_states_shape[2];
  int hidden_size = linear_weights_shape[1];
  return {{bsz, seq_len, hidden_size}};
}

std::vector<paddle::DataType> FusedSdpaProjV2Dtype(
    const paddle::DataType& query_states_dtype,
    const paddle::DataType& key_value_states_dtype,
    const paddle::DataType& attn_mask_dtype,
    const paddle::DataType& linear_weights_dtype) {
  return {query_states_dtype};
}

PD_BUILD_OP(fused_sdpa_proj_v2)
    .Inputs({"query_states", "key_value_states", "attn_mask", "linear_weights"})
    .Outputs({"out_linear"})
    .Attrs({"scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(FusedSdpaProjV2))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedSdpaProjV2Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedSdpaProjV2Dtype));
