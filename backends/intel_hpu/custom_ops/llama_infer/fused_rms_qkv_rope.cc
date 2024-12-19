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

struct FusedRmsQkvRopeParams {
  ns_LayerNormKernel::Params rmsnorm_params;

  int head_dim;
  int num_head;
  int kv_num_head;
};

class FusedRmsQkvRope : public HpuOperator {
 public:
  explicit FusedRmsQkvRope(synDataType dtype)
      : HpuOperator("fused_rms_qkv_rope_fwd_", false), dtype_(dtype) {}

  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               FusedRmsQkvRopeParams& params) {
    synStatus status = synFail;

    std::string name_reshape = guid_ + "reshape";
    std::string name_rmsnorm = guid_ + "rmsnorm";
    std::string name_rope = guid_ + "rope";

    std::string guid_reshape = "reshape";
    std::string guid_rmsnorm = "rms_norm_ex_fwd_";
    std::string guid_rope = "rotary_pos_embedding_fwd_";

    if (dtype_ == syn_type_fp16) {
      guid_rmsnorm = guid_rmsnorm + "f16";
      guid_rope = guid_rope + "f16";
    } else if (dtype_ == syn_type_bf16) {
      guid_rmsnorm = guid_rmsnorm + "bf16";
      guid_rope = guid_rope + "bf16";
    }

    auto src = createTensor(ins[0].size(), dtype_, ins[0], true, "src");
    auto ln_scales =
        createTensor(ins[1].size(), dtype_, ins[1], true, "ln_scales");

    std::vector<synTensor> rmsnorm_inputs;
    rmsnorm_inputs.push_back(src);
    rmsnorm_inputs.push_back(ln_scales);

    auto tmp_dims = ins[0];
    tmp_dims[2] = 1;
    auto norm_out =
        createTensor(ins[0].size(), dtype_, ins[0], false, "norm_out");
    auto norm_var =
        createTensor(tmp_dims.size(), dtype_, tmp_dims, false, "norm_var");

    std::vector<synTensor> rmsnorm_outputs;
    rmsnorm_outputs.push_back(norm_out);
    rmsnorm_outputs.push_back(norm_var);

    status = synNodeCreate(graphHandle_,
                           rmsnorm_inputs.data(),
                           rmsnorm_outputs.data(),
                           2,
                           2,
                           &params.rmsnorm_params,
                           sizeof(params.rmsnorm_params),
                           guid_rmsnorm.c_str(),
                           name_rmsnorm.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    auto qkv_weights =
        createTensor(ins[2].size(), dtype_, ins[2], true, "qkv_weights");
    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(norm_out);
    mul_inputs.push_back(qkv_weights);

    auto wt_dims = ins[2];
    tmp_dims[2] = wt_dims[0];

    auto qkv_out =
        createTensor(tmp_dims.size(), dtype_, tmp_dims, false, "qkv_out");
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(qkv_out);

    synGEMMParams gemm_params;
    gemm_params.transpose_a = false;
    gemm_params.transpose_b = true;
    std::string guid_gemm = "gemm";
    std::string gemm_name = guid_ + "gemm";
    status = synNodeCreate(graphHandle_,
                           mul_inputs.data(),
                           mul_outputs.data(),
                           2,
                           1,
                           &gemm_params,
                           sizeof(gemm_params),
                           guid_gemm.c_str(),
                           gemm_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    mul_outputs.push_back(qkv_out);

    auto reshape_dims = ins[0];
    reshape_dims[2] = params.num_head + 2 * params.kv_num_head;
    reshape_dims.push_back(params.head_dim);

    std::vector<synTensor> reshape_outputs;
    auto reshape_out = createTensor(
        reshape_dims.size(), dtype_, reshape_dims, false, "reshape_out");
    reshape_outputs.push_back(reshape_out);

    status = synNodeCreate(graphHandle_,
                           mul_outputs.data(),
                           reshape_outputs.data(),
                           1,
                           1,
                           nullptr,
                           0,
                           guid_reshape.c_str(),
                           name_reshape.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    int rank = reshape_dims.size();
    std::vector<int> axis = {0, 2, 1, 3};
    synTransposeParams trans_params;
    for (size_t i = 0; i < axis.size(); i++) {
      trans_params.permutation[i] =
          static_cast<TransposePermutationDim>(axis[i]);
    }
    trans_params.tensorDim = rank;

    std::vector<int64_t> trans_dims(reshape_dims.cbegin(), reshape_dims.cend());
    trans_dims[rank - 3] = reshape_dims[rank - 2];
    trans_dims[rank - 2] = reshape_dims[rank - 3];

    std::vector<synTensor> transpose_outputs;
    auto trans_out =
        createTensor(trans_dims.size(), dtype_, trans_dims, false, "trans_out");
    transpose_outputs.push_back(trans_out);

    std::string guid_trans = "transpose";
    std::string name_trans = guid_ + "transpose";
    status = synNodeCreate(graphHandle_,
                           reshape_outputs.data(),
                           transpose_outputs.data(),
                           1,
                           1,
                           &trans_params,
                           sizeof(trans_params),
                           guid_trans.c_str(),
                           name_trans.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    synSplitParams splitParams;
    splitParams.axis = 2;

    std::vector<synTensor> split_outpus;
    auto q = createTensor(outs[0].size(), dtype_, outs[0], false, "q_states");
    split_outpus.push_back(q);

    auto k = createTensor(outs[1].size(), dtype_, outs[1], false, "k_states");
    split_outpus.push_back(k);

    auto v =
        createTensor(outs[2].size(), dtype_, outs[2], true, "value_states");
    split_outpus.push_back(v);

    std::string split_guid = "split";
    std::string split_name = guid_ + "split";
    status = synNodeCreate(graphHandle_,
                           transpose_outputs.data(),
                           split_outpus.data(),
                           1,
                           split_outpus.size(),
                           &splitParams,
                           sizeof(splitParams),
                           split_guid.c_str(),
                           split_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    std::vector<synTensor> sin_inputs;
    std::vector<synTensor> cos_inputs;
    auto sin = createTensor(ins[3].size(), dtype_, ins[3], true, "sin");
    auto cos = createTensor(ins[4].size(), dtype_, ins[4], true, "cos");
    sin_inputs.push_back(sin);
    cos_inputs.push_back(cos);

    auto sin_cos_dims = ins[3];
    sin_cos_dims[2] = sin_cos_dims[1];
    sin_cos_dims[1] = 1;

    std::vector<synTensor> sin_squeezed;
    std::vector<synTensor> cos_squeezed;
    auto sin_sq = createTensor(
        sin_cos_dims.size(), dtype_, sin_cos_dims, false, "sin_squeezed");
    auto cos_sq = createTensor(
        sin_cos_dims.size(), dtype_, sin_cos_dims, false, "cos_squeezed");
    sin_squeezed.push_back(sin_sq);
    cos_squeezed.push_back(cos_sq);
    status = synNodeCreate(graphHandle_,
                           sin_inputs.data(),
                           sin_squeezed.data(),
                           1,
                           1,
                           nullptr,
                           0,
                           guid_reshape.c_str(),
                           name_reshape.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);
    status = synNodeCreate(graphHandle_,
                           cos_inputs.data(),
                           cos_squeezed.data(),
                           1,
                           1,
                           nullptr,
                           0,
                           guid_reshape.c_str(),
                           name_reshape.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    std::vector<synTensor> inputs_q;
    std::vector<synTensor> outputs_q;
    inputs_q.push_back(q);
    inputs_q.push_back(sin_sq);
    inputs_q.push_back(cos_sq);
    auto position_ids = createTensor(
        ins[5].size(), syn_type_int32, ins[5], true, "position_ids");
    inputs_q.push_back(position_ids);

    auto q_states =
        createTensor(outs[0].size(), dtype_, outs[0], true, "query_states");
    outputs_q.push_back(q_states);

    ns_RoPESt2::ParamsV2 ropeParams;
    ropeParams.offset = 0;
    ropeParams.mode = ROTARY_POS_EMBEDDING_MODE_BLOCKWISE;

    status = synNodeCreate(graphHandle_,
                           inputs_q.data(),
                           outputs_q.data(),
                           inputs_q.size(),
                           outputs_q.size(),
                           &ropeParams,
                           sizeof(ropeParams),
                           guid_rope.c_str(),
                           name_rope.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);

    std::vector<synTensor> inputs_k;
    std::vector<synTensor> outputs_k;
    inputs_k.push_back(k);
    inputs_k.push_back(sin_sq);
    inputs_k.push_back(cos_sq);
    inputs_k.push_back(position_ids);

    auto k_states =
        createTensor(outs[1].size(), dtype_, outs[1], true, "key_states");
    outputs_k.push_back(k_states);

    status = synNodeCreate(graphHandle_,
                           inputs_k.data(),
                           outputs_k.data(),
                           inputs_k.size(),
                           outputs_k.size(),
                           &ropeParams,
                           sizeof(ropeParams),
                           guid_rope.c_str(),
                           name_rope.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsQkvRopeKernel synNodeCreate () failed = ",
             status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedRmsQkvRopeKernel(const Context& dev_ctx,
                           const phi::DenseTensor& src,
                           const phi::DenseTensor& ln_scales,
                           const phi::DenseTensor& qkv_weights,
                           const phi::DenseTensor& cos,
                           const phi::DenseTensor& sin,
                           const phi::DenseTensor& position_ids,
                           phi::DenseTensor* query_states,
                           phi::DenseTensor* key_states,
                           phi::DenseTensor* value_states,
                           const phi::Scalar& epsilon,
                           const phi::Scalar& head_dim,
                           const phi::Scalar& num_head) {
  std::vector<int64_t> src_dims = phi::vectorize<int64_t>(src.dims());
  std::vector<int64_t> ln_scales_dims =
      phi::vectorize<int64_t>(ln_scales.dims());
  std::vector<int64_t> qkv_weights_dims =
      phi::vectorize<int64_t>(qkv_weights.dims());
  std::vector<int64_t> cos_dims = phi::vectorize<int64_t>(cos.dims());
  std::vector<int64_t> sin_dims = phi::vectorize<int64_t>(sin.dims());
  std::vector<int64_t> position_ids_dims =
      phi::vectorize<int64_t>(position_ids.dims());

  std::vector<int64_t> out_q_dim =
      phi::vectorize<int64_t>(query_states->dims());
  std::vector<int64_t> out_k_dim = phi::vectorize<int64_t>(key_states->dims());
  std::vector<int64_t> out_v_dim =
      phi::vectorize<int64_t>(value_states->dims());

  std::vector<DIMS> inputs = {src_dims,
                              ln_scales_dims,
                              qkv_weights_dims,
                              sin_dims,
                              cos_dims,
                              position_ids_dims};
  std::vector<DIMS> outputs = {out_q_dim, out_k_dim, out_v_dim};

  int head_dim_ = head_dim.to<int>();
  int num_head_ = num_head.to<int>();
  const int64_t bsz = src_dims[0];
  const int64_t seq_len = src_dims[1];
  const int64_t fused_hidden_size = qkv_weights_dims[0];
  const int64_t hidden_size = qkv_weights_dims[1];
  const int kv_num_head =
      (fused_hidden_size - num_head_ * head_dim_) / head_dim_ / 2;
  const int num_groups = num_head_ / kv_num_head;

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(
      "fused_rms_qkv_rope_fwd_", {src_dims}, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedRmsQkvRopeParams params;
    memset(
        reinterpret_cast<void*>(&params), 0x00, sizeof(FusedRmsQkvRopeParams));
    params.rmsnorm_params.epsValid = true;
    params.rmsnorm_params.eps = epsilon.to<float>();
    params.head_dim = head_dim_;
    params.num_head = num_head_;
    params.kv_num_head = kv_num_head;

    FusedRmsQkvRope op(op_info.datatype_);
    op.AddNode(inputs, outputs, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["src"] = reinterpret_cast<uint64_t>(src.data<T>());
  tensors["ln_scales"] = reinterpret_cast<uint64_t>(ln_scales.data<T>());
  tensors["qkv_weights"] = reinterpret_cast<uint64_t>(qkv_weights.data<T>());
  tensors["sin"] = reinterpret_cast<uint64_t>(sin.data<T>());
  tensors["cos"] = reinterpret_cast<uint64_t>(cos.data<T>());
  tensors["position_ids"] =
      reinterpret_cast<uint64_t>(position_ids.data<int64_t>());

  tensors["query_states"] = reinterpret_cast<uint64_t>(query_states->data<T>());
  tensors["key_states"] = reinterpret_cast<uint64_t>(key_states->data<T>());
  tensors["value_states"] = reinterpret_cast<uint64_t>(value_states->data<T>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}
}  // namespace custom_kernel

template <typename Context>
void CallFusedRmsQkvRopeKernel(const Context& dev_ctx,
                               const phi::DenseTensor& src,
                               const phi::DenseTensor& ln_scales,
                               const phi::DenseTensor& qkv_weights,
                               const phi::DenseTensor& cos,
                               const phi::DenseTensor& sin,
                               const phi::DenseTensor& position_ids,
                               phi::DenseTensor* query_states,
                               phi::DenseTensor* key_states,
                               phi::DenseTensor* value_states,
                               const phi::Scalar& epsilon,
                               const phi::Scalar& head_dim,
                               const phi::Scalar& num_head) {
  if (src.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedRmsQkvRopeKernel<phi::dtype::float16>(dev_ctx,
                                                              src,
                                                              ln_scales,
                                                              qkv_weights,
                                                              cos,
                                                              sin,
                                                              position_ids,
                                                              query_states,
                                                              key_states,
                                                              value_states,
                                                              epsilon,
                                                              head_dim,
                                                              num_head);
  } else if (src.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedRmsQkvRopeKernel<phi::dtype::bfloat16>(dev_ctx,
                                                               src,
                                                               ln_scales,
                                                               qkv_weights,
                                                               cos,
                                                               sin,
                                                               position_ids,
                                                               query_states,
                                                               key_states,
                                                               value_states,
                                                               epsilon,
                                                               head_dim,
                                                               num_head);
  } else {
    throw std::runtime_error("Unsupported data type for FusedRmsQkvRopeKernel");
  }
}

std::vector<paddle::Tensor> FusedRmsQkvRope(const paddle::Tensor& src,
                                            const paddle::Tensor& ln_scales,
                                            const paddle::Tensor& qkv_weights,
                                            const paddle::Tensor& cos,
                                            const paddle::Tensor& sin,
                                            const paddle::Tensor& position_ids,
                                            float epsilon,
                                            int head_dim,
                                            int num_head) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(src.place()));
  auto src_tensor = static_cast<const phi::DenseTensor*>(src.impl().get());
  auto ln_scales_tensor =
      static_cast<const phi::DenseTensor*>(ln_scales.impl().get());
  auto qkv_weights_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights.impl().get());
  auto cos_tensor = static_cast<const phi::DenseTensor*>(cos.impl().get());
  auto sin_tensor = static_cast<const phi::DenseTensor*>(sin.impl().get());
  auto position_ids_tensor =
      static_cast<const phi::DenseTensor*>(position_ids.impl().get());

  // allocate memory on device.
  int64_t bsz = src.dims()[0];
  int64_t seq_len = src.dims()[1];
  int64_t fused_hidden_size = qkv_weights.dims()[0];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;

  std::shared_ptr<phi::DenseTensor> query_states =
      std::make_shared<phi::DenseTensor>();
  query_states->Resize(phi::make_ddim({bsz, num_head, seq_len, head_dim}));
  dev_ctx->Alloc(query_states.get(), src_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> key_states =
      std::make_shared<phi::DenseTensor>();
  key_states->Resize(phi::make_ddim({bsz, kv_num_head, seq_len, head_dim}));
  dev_ctx->Alloc(key_states.get(), src_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> value_states =
      std::make_shared<phi::DenseTensor>();
  value_states->Resize(phi::make_ddim({bsz, kv_num_head, seq_len, head_dim}));
  dev_ctx->Alloc(value_states.get(), src_tensor->dtype());

  CallFusedRmsQkvRopeKernel(*dev_ctx,
                            *src_tensor,
                            *ln_scales_tensor,
                            *qkv_weights_tensor,
                            *cos_tensor,
                            *sin_tensor,
                            *position_ids_tensor,
                            query_states.get(),
                            key_states.get(),
                            value_states.get(),
                            phi::Scalar(epsilon),
                            phi::Scalar(head_dim),
                            phi::Scalar(num_head));
  return {paddle::Tensor(query_states),
          paddle::Tensor(key_states),
          paddle::Tensor(value_states)};
}

std::vector<std::vector<int64_t>> FusedRmsQkvRopeShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& ln_scales_shape,
    const std::vector<int64_t>& qkv_weights_shape,
    const std::vector<int64_t>& cos_shape,
    const std::vector<int64_t>& sin_shape,
    const std::vector<int64_t>& position_ids_shape,
    float epsilon,
    int head_dim,
    int num_head) {
  int64_t bsz = src_shape[0];
  int64_t seq_len = src_shape[1];
  int64_t fused_hidden_size = qkv_weights_shape[0];
  int kv_num_head = (fused_hidden_size - num_head * head_dim) / head_dim / 2;
  return {{bsz, num_head, seq_len, head_dim},
          {bsz, kv_num_head, seq_len, head_dim},
          {bsz, kv_num_head, seq_len, head_dim}};
}

std::vector<paddle::DataType> FusedRmsQkvRopeDtype(
    const paddle::DataType& src_dtype,
    const paddle::DataType& ln_scales_dtype,
    const paddle::DataType& qkv_weights_dtype,
    const paddle::DataType& cos_dtype,
    const paddle::DataType& sin_dtype,
    const paddle::DataType& position_ids_dtype) {
  return {src_dtype, src_dtype, src_dtype};
}

PD_BUILD_OP(fused_rms_qkv_rope)
    .Inputs({"src", "ln_scales", "qkv_weights", "cos", "sin", "position_ids"})
    .Outputs({"query_states", "key_states", "value_states"})
    .Attrs({"epsilon: float", "head_dim: int", "num_head: int"})
    .SetKernelFn(PD_KERNEL(FusedRmsQkvRope))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedRmsQkvRopeShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedRmsQkvRopeDtype));
