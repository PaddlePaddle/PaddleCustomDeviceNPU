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
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void TransposeKernel(const Context &dev_ctx,
                     const phi::DenseTensor &x,
                     const std::vector<int> &axis,
                     phi::DenseTensor *out);

class FSDPA : public HpuOperator {
 public:
  explicit FSDPA(std::string guid_prefix, synDataType dtype)
      : HpuOperator(guid_prefix), dtype_(dtype) {}
  void AddNode(ConvertTensors &ct, ns_Sdpa::ParamsV2 params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<int64_t> q_dims = std::vector<int64_t>(inputs[0].dims);
    std::vector<int64_t> qt_dims(q_dims.cbegin(), q_dims.cend());
    std::vector<int64_t> kv_dims = std::vector<int64_t>(inputs[1].dims);
    std::vector<int64_t> kvt_dims(kv_dims.cbegin(), kv_dims.cend());

    int rank = q_dims.size();

    std::vector<int> axis = {0, 2, 1, 3};
    synTransposeParams trans_params;
    for (size_t i = 0; i < axis.size(); i++) {
      trans_params.permutation[i] =
          static_cast<TransposePermutationDim>(axis[i]);
    }
    trans_params.tensorDim = rank;

    qt_dims[rank - 3] = q_dims[rank - 2];
    qt_dims[rank - 2] = q_dims[rank - 3];
    kvt_dims[rank - 3] = kv_dims[rank - 2];
    kvt_dims[rank - 2] = kv_dims[rank - 3];

    synTensor q_transpose_inputs[1] = {createTensor(inputs[0].dims.size(),
                                                    inputs[0].type,
                                                    inputs[0].dims,
                                                    true,
                                                    inputs[0].name)};

    synTensor q_transpose_outputs[1] = {createTensor(
        inputs[0].dims.size(), inputs[0].type, qt_dims, false, "q_t")};

    synTensor k_transpose_inputs[1] = {createTensor(inputs[1].dims.size(),
                                                    inputs[1].type,
                                                    inputs[1].dims,
                                                    true,
                                                    inputs[1].name)};

    synTensor k_transpose_outputs[1] = {createTensor(
        inputs[1].dims.size(), inputs[1].type, kvt_dims, false, "k_t")};

    synTensor v_transpose_inputs[1] = {createTensor(inputs[2].dims.size(),
                                                    inputs[2].type,
                                                    inputs[2].dims,
                                                    true,
                                                    inputs[2].name)};

    synTensor v_transpose_outputs[1] = {createTensor(
        inputs[2].dims.size(), inputs[2].type, kvt_dims, false, "v_t")};

    std::string trans = "transpose";
    if (dtype_ == syn_type_fp16) {
      trans = trans + "_f16";
    } else if (dtype_ == syn_type_bf16) {
      trans = trans + "_bf16";
    } else if (dtype_ == syn_type_single) {
      trans = trans + "_f32";
    }

    synStatus status = synNodeCreate(graphHandle_,
                                     q_transpose_inputs,
                                     q_transpose_outputs,
                                     1,
                                     1,
                                     &trans_params,
                                     sizeof(trans_params),
                                     trans.c_str(),
                                     "q_transpose",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FSDPA q_transpose synNodeCreate () failed = ",
             status);

    status = synNodeCreate(graphHandle_,
                           k_transpose_inputs,
                           k_transpose_outputs,
                           1,
                           1,
                           &trans_params,
                           sizeof(trans_params),
                           trans.c_str(),
                           "k_transpose",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FSDPA k_transpose synNodeCreate () failed = ",
             status);

    status = synNodeCreate(graphHandle_,
                           v_transpose_inputs,
                           v_transpose_outputs,
                           1,
                           1,
                           &trans_params,
                           sizeof(trans_params),
                           trans.c_str(),
                           "v_transpose",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FSDPA v_transpose synNodeCreate () failed = ",
             status);

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(q_transpose_outputs[0]);
    syn_inputs.push_back(k_transpose_outputs[0]);
    syn_inputs.push_back(v_transpose_outputs[0]);
    for (size_t i = 3; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                        inputs[i].type,
                                        inputs[i].dims,
                                        true,
                                        inputs[i].name));
    }

    std::vector<synTensor> syn_outputs;

    synTensor attn_outputs[1] = {createTensor(
        inputs[0].dims.size(), inputs[0].type, qt_dims, false, "attn_t")};
    syn_outputs.push_back(attn_outputs[0]);

    if (!params.is_inference) {
      for (size_t i = 1; i < outputs.size(); i++) {
        syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                           outputs[i].type,
                                           outputs[i].dims,
                                           true,
                                           outputs[i].name));
      }
    }

    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           syn_outputs.data(),
                           syn_inputs.size(),
                           syn_outputs.size(),
                           &params,
                           sizeof(params),
                           guid_.c_str(),
                           "FSDPA",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FSDPA sdpa_recomp_fwd synNodeCreate () failed = ",
             status);

    synTensor attn_transpose_outputs[1] = {createTensor(outputs[0].dims.size(),
                                                        outputs[0].type,
                                                        outputs[0].dims,
                                                        true,
                                                        outputs[0].name)};

    status = synNodeCreate(graphHandle_,
                           attn_outputs,
                           attn_transpose_outputs,
                           1,
                           1,
                           &trans_params,
                           sizeof(trans_params),
                           trans.c_str(),
                           "attn_transpose",
                           nullptr,
                           nullptr);

    PD_CHECK(status == synSuccess,
             "[RUNTIME] FSDPA attn_transpose synNodeCreate () failed = ",
             status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedDotProductAttentionKernel(
    const Context &dev_ctx,
    const phi::DenseTensor &q,
    const phi::DenseTensor &k,
    const phi::DenseTensor &v,
    const paddle::optional<phi::DenseTensor> &attention_mask,
    const paddle::optional<phi::DenseTensor> &cu_seqlen_q,
    const paddle::optional<phi::DenseTensor> &cu_seqlen_kv,
    float scaling_factor,
    float dropout_probability,
    bool is_training,
    const std::string &mask_type_str,
    const std::string &bias_type_str,
    phi::DenseTensor *out,
    phi::DenseTensor *softmax_out,
    phi::DenseTensor *rng_state) {
  dev_ctx.template Alloc<T>(out);
  if (is_training) {
    dev_ctx.template Alloc<T>(softmax_out);
  }

  ConvertTensors ct;
  ct.Add(q);
  ct.Add(k);
  ct.Add(v);
  if (attention_mask.get_ptr()) {
    ct.Add(attention_mask.get_ptr());
  }
  ct.Add(out, false);
  if (is_training) {
    ct.Add(softmax_out, false);
  }
  std::vector<DIMS> in_out_dims = ct.GetDims();
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(), out_dims.begin(), out_dims.end());

  ns_Sdpa::ParamsV2 params;
  memset(reinterpret_cast<void *>(&params), 0x00, sizeof(ns_Sdpa::ParamsV2));
  params.scale = scaling_factor;
  params.is_causal = (mask_type_str == "causal");
  params.dropout.ratio = dropout_probability;
  params.dropout.disableMaskOut = false;
  params.is_inference = !is_training;
  params.softmax_mode = SDPA_DEFAULT_SOFTMAX;

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_Sdpa::ParamsV2>(
      "sdpa_recomp_fwd", in_out_dims, &params);

  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FSDPA op(op_info.guid_, op_info.datatype_);

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

PD_REGISTER_PLUGIN_KERNEL(fused_dot_product_attention,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::FusedDotProductAttentionKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
