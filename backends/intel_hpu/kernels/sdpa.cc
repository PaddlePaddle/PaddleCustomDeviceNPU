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
  explicit FSDPA(std::string guid_prefix) : HpuOperator(guid_prefix) {}
  void AddNode(ConvertTensors &ct, ns_Sdpa::ParamsV2 params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                        inputs[i].type,
                                        inputs[i].dims,
                                        true,
                                        inputs[i].name));
    }

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < 1; i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }
    if (!params.is_inference) {
      for (size_t i = 1; i < outputs.size(); i++) {
        syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                           outputs[i].type,
                                           outputs[i].dims,
                                           true,
                                           outputs[i].name));
      }
    }

    synStatus status = synNodeCreate(graphHandle_,
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
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void FusedDotProductAttentionKernel(
    const Context &dev_ctx,
    const phi::DenseTensor &q,
    const phi::DenseTensor &k,
    const phi::DenseTensor &v,
    const phi::DenseTensor &mask,
    // const paddle::optional<phi::DenseTensor> &attention_mask,
    // const paddle::optional<phi::DenseTensor> &cu_seqlen_q,
    // const paddle::optional<phi::DenseTensor> &cu_seqlen_kv,
    float scaling_factor,
    float dropout_probability,
    bool is_training,
    bool is_causal_masking,
    // const std::string &mask_type_str,
    // const std::string &bias_type_str,
    phi::DenseTensor *out,
    phi::DenseTensor *softmax_out,
    phi::DenseTensor *rng_state) {
  std::vector<int> axis = {0, 2, 1, 3};
  phi::DenseTensor qt;
  // auto q_dims = q.dims();
  std::vector<int64_t> q_dims = phi::vectorize<int64_t>(q.dims());
  std::vector<int64_t> qt_dims(q_dims.cbegin(), q_dims.cend());

  int rank = q_dims.size();
  qt_dims[rank - 3] = q_dims[rank - 2];
  qt_dims[rank - 2] = q_dims[rank - 3];

  phi::DenseTensorMeta qt_meta({q.dtype(), phi::make_ddim(qt_dims)});
  qt.set_meta(qt_meta);
  custom_kernel::TransposeKernel<T, Context>(dev_ctx, q, axis, &qt);

  phi::DenseTensor kt;
  phi::DenseTensor vt;
  std::vector<int64_t> kv_dims = phi::vectorize<int64_t>(k.dims());
  std::vector<int64_t> kvt_dims(kv_dims.cbegin(), kv_dims.cend());
  kvt_dims[rank - 3] = kv_dims[rank - 2];
  kvt_dims[rank - 2] = kv_dims[rank - 3];
  phi::DenseTensorMeta kvt_meta({k.dtype(), phi::make_ddim(kvt_dims)});
  kt.set_meta(kvt_meta);
  vt.set_meta(kvt_meta);
  custom_kernel::TransposeKernel<T, Context>(dev_ctx, k, axis, &kt);
  custom_kernel::TransposeKernel<T, Context>(dev_ctx, v, axis, &vt);

  out->Resize(phi::make_ddim(qt_dims));
  dev_ctx.template Alloc<T>(out);
  if (is_training) {
    dev_ctx.template Alloc<T>(softmax_out);
  }

  ConvertTensors ct;
  ct.Add(qt);
  ct.Add(kt);
  ct.Add(vt);
  ct.Add(mask);
  /*
  if (attention_mask.get_ptr()) {
    ct.Add(attention_mask.get_ptr());
  }
  */
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
  params.is_causal = is_causal_masking;
  // params.is_causal = (mask_type_str == "causal");
  params.dropout.ratio = dropout_probability;
  params.dropout.disableMaskOut = false;
  params.is_inference = !is_training;
  params.softmax_mode = SDPA_DEFAULT_SOFTMAX;

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_Sdpa::ParamsV2>(
      "sdpa_recomp_fwd", in_out_dims, &params);

  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FSDPA op(op_info.guid_);

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
