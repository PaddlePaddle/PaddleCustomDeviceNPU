// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

class FSDPA : public HpuOperator {
 public:
  explicit FSDPA(std::string guid_prefix)
      : HpuOperator(guid_prefix) {}
  void AddNode(ConvertTensors& ct,
               ns_Sdpa::ParamsV2 params) {
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
    for (size_t i = 1; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         false,
                                         outputs[i].name));
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
void FusedDotProductAttentionKernel(const Context& dev_ctx,
                     const phi::DenseTensor &q,
                     const phi::DenseTensor &k,
                     const phi::DenseTensor &v,
                     const phi::DenseTensor &mask,  //const paddle::optional<phi::DenseTensor> &attention_mask,
                                                    //const paddle::optional<phi::DenseTensor> &cu_seqlen_q,
                                                    //const paddle::optional<phi::DenseTensor> &cu_seqlen_kv,
                     float scaling_factor,
                     float dropout_probability,
                     bool is_training,
                     bool is_causal_masking,
                                                    //const std::string &mask_type_str,
                                                    //const std::string &bias_type_str,
                     phi::DenseTensor *out,
                     phi::DenseTensor *softmax_out,
                     phi::DenseTensor *rng_state) {
  dev_ctx.template Alloc<T>(out);
  
  std::vector<int64_t> q_dims = phi::vectorize<int64_t>(q.dims());
  std::vector<int64_t> k_dims = phi::vectorize<int64_t>(k.dims());
  std::vector<int64_t> v_dims = phi::vectorize<int64_t>(v.dims());
  std::vector<int64_t> output_dims = phi::vectorize<int64_t>(out->dims());
  std::vector<int64_t> softmax_out_dims = phi::vectorize<int64_t>(softmax_out->dims());
  std::vector<int64_t> transposed_shape;
  transposed_shape.push_back(k_dims[0]);
  transposed_shape.push_back(k_dims[1]);
  transposed_shape.push_back(softmax_out_dims[1]);
  transposed_shape.push_back(k_dims[2]);
  softmax_out->Resize(phi::make_ddim(transposed_shape));
  dev_ctx.template Alloc<T>(softmax_out);

  ConvertTensors ct;
  ct.Add(q);
  ct.Add(k);
  ct.Add(v);
  ct.Add(mask);
  /*
  if (attention_mask.get_ptr()) {
    ct.Add(attention_mask.get_ptr());
  }
  */
  ct.Add(out, false);
  ct.Add(softmax_out, false);
  std::vector<DIMS> in_out_dims = ct.GetDims();
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(), out_dims.begin(), out_dims.end());

  ns_Sdpa::ParamsV2 params;
  memset( (void *)&params, 0x00, sizeof(ns_Sdpa::ParamsV2 ));
  params.scale = scaling_factor;
  params.is_causal = is_causal_masking;
  //params.is_causal = (mask_type_str == "causal");
  params.dropout.ratio = dropout_probability;
  params.dropout.disableMaskOut = false;
  params.is_inference = !is_training;
  params.softmax_mode = SDPA_DEFAULT_SOFTMAX;
  
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_Sdpa::ParamsV2>(
      "sdpa_fwd", in_out_dims, &params);
  
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
