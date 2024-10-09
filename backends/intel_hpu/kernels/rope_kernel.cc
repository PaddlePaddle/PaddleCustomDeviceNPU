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

class ROPE : public HpuOperator {
 public:
  explicit ROPE(std::string guid_prefix, synDataType dtype)
      : HpuOperator(guid_prefix), dtype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               ns_RoPESt2::ParamsV2 params) {
    std::vector<synTensor> inputs;
    std::vector<synTensor> outputs;

    auto q = createTensor(ins[0].size(), dtype_, ins[0], true, "q");
    inputs.push_back(q);
    auto sin = createTensor(ins[1].size(), dtype_, ins[1], true, "sin");
    inputs.push_back(sin);
    auto cos = createTensor(ins[2].size(), dtype_, ins[2], true, "cos");
    inputs.push_back(cos);
    if (ins.size() == 4) {
      auto position_ids = createTensor(
          ins[3].size(), syn_type_int32, ins[3], true, "position_ids");
      inputs.push_back(position_ids);
    }

    auto out = createTensor(outs[0].size(), dtype_, outs[0], true, "out");
    outputs.push_back(out);

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs.data(),
                                     inputs.size(),
                                     outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "ROPE",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedRopeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& q,
                     const paddle::optional<phi::DenseTensor>& k,
                     const paddle::optional<phi::DenseTensor>& v,
                     const paddle::optional<phi::DenseTensor>& sin,
                     const paddle::optional<phi::DenseTensor>& cos,
                     const paddle::optional<phi::DenseTensor>& position_ids,
                     bool use_neox_rotary_style,
                     bool time_major,
                     float rotary_emb_base,
                     phi::DenseTensor* out_q,
                     phi::DenseTensor* out_k,
                     phi::DenseTensor* out_v) {
  dev_ctx.template Alloc<T>(out_q);

  if (!sin.get_ptr() || !cos.get_ptr() || k || v) {
    PADDLE_THROW(
        "FusedRopeKernel supports (p, sin, cos, [position_ids]) only.");
  }
  std::vector<int64_t> q_dims = phi::vectorize<int64_t>(q.dims());
  std::vector<int64_t> sin_dims =
      phi::vectorize<int64_t>(sin.get_ptr()->dims());
  std::vector<int64_t> cos_dims =
      phi::vectorize<int64_t>(cos.get_ptr()->dims());
  std::vector<int64_t> out_q_dim = phi::vectorize<int64_t>(out_q->dims());

  ns_RoPESt2::ParamsV2 params;
  params.offset = 0;
  params.mode = ROTARY_POS_EMBEDDING_MODE_BLOCKWISE;

  std::vector<DIMS> inputs = {q_dims, sin_dims, cos_dims};

  if (position_ids.get_ptr()) {
    std::vector<int64_t> position_ids_dims =
        phi::vectorize<int64_t>(position_ids.get_ptr()->dims());
    inputs.push_back(position_ids_dims);
  }

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_RoPESt2::ParamsV2>(
      "rotary_pos_embedding_fwd", inputs, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    ROPE op(op_info.guid_, op_info.datatype_);
    op.AddNode(inputs, {out_q_dim}, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["q"] = reinterpret_cast<uint64_t>(q.data<T>());
  tensors["sin"] = reinterpret_cast<uint64_t>(sin.get_ptr()->data<T>());
  tensors["cos"] = reinterpret_cast<uint64_t>(cos.get_ptr()->data<T>());
  if (position_ids.get_ptr()) {
    tensors["position_ids"] =
        reinterpret_cast<uint64_t>(position_ids.get_ptr()->data<int64_t>());
  }
  tensors["out"] = reinterpret_cast<uint64_t>(out_q->data<T>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fused_rotary_position_embedding,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::FusedRopeKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
