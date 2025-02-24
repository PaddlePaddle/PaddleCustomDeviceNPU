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
#include <chrono>
#include <random>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

class TopP : public HpuOperator {
 public:
  TopP() : HpuOperator("top_p") {}

  synTensor AddAddNode(synTensor x,
                       synTensor y,
                       synDataType x_dtype,
                       std::vector<int64_t> x_shape) {
    synTensor syn_inputs[2] = {x, y};
    synTensor syn_outputs[1];
    syn_outputs[0] = {
        createTensor(x_shape.size(), x_dtype, x_shape, false, "to_remove_s4")};

    std::string node_guid = "add_fwd_" + SynDataTypeToStr(x_dtype);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     2,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::Reshape synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }

  synTensor AddCastNode(synTensor x,
                        synDataType src_dtype,
                        std::vector<int64_t> dst_shape,
                        synDataType dst_dtype,
                        std::string dst_name,
                        bool create_dst_section = false) {
    synTensor syn_inputs[1] = {x};
    synSectionHandle section_shared = nullptr;
    if (create_dst_section) {
      section_shared = createSection();
    }

    synTensor syn_outputs[1] = {createTensor(dst_shape.size(),
                                             dst_dtype,
                                             dst_shape,
                                             false,
                                             dst_name.c_str(),
                                             section_shared)};
    std::string node_guid = "cast_" + SynDataTypeToStr(src_dtype) + "_to_" +
                            SynDataTypeToStr(dst_dtype);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::Cast0 synNodeCreate() failed = ",
             status);

    return syn_outputs[0];
  }

  void AddCastNode(synTensor x, synDataType src_dtype, ConvertTensors* ct) {
    synTensor syn_inputs[1] = {x};

    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::Cast1 input ct is a nullptr");
    auto outputs = ct->GetTensors(false);
    synTensor syn_outputs[1] = {createTensor(outputs[1].dims.size(),
                                             outputs[1].type,
                                             outputs[1].dims,
                                             true,
                                             outputs[1].name)};
    std::string node_guid = "cast_" + SynDataTypeToStr(src_dtype) + "_to_" +
                            SynDataTypeToStr(outputs[1].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::Cast1 synNodeCreate() failed = ",
             status);
  }

  synTensor AddCumNode(synTensor x, ConvertTensors* ct) {
    ns_CumSumKernel::Params params{0, 0, 0};

    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::CumSum input ct is a nullptr");
    auto inputs = ct->GetTensors();
    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(x);

    auto outputs = ct->GetTensors(false);
    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(inputs[0].dims.size(),
                                       inputs[0].type,
                                       inputs[0].dims,
                                       false,
                                       "cumsum_probs"));

    std::string node_guid = "cumsum_fwd_" + SynDataTypeToStr(outputs[0].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     "cumsum",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::CumSum synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }

  synTensor AddFullNode(synDataType dtype,
                        std::vector<int64_t> shape,
                        std::string name,
                        ns_ConstantKernel::Params params) {
    synTensor syn_outputs[1] = {
        createTensor(shape.size(), dtype, shape, false, name)};
    std::string node_guid = "constant_" + SynDataTypeToStr(dtype);
    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     syn_outputs,
                                     0,
                                     1,
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     "constant",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::Full synNodeCreate() failed =  ",
             status);
    return syn_outputs[0];
  }

  synTensor AddGatherElementsNode(synTensor x,
                                  synDataType x_dtype,
                                  std::vector<int64_t> shape,
                                  synTensor index) {
    synTensor syn_inputs[2] = {x, index};
    synTensor syn_outputs[1] = {
        createTensor(shape.size(), x_dtype, shape, false, "probs_restored")};

    std::string node_guid = "gather_elements_fwd_" + SynDataTypeToStr(x_dtype);
    ns_GatherElementsKernel::Params params;
    params.axis = 0;
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     2,
                                     1,
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::GatherElements synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }

  synTensor AddGreaterNode(synTensor x,
                           ConvertTensors* ct,
                           std::string out_name) {
    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::Greater input ct is a nullptr");
    auto inputs = ct->GetTensors();
    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(x);
    syn_inputs.push_back(createTensor(inputs[1].dims.size(),
                                      inputs[1].type,
                                      inputs[1].dims,
                                      true,
                                      inputs[1].name));

    auto outputs = ct->GetTensors(false);
    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(inputs[0].dims.size(),
                                       syn_type_int8,
                                       inputs[0].dims,
                                       false,
                                       out_name.c_str()));

    std::string node_guid = "greater_fwd_" + SynDataTypeToStr(inputs[1].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     "greater",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::Greate synNodeCreate () failed =",
             status);
    return syn_outputs[0];
  }

  synTensor AddIndexFillNode(synTensor x,
                             synTensor index_features,
                             synTensor update_features,
                             ConvertTensors* ct) {
    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::IndexFill input ct is a nullptr");
    auto inputs = ct->GetTensors();
    synTensor syn_inputs[3] = {x, index_features, update_features};
    synTensor syn_outputs[1] = {createTensor(inputs[0].dims.size(),
                                             syn_type_int32,
                                             inputs[0].dims,
                                             false,
                                             "to_remove_s3")};

    ns_IndexCopy::Params params{};
    params.axis = 1;
    std::string node_guid =
        "index_copy_fwd_" + SynDataTypeToStr(syn_type_int32);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     3,
                                     1,
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     "IndexFill",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::IndexFill synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }

  synTensor AddMultiNomialNode(synTensor x,
                               synDataType out_dtype,
                               int seed,
                               std::string out_name,
                               ConvertTensors* ct) {
    synTensor syn_inputs[1] = {x};

    PD_CHECK(ct != nullptr,
             "[RUNTIME] TopP::MultiNomial input ct is a nullptr");
    auto outputs = ct->GetTensors(false);
    synTensor syn_outputs[1] = {createTensor(outputs[1].dims.size(),
                                             out_dtype,
                                             outputs[1].dims,
                                             false,
                                             out_name.c_str())};

    ns_RandomMultinomial::ParamsV2 params;
    params.num_samples = 1;
    params.replacement = false;
    params.seed = seed;

    std::string node_guid =
        "random_multinomial_fwd_" + SynDataTypeToStr(outputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     "multinomial",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::MultiNomial synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }

  synTensor AddReshapeNode(synTensor x,
                           synDataType dtype,
                           std::vector<int64_t> out_shape,
                           std::string out_name) {
    synTensor syn_inputs[1] = {x};
    synTensor syn_outputs[1] = {createTensor(
        out_shape.size(), dtype, out_shape, false, out_name.c_str())};

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     "reshape",
                                     "reshape",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::reshape synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }

  std::vector<synTensor> AddTopKNode(ConvertTensors* ct,
                                     ns_TopkNodeV2::ParamsV4 params) {
    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::AddTopK0 input ct is a nullptr");
    auto inputs = ct->GetTensors();
    auto outputs = ct->GetTensors(false);

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                      inputs[0].type,
                                      inputs[0].dims,
                                      true,
                                      inputs[0].name));

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(inputs[0].dims.size(),
                                       inputs[0].type,
                                       inputs[0].dims,
                                       false,
                                       "sorted_probs"));
    syn_outputs.push_back(createTensor(inputs[0].dims.size(),
                                       syn_type_int32,
                                       inputs[0].dims,
                                       false,
                                       "sorted_indices"));

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     "topk",
                                     "topk",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::TopK0 synNodeCreate () failed = ",
             status);
    return syn_outputs;
  }

  std::vector<synTensor> AddTopKNode(synTensor x,
                                     synDataType src_dtype,
                                     std::vector<int64_t> dst_shape,
                                     ns_TopkNodeV2::ParamsV4 params,
                                     ConvertTensors* ct) {
    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::AddTopK1 input ct is a nullptr");
    auto inputs = ct->GetTensors();
    auto outputs = ct->GetTensors(false);

    synTensor syn_inputs[1] = {x};
    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(inputs[0].dims.size(),
                                       inputs[0].type,
                                       inputs[0].dims,
                                       false,
                                       "reverse_values"));
    syn_outputs.push_back(createTensor(inputs[0].dims.size(),
                                       syn_type_int32,
                                       inputs[0].dims,
                                       false,
                                       "reverse_indices"));

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs.data(),
                                     1,
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     "topk",
                                     "topk",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::TopK1 synNodeCreate () failed = ",
             status);
    return syn_outputs;
  }

  synTensor AddWhereNode(synTensor condition,
                         synTensor x,
                         synTensor y,
                         synDataType out_dtype,
                         std::vector<int64_t> out_shape,
                         std::string out_name,
                         ConvertTensors* ct) {
    PD_CHECK(ct != nullptr, "[RUNTIME] TopP::Where input ct is a nullptr");
    auto inputs = ct->GetTensors();
    synTensor syn_inputs[3] = {condition, x, y};
    synTensor syn_outputs[1] = {createTensor(
        out_shape.size(), out_dtype, out_shape, false, out_name.c_str())};

    std::string node_guid = "where_fwd_" + SynDataTypeToStr(out_dtype);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     3,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     "Where",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TopP::Where synNodeCreate () failed = ",
             status);
    return syn_outputs[0];
  }
};
}  // namespace custom_kernel

namespace custom_kernel {
template <typename T, typename Context>
void TopPSamplingKernel_hpu(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& ps,
                            const paddle::optional<phi::DenseTensor>& threshold,
                            const paddle::optional<phi::DenseTensor>& topp_seed,
                            int seed,
                            int k,
                            const std::string& mode,
                            phi::DenseTensor* out,
                            phi::DenseTensor* ids,
                            phi::DenseTensor* topk_scores,
                            phi::DenseTensor* topk_ids) {
  auto x_dims = phi::vectorize<int64_t>(x.dims());
  int length = x_dims[1];

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(ids);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(ps);
  ct.Add(*out, false);
  ct.Add(*ids, false);

  ns_TopkNodeV2::ParamsV4 params{};
  params.bsw = length;
  params.axis = 0;
  params.bottomK = false;
  params.isVcData = false;
  params.isStable = false;

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_TopkNodeV2::ParamsV4>(
      "ToppKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    TopP op;
    auto sorted_probs_and_idx = op.AddTopKNode(&ct, params);

    auto cumsum_probs = op.AddCumNode(sorted_probs_and_idx[0], &ct);

    auto to_remove_i8 = op.AddGreaterNode(cumsum_probs, &ct, "to_remove_i8");

    auto to_remove_i32 = op.AddCastNode(to_remove_i8,
                                        syn_type_int8,
                                        phi::vectorize(x.dims()),
                                        syn_type_int32,
                                        "to_remove_i32");

    std::vector<int64_t> index_features_shape;
    ns_ConstantKernel::Params constant_params;
    index_features_shape.push_back(1);
    constant_params.constant.i = 0;
    auto index_features = op.AddFullNode(syn_type_int32,
                                         index_features_shape,
                                         "index_features",
                                         constant_params);

    std::vector<int64_t> update_features_shape;
    update_features_shape.push_back(x_dims[0]);
    update_features_shape.push_back(1);
    constant_params.constant.i = 0;
    auto update_features = op.AddFullNode(syn_type_int32,
                                          update_features_shape,
                                          "update_features",
                                          constant_params);

    // Reserve column 0
    auto to_remove_s3 = op.AddIndexFillNode(
        to_remove_i32, index_features, update_features, &ct);

    constant_params.constant.f = 0.0;
    auto zero_probs =
        op.AddFullNode(syn_type_float, x_dims, "zero_probs", constant_params);

    auto condition_i8 = op.AddCastNode(
        to_remove_s3, syn_type_int32, x_dims, syn_type_int8, "condition_i8");

    auto probs_filtered = op.AddWhereNode(condition_i8,
                                          zero_probs,
                                          sorted_probs_and_idx[0],
                                          syn_type_float,
                                          x_dims,
                                          "probs_filtered",
                                          &ct);

    params.bottomK = true;
    auto reverse_indices = op.AddTopKNode(
        sorted_probs_and_idx[1], syn_type_int32, x_dims, params, &ct);

    // restore filtered probs into original order
    auto probs_restored = op.AddGatherElementsNode(
        probs_filtered, syn_type_float, x_dims, reverse_indices[1]);

    std::random_device rd;
    auto time_seed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int32_t rd_seed = static_cast<int32_t>(rd() ^ time_seed);
    auto token_i32 = op.AddMultiNomialNode(
        probs_restored, syn_type_int32, rd_seed, "token_i32", &ct);

    op.AddCastNode(token_i32, syn_type_int32, &ct);

    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(top_p_sampling,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TopPSamplingKernel_hpu,
                          float) {}
