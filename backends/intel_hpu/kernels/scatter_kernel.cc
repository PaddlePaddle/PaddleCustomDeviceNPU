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

class Scatter : public HpuOperator {
 public:
  Scatter() : HpuOperator("scatter_fwd_") {}

  static void BuildRecipeName(std::string& recipe_name,
                              synDataType dtype,
                              bool is_inplace,
                              bool cast,
                              bool overwrite) {
    recipe_name = "ScatterKernel";
    if (is_inplace) {
      recipe_name += "i_";
    }
    if (cast) {
      recipe_name += "c_";
    }
    if (overwrite) {
      recipe_name += "o_";
    }
    recipe_name += SynDataTypeToStr(dtype);
  }

  synTensor AddCastNode(ConvertTensors* ct, bool cast) {
    PD_CHECK(ct != nullptr,
             "[RUNTIME] ScatterKernel AddCastNode() ct is nullptr");

    auto inputs = ct->GetTensors();
    synTensor syn_inputs[1] = {createTensor(inputs[1].dims.size(),
                                            inputs[1].type,
                                            inputs[1].dims,
                                            true,
                                            inputs[1].name)};

    if (!cast) {
      return syn_inputs[0];
    }

    synTensor syn_outputs[1] = {createTensor(inputs[1].dims.size(),
                                             syn_type_int32,
                                             inputs[1].dims,
                                             false,
                                             "index_dst")};

    std::string cast_guid = "cast_i64_to_i32";
    std::string name = guid_ + cast_guid.c_str();

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     cast_guid.c_str(),
                                     name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] ScatterKernel AddCastNode() failed =  ",
             status);

    return syn_outputs[0];
  }

  synTensor AddExpandNode(synTensor expandSrc, ConvertTensors* ct) {
    PD_CHECK(ct != nullptr,
             "[RUNTIME] ScatterKernel AddExpandNode() ct is nullptr");

    auto inputs = ct->GetTensors();
    synTensor syn_inputs[1] = {expandSrc};
    synTensor syn_outputs[1] = {createTensor(inputs[2].dims.size(),
                                             syn_type_int32,
                                             inputs[2].dims,
                                             false,
                                             "expand_dst")};

    std::string expand_guid = "broadcast";
    std::string name = guid_ + expand_guid.c_str();

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     expand_guid.c_str(),
                                     name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] ScatterKernel AddExpandNode() failed = ",
             status);

    return syn_outputs[0];
  }

  synTensor AddZeroNode(ConvertTensors* ct) {
    PD_CHECK(ct != nullptr,
             "[RUNTIME] ScatterKernel AddZeroNode() ct is nullptr");

    auto scalar_zero = phi::Scalar(0.f);
    auto inputs = ct->GetTensors();
    ns_ConstantKernel::Params params;
    params.constant.f = scalar_zero.to<float>();

    std::string full = "constant_" + SynDataTypeToStr(inputs[2].type);

    synTensor syn_outputs[1] = {createTensor(
        inputs[2].dims.size(), inputs[2].type, inputs[2].dims, false, "zeros")};
    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     syn_outputs,
                                     0,
                                     1,
                                     &params,
                                     sizeof(params),
                                     full.c_str(),
                                     "full_zero",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] ScatterKernel AddZeroNode () failed = ",
             status);
    return syn_outputs[0];
  }

  synTensor AddOverwriteNode(ConvertTensors* ct,
                             synTensor index_tensor,
                             synTensor* update_tensor,
                             ns_ScatterKernel::Params params,
                             bool is_inplace,
                             bool is_output_persist = true) {
    PD_CHECK(ct != nullptr,
             "[RUNTIME] ScatterKernel AddOverwriteNode() ct is nullptr");

    auto inputs = ct->GetTensors();
    auto outputs = ct->GetTensors(false);
    std::vector<synTensor> syn_inputs;
    synSectionHandle section_shared = nullptr;

    // handle x tensor
    if (is_inplace) {
      section_shared = createSection();
    }
    syn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                      inputs[0].type,
                                      inputs[0].dims,
                                      true,
                                      inputs[0].name,
                                      section_shared));
    // handle index tensor
    syn_inputs.push_back(index_tensor);
    // handle update tensor
    if (update_tensor != nullptr) {
      syn_inputs.push_back(*update_tensor);
    } else {
      syn_inputs.push_back(createTensor(inputs[2].dims.size(),
                                        inputs[2].type,
                                        inputs[2].dims,
                                        true,
                                        inputs[2].name));
    }
    // handle output tensor
    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       is_output_persist,
                                       outputs[0].name,
                                       section_shared));

    guid_ = guid_ + SynDataTypeToStr(inputs[0].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "ScatterOverwrite",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] ScatterKernel AddScatterNode failed = ",
             status);

    return syn_outputs[0];
  }

  void AddAddNode(ConvertTensors* ct,
                  synTensor x,
                  synTensor index,
                  ns_ScatterKernel::Params params) {
    PD_CHECK(ct != nullptr,
             "[RUNTIME] ScatterKernel AddOverwriteNode() ct is nullptr");

    auto inputs = ct->GetTensors();
    auto outputs = ct->GetTensors(false);
    std::string node_name =
        "unsorted_scatter_add_fwd_" + SynDataTypeToStr(inputs[0].type);

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(x);
    syn_inputs.push_back(index);
    syn_inputs.push_back(createTensor(inputs[2].dims.size(),
                                      inputs[2].type,
                                      inputs[2].dims,
                                      true,
                                      inputs[2].name));

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     node_name.c_str(),
                                     "ScatterNoOverwrite",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] ScatterKernel AddAddNode () failed = ",
             status);
  }
};

template <typename T, typename Context>
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& update,
                   bool overwrite,
                   phi::DenseTensor* out) {
  PD_CHECK(index.dtype() == phi::DataType::INT32 ||
               index.dtype() == phi::DataType::INT64,
           "Scatter requires the index type be either int32 or int64");

  auto index_dims = phi::vectorize<int>(index.dims());
  auto update_dims = phi::vectorize<int>(update.dims());
  PD_CHECK(update_dims[0] == index_dims[0],
           "Scatter requires the 1st dim of update match the 1st dim of index");

  if (index_dims.size() == 2) {
    PD_CHECK(index_dims[1] != 1,
             "Scatter's index 2nd dim must be 1 for 2D index");
  } else if (index_dims.size() == 1) {
    index_dims.push_back(1);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Scatter requires the index type "
                                     "be either int32 or int64."));
  }

  // generate kernel name
  dev_ctx.template Alloc<T>(out);
  bool is_inplace = (out->data() == x.data());
  bool cast = (index.dtype() == phi::DataType::INT64);
  std::string recipe_name;

  auto dtype = PDDataTypeToSynDataType(x.dtype());
  Scatter::BuildRecipeName(recipe_name, dtype, is_inplace, cast, overwrite);

  // prepare kernel parameters
  phi::DenseTensor index_new(index);
  phi::DenseTensorMeta meta({index.dtype(), {phi::make_ddim(index_dims)}});
  index_new.set_meta(meta);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(index_new);
  ct.Add(update);
  ct.Add(out, false);

  OpCacheOperator op_info;
  ns_ScatterKernel::Params params;
  params.axis = x.dims().size() - 1;
  std::vector<DIMS> inputs_dims = ct.GetDims();

  // build scatter kernel if needed
  op_info.prepareOpInfo<T, ns_ScatterKernel::Params>(
      recipe_name, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Scatter op;
    auto index_tensor = op.AddCastNode(&ct, cast);
    auto expand_tensor = op.AddExpandNode(index_tensor, &ct);
    if (overwrite) {
      op.AddOverwriteNode(&ct, expand_tensor, nullptr, params, is_inplace);
    } else {
      auto zero_tensor = op.AddZeroNode(&ct);
      auto scatter_ow_tensor = op.AddOverwriteNode(
          &ct, expand_tensor, &zero_tensor, params, false, false);
      op.AddAddNode(&ct, scatter_ow_tensor, expand_tensor, params);
    }
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
