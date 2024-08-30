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

#include "funcs.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/utills.h"

namespace custom_kernel {

struct ReduceParams {
  ns_Reduction::Params params;
  bool keep_dim;
  int dim;
  std::string op;
  bool reduce_all;
};

class Reduce : public HpuOperator {
 public:
  Reduce() : HpuOperator("reduce") {}
  void AddNode(ConvertTensors& ct, ReduceParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);
    if (params.reduce_all) {
      inputs[0].dims = std::vector<int64_t>({static_cast<int64_t>(inputs[0].num_elements)});
      outputs[0].dims = std::vector<int64_t>({1});
    }
    std::vector<synTensor> syn_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                        inputs[i].type,
                                        inputs[i].dims,
                                        true,
                                        inputs[i].name));
    }

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    auto reduce_guid =
        guid_ + "_" + params.op + "_fwd_" + SynDataTypeToStr(inputs[0].type);
    std::string reduce_name = guid_ + "_" + params.op + "_before_reshape";
    std::vector<synTensor> reduce_outputs;

    auto tmp_dims = inputs[0].dims;
    tmp_dims[params.dim] = 1;
    reduce_outputs.push_back(createTensor(
        tmp_dims.size(), inputs[0].type, tmp_dims, false, "reduce_out"));

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     reduce_outputs.data(),
                                     syn_inputs.size(),
                                     reduce_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     reduce_guid.c_str(),
                                     reduce_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
    std::string reshape_name = guid_ + "_reshape";
    std::string reshape_guid = "reshape";
    status = synNodeCreate(graphHandle_,
                           reduce_outputs.data(),
                           syn_outputs.data(),
                           reduce_outputs.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           reshape_guid.c_str(),
                           reshape_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  ReduceParams params;
  params.keep_dim = keep_dim;
  params.op = "mean";
  if (dims.size() == 0) {
    params.dim = 0;
    params.reduce_all = true;

    params.params.reductionDimension = 0;
  } else {
    PD_CHECK(dims.size() == 1,
             "Reduction only support axis = 1 but got %d.",
             dims.size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(rank));
    params.dim = dim;
    params.reduce_all = false;
    params.params.reductionDimension = rank - 1 - dim;
  }

  op_info.prepareOpInfo<T, ReduceParams>("MeanKernel", {inputs_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Reduce op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  ReduceParams params;
  params.keep_dim = keep_dim;
  params.op = "max";
  if (dims.size() == 0) {
    params.dim = 0;
    params.reduce_all = true;

    params.params.reductionDimension = 0;
  } else {
    PD_CHECK(dims.size() == 1,
             "Reduction only support axis = 1 but got %d.",
             dims.size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(rank));
    params.dim = dim;
    params.reduce_all = false;
    params.params.reductionDimension = rank - 1 - dim;
  }

  op_info.prepareOpInfo<T, ReduceParams>("MaxKernel", {inputs_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Reduce op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  ReduceParams params;
  params.keep_dim = keep_dim;
  params.op = "min";
  if (dims.size() == 0) {
    params.dim = 0;
    params.reduce_all = true;

    params.params.reductionDimension = 0;
  } else {
    PD_CHECK(dims.size() == 1,
             "Reduction only support axis = 1 but got %d.",
             dims.size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(rank));
    params.dim = dim;
    params.reduce_all = false;
    params.params.reductionDimension = rank - 1 - dim;
  }

  op_info.prepareOpInfo<T, ReduceParams>("MinKernel", {inputs_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Reduce op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  ReduceParams params;
  params.keep_dim = keep_dim;
  params.op = "sum";
  if (dims.size() == 0) {
    params.dim = 0;
    params.reduce_all = true;

    params.params.reductionDimension = 0;
  } else {
    PD_CHECK(dims.size() == 1,
             "Reduction only support axis = 1 but got %d.",
             dims.size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(rank));
    params.dim = dim;
    params.reduce_all = false;
    params.params.reductionDimension = rank - 1 - dim;
  }

  op_info.prepareOpInfo<T, ReduceParams>("SumKernel", {inputs_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Reduce op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                phi::DataType out_dtype,
                bool keep_dim,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  auto rank = static_cast<int32_t>(x.dims().size());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  ReduceParams params;
  params.keep_dim = keep_dim;
  params.op = "prod";
  if (dims.size() == 0) {
    params.dim = 0;
    params.reduce_all = true;

    params.params.reductionDimension = 0;
  } else {
    PD_CHECK(dims.size() == 1,
             "Reduction only support axis = 1 but got %d.",
             dims.size());
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(rank));
    params.dim = dim;
    params.reduce_all = false;
    params.params.reductionDimension = rank - 1 - dim;
  }

  op_info.prepareOpInfo<T, ReduceParams>("ProdKernel", {inputs_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Reduce op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(max,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(prod,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ProdKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(min,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::MinKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}