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
#include "utils/utills.h"

namespace custom_kernel {
struct LogicalParams {
  std::string op;
};

class Logical : public HpuOperator {
 public:
  Logical() : HpuOperator("logical") {}
  void AddNode(ConvertTensors& ct, LogicalParams& params) {
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
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    std::string guid = params.op + "_" + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     guid.c_str(),
                                     params.op.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  LogicalParams params;
  params.op = "not";
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, LogicalParams>(
      "LogicalNotKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Logical op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void LogicalOrKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  OpCacheOperator op_info;
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);

  ct.Add(out, false);

  LogicalParams params;
  params.op = "or";
  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, LogicalParams>(
      "LogicalOrKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Logical op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void LogicalAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  OpCacheOperator op_info;
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  LogicalParams params;
  params.op = "and";
  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, LogicalParams>(
      "LogicalAndKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Logical op;

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

PD_REGISTER_PLUGIN_KERNEL(logical_not,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalNotKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_or,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalOrKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_and,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalAndKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
