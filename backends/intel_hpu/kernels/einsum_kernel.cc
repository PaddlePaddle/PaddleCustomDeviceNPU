// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

struct EinsumParams {
  synEinsumParams params;
};

class Einsum : public HpuOperator {
 public:
  Einsum() : HpuOperator("einsum") {}

  void AddNode(ConvertTensors& ct, EinsumParams& params) {
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

    guid_ = guid_ + "_" + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "Einsum",
                                     nullptr,
                                     nullptr);
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void EinsumKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& inputs,
                  const std::string& equation,
                  phi::DenseTensor* out,
                  std::vector<phi::DenseTensor*> cache,
                  std::vector<phi::DenseTensor*> xshape UNUSED) {
  dev_ctx.template Alloc<T>(out);
  ConvertTensors ct;
  for (size_t i = 0; i < inputs.size(); i++) {
    ct.Add(inputs[i]);
  }

  ct.Add(out, false);

  OpCacheOperator op_info;
  EinsumParams params;
  params.params = synEinsumParams(equation.c_str());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, EinsumParams>("GatherKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Einsum op;

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

PD_REGISTER_PLUGIN_KERNEL(einsum,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EinsumKernel,
                          float,
                          int32_t,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
