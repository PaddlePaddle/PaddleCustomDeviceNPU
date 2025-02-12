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
#include "paddle/phi/common/amp_type_traits.h"
#include "utils/utils.h"

namespace custom_kernel {

class Topk : public HpuOperator {
 public:
  Topk() : HpuOperator("topk") {}

  void AddNode(ConvertTensors& ct, ns_TopkNodeV2::ParamsV4 params) {
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

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "topk",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] Topk synNodeCreate () failed = ",
             status);
  }
};

template <typename T, typename Context>
void TopkKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& k_scalar,
                int axis,
                bool largest,
                bool sorted,
                phi::DenseTensor* out,
                phi::DenseTensor* indices) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int>(indices);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);
  ct.Add(indices, false);

  ns_TopkNodeV2::ParamsV4 params{};
  params.bsw = k_scalar.to<int>();
  axis = axis < 0 ? axis + x.dims().size() : axis;
  params.axis = x.dims().size() - axis - 1;
  params.bottomK = !largest;
  params.isVcData = false;
  params.isStable = false;

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_TopkNodeV2::ParamsV4>(
      "TopkKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Topk op;
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

PD_REGISTER_PLUGIN_KERNEL(topk,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TopkKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          int) {}
