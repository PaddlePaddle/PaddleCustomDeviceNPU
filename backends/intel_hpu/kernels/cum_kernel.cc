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

#include <iostream>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

class CumsumOperator : public HpuOperator {
 public:
  CumsumOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}
  void AddNode(ConvertTensors& ct, ns_CumSumKernel::Params params) {
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

    std::string guid = +"cumsum_fwd_" + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
  std::string pName_;
};

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis_scalar,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  ConvertTensors ct;
  auto input_tensor = x;
  int params_axis = axis_scalar.to<int>();

  if (flatten) {
    PADDLE_ENFORCE_EQ(
        params_axis,
        -1,
        phi::errors::InvalidArgument(
            "when flatten is true, attr axis must be default %d, but got %d",
            -1,
            params_axis));

    input_tensor.Resize(phi::make_ddim({x.numel()}));
  }

  std::vector<int64_t> inputs_dim =
      phi::vectorize<int64_t>(input_tensor.dims());
  ct.Add(input_tensor);
  int params_exclusive = static_cast<int>(exclusive);
  int params_reverse = static_cast<int>(reverse);
  ns_CumSumKernel::Params params{params_axis, params_exclusive, params_reverse};

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_CumSumKernel::Params>(
      "cumsum_fwd_", {inputs_dim}, &params);

  auto recipe = op_info.GetRecipe();
  ct.Add(out, false);
  if (recipe == nullptr) {
    // compile
    CumsumOperator op(op_info.guid_, "cumsum_op");
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

PD_REGISTER_PLUGIN_KERNEL(cumsum,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::CumsumKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
