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
#include "utils/utils.h"

namespace custom_kernel {
class SplitOp : public HpuOperator {
 public:
  explicit SplitOp(synDataType dtype) : HpuOperator("split"), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, synSplitParams params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensor(
          inputs[i].dims.size(), dtype_, inputs[i].dims, true, inputs[i].name));
    }

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         dtype_,
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
                                     "SPLIT",
                                     nullptr,
                                     nullptr);

    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  // control with environment variables
  PADDLE_ENFORCE_EQ(
      num_or_sections.size(),
      outs.size(),
      phi::errors::InvalidArgument("The size of num_or_sections must be equal "
                                   "to the size of out tensors."));

  auto sections = num_or_sections.GetData();
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = x.dims().size() + axis;
  }

  ConvertTensors ct;
  ct.Add(x);
  for (size_t i = 0; i < outs.size(); i++) {
    dev_ctx.template Alloc<T>(outs[i]);
    ct.Add(outs[i], false);
  }

  synSplitParams params = {{0}};
  params.axis = x.dims().size() - 1 - axis;

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synSplitParams>("split", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    SplitOp op(op_info.datatype_);
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  // runtime
  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(split,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SplitKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
