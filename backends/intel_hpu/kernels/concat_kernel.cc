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

#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

class Concat : public HpuOperator {
 public:
  explicit Concat(synDataType dtype) : HpuOperator("concat"), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, unsigned params) {
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
                                     "CONCAT",
                                     nullptr,
                                     nullptr);

    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  int axis = axis_scalar.to<int>();
  axis = CanonicalAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
  axis = static_cast<int64_t>(ins[0]->dims().size()) - 1 - axis;

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  for (size_t i = 0; i < ins.size(); i++) {
    ct.Add(ins[i]);
  }
  ct.Add(out, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  unsigned params = static_cast<unsigned>(axis);
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, unsigned>("ConcatKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Concat op(op_info.datatype_);
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

PD_REGISTER_PLUGIN_KERNEL(concat,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          float,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
