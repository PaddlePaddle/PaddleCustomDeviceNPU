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

class Transpose : public HpuOperator {
 public:
  Transpose(synDataType dtype) : HpuOperator("transpose"), dtype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synTransposeParams params) {
    std::vector<synTensor> inputs;
    inputs.push_back(createTensor(ins[0].size(), dtype_, ins[0], true, "x"));
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "out")};

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs.data(),
                                     outputs,
                                     ins.size(),
                                     1,
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     "transpose",
                                     nullptr,
                                     nullptr);
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  synTransposeParams params;
  auto rank = axis.size();
  std::vector<int> tmp_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    tmp_axis[i] = rank - 1 - axis[i];
  }
  std::reverse(tmp_axis.begin(), tmp_axis.end());
  for (size_t i = 0; i < tmp_axis.size(); i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(tmp_axis[i]);
  }
  params.tensorDim = rank;

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synTransposeParams>(
      "TransposeKernel", {x_dims}, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Transpose op(op_info.datatype_);
    op.AddNode({x_dims}, {outputs_dim}, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["out"] = reinterpret_cast<uint64_t>(out->data<T>());
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(transpose,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeKernel,
                          int,
                          uint8_t,
                          int8_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
