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

#include <climits>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

class TrilTriuOperator : public HpuOperator {
 public:
  TrilTriuOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype,
               ns_MatrixBandPartKernel::triParams params) {
    assert(ins.size() == 1 && "input size should be 1");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "input")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), datatype, outs[0], true, "output")};

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     &params,
                                     sizeof(params),
                                     guid_.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
  std::string pName_;
};

template <typename T, typename Context>
void TrilTriuKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  std::vector<int64_t> inputs_dim = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());
  ns_MatrixBandPartKernel::triParams params;

  if (lower) {
    params.numLower = INT_MIN;
    params.numUpper = diagonal;
    params.excludeDiag = 1;
  } else {
    params.numLower = diagonal;
    params.numUpper = INT_MAX;
    params.excludeDiag = 1;
  }

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_MatrixBandPartKernel::triParams>(
      "matrix_band_part_fwd", {inputs_dim}, &params);

  auto recipe = op_info.GetRecipe();
  if (recipe == nullptr) {
    // compile
    TrilTriuOperator op(op_info.guid_, lower ? "Tril_op" : "Triu_op");
    op.AddNode({inputs_dim}, {outputs_dim}, op_info.datatype_, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  // runtime
  std::map<std::string, uint64_t> tensors;
  tensors["input"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void TrilKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  custom_kernel::TrilTriuKernel<T, Context>(dev_ctx, x, diagonal, true, out);
}

template <typename T, typename Context>
void TriuKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  custom_kernel::TrilTriuKernel<T, Context>(dev_ctx, x, diagonal, false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          bool,
                          float,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(tril,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TrilKernel,
                          bool,
                          float,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(triu,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TriuKernel,
                          bool,
                          float,
                          phi::dtype::bfloat16) {}
