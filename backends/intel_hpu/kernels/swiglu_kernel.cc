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
#include "utils/utills.h"

namespace custom_kernel {

class SwiGlu : public HpuOperator {
 public:
  explicit SwiGlu(synDataType dtype)
      : HpuOperator("swiglu_fwd"), dtype_(dtype) {}
  void AddNode(const std::vector<DIMS>& ins, const std::vector<DIMS>& outs) {
    synTensor silu_inputs[1] = {
        createTensor(ins[0].size(), dtype_, ins[0], true, "x")};

    synTensor silu_outputs[1] = {
        createTensor(ins[0].size(), dtype_, ins[0], false, "silu_out")};

    synTensor mul_inputs[2] = {
        silu_outputs[0],
        createTensor(ins[1].size(), dtype_, ins[1], true, "y")};

    synTensor mul_outputs[outs.size()] = {
        createTensor(outs[0].size(), dtype_, outs[0], true, "output")};

    std::string mul = "mult_fwd_";
    std::string silu = "silu_fwd_";
    if (dtype_ == syn_type_fp16) {
      mul = mul + "f16";
      silu = silu + "f16";
    } else if (dtype_ == syn_type_bf16) {
      mul = mul + "bf16";
      silu = silu + "bf16";
    } else if (dtype_ == syn_type_single) {
      mul = mul + "f32";
      silu = silu + "f32";
    }
    std::string silu_name = guid_ + "_silu";
    std::string mul_name = guid_ + "_mul";

    synStatus status = synNodeCreate(graphHandle_,
                                     silu_inputs,
                                     silu_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     silu.c_str(),
                                     silu_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           mul_inputs,
                           mul_outputs,
                           2,
                           1,
                           nullptr,
                           0,
                           mul.c_str(),
                           mul_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void SwiGluKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const paddle::optional<phi::DenseTensor>& y,
                  phi::DenseTensor* out) {
  if (y) {
    // allocate memory on device.
    dev_ctx.template Alloc<T>(out);
    if (out->numel() == 0) {
      return;
    }

    std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
    const auto& y_tensor = y.get();
    std::vector<int64_t> y_dims = phi::vectorize<int64_t>(y_tensor.dims());
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

    OpCacheOperator op_info;
    op_info.prepareOpInfo<T, nullptr_t>(
        "SwiGluKernel", {x_dims, y_dims}, nullptr);
    auto recipe = op_info.GetRecipe();

    if (recipe == nullptr) {
      SwiGlu op(op_info.datatype_);

      op.AddNode({x_dims, y_dims}, {outputs_dim});
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }

    std::map<std::string, uint64_t> tensors;
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());
    tensors["y"] = reinterpret_cast<uint64_t>(y_tensor.data<T>());
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());
    RecipeRunner runner(recipe);
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(swiglu,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SwiGluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
