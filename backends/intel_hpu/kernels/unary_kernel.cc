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

#define PD_REGISTER_PLUGIN_KERNEL_32f(OP_NAME, GUID)                        \
  PD_REGISTER_PLUGIN_KERNEL(                                                \
      GUID, intel_hpu, ALL_LAYOUT, custom_kernel::OP_NAME##Kernel, float) { \
    kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype())); \
  }

#define PD_REGISTER_PLUGIN_KERNEL_16f32f(OP_NAME, GUID)                     \
  PD_REGISTER_PLUGIN_KERNEL(GUID,                                           \
                            intel_hpu,                                      \
                            ALL_LAYOUT,                                     \
                            custom_kernel::OP_NAME##Kernel,                 \
                            float,                                          \
                            phi::dtype::bfloat16,                           \
                            phi::dtype::float16) {                          \
    kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype())); \
  }

class Unary : public HpuOperator {
 public:
  Unary(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}

  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype) {
    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "x")};
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), datatype, outs[0], true, "output")};
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
  std::string pName_;
};

#define UNARY_KERNEL(kernel_func, node_name)                                  \
  template <typename T, typename Context>                                     \
  void kernel_func##Kernel(const Context& dev_ctx,                            \
                           const phi::DenseTensor& x,                         \
                           phi::DenseTensor* out) {                           \
    dev_ctx.template Alloc<T>(out);                                           \
    if (out->numel() == 0) {                                                  \
      return;                                                                 \
    }                                                                         \
    std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());           \
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());  \
                                                                              \
    OpCacheOperator op_info;                                                  \
    op_info.prepareOpInfo<T, nullptr_t>(#node_name "_fwd", {x_dim}, nullptr); \
    auto recipe = op_info.GetRecipe();                                        \
                                                                              \
    if (recipe == nullptr) {                                                  \
      Unary op(op_info.guid_, #node_name);                                    \
      op.AddNode({x_dim}, {outputs_dim}, op_info.datatype_);                  \
      op.Compile();                                                           \
      op_info.setOp(op);                                                      \
      recipe = op_info.GetRecipe();                                           \
    }                                                                         \
                                                                              \
    std::map<std::string, uint64_t> tensors;                                  \
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());                   \
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());           \
                                                                              \
    RecipeRunner runner(recipe);                                              \
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);        \
                                                                              \
    return;                                                                   \
  }

UNARY_KERNEL(Abs, abs)
UNARY_KERNEL(Tanh, tanh)
UNARY_KERNEL(Cos, cos)
UNARY_KERNEL(Sin, sin)
UNARY_KERNEL(Tan, tan)
UNARY_KERNEL(Asin, asin)
UNARY_KERNEL(Atan, atan)
UNARY_KERNEL(Acos, acos)
UNARY_KERNEL(Sinh, sinh)
UNARY_KERNEL(Cosh, cosh)
UNARY_KERNEL(Asinh, asinh)
UNARY_KERNEL(Acosh, acosh)
UNARY_KERNEL(Atanh, atanh)
UNARY_KERNEL(Relu, relu)
UNARY_KERNEL(Silu, silu)
UNARY_KERNEL(Exp, exp)
UNARY_KERNEL(Square, square)
UNARY_KERNEL(Sqrt, sqrt)
UNARY_KERNEL(Rsqrt, rsqrt)
UNARY_KERNEL(Softsign, softsign)
UNARY_KERNEL(Sigmoid, sigmoid)
UNARY_KERNEL(Log, log)
UNARY_KERNEL(Negative, neg)
UNARY_KERNEL(Floor, floor)
UNARY_KERNEL(Ceil, ceil);
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL_16f32f(Abs, abs);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Tanh, tanh);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Cos, cos);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Sin, sin);
PD_REGISTER_PLUGIN_KERNEL_32f(Tan, tan);
PD_REGISTER_PLUGIN_KERNEL_32f(Asin, asin);
PD_REGISTER_PLUGIN_KERNEL_32f(Atan, atan);
PD_REGISTER_PLUGIN_KERNEL_32f(Acos, acos);
PD_REGISTER_PLUGIN_KERNEL_32f(Sinh, sinh);
PD_REGISTER_PLUGIN_KERNEL_32f(Cosh, cosh);
PD_REGISTER_PLUGIN_KERNEL_32f(Asinh, asinh);
PD_REGISTER_PLUGIN_KERNEL_32f(Acosh, acosh);
PD_REGISTER_PLUGIN_KERNEL_32f(Atanh, atanh);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Relu, relu);
PD_REGISTER_PLUGIN_KERNEL_32f(Silu, silu);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Exp, exp);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Sqrt, sqrt);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Rsqrt, rsqrt);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Softsign, softsign);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Sigmoid, sigmoid);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Log, log);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Negative, neg);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Floor, floor);
PD_REGISTER_PLUGIN_KERNEL_16f32f(Ceil, ceil);
