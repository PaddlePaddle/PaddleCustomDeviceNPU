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
#include "glog/logging.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/utills.h"

namespace custom_kernel {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out);

class BinaryOperator : public HpuOperator {
 public:
  BinaryOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}

  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype) {
    assert(ins.size() == 2 && "input size should be 2");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "x"),
        createTensor(ins[1].size(), datatype, ins[1], true, "y")};
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
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate binary fwd () failed = %d",
             status);
  }
  std::string pName_;
};

#define BINARY_RAW_KERNEL(kernel_func, node_name)                            \
  template <typename T, typename Context>                                    \
  void kernel_func##RawKernel(const Context& dev_ctx,                        \
                              const phi::DenseTensor& x,                     \
                              const phi::DenseTensor& y,                     \
                              int axis,                                      \
                              phi::DenseTensor* out) {                       \
    dev_ctx.template Alloc<T>(out);                                          \
    VLOG(6) << "CALL HPU " << #kernel_func << "RawKernel";                   \
    std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());          \
    std::vector<int64_t> y_dim = phi::vectorize<int64_t>(y.dims());          \
    if (y_dim.size() == 0) {                                                 \
      y_dim.push_back(1);                                                    \
    }                                                                        \
    if (x_dim.size() == 0) {                                                 \
      x_dim.push_back(1);                                                    \
    }                                                                        \
    std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims()); \
    if (outputs_dim.size() == 0) {                                           \
      outputs_dim.push_back(1);                                              \
    }                                                                        \
    OpCacheOperator op_info;                                                 \
    op_info.prepareOpInfo<T, nullptr_t>(                                     \
        #node_name "_fwd", {x_dim, y_dim}, nullptr);                         \
    auto recipe = op_info.GetRecipe();                                       \
                                                                             \
    if (recipe == nullptr) {                                                 \
      BinaryOperator op(op_info.guid_, #node_name);                          \
      op.AddNode({x_dim, y_dim}, {outputs_dim}, op_info.datatype_);          \
      op.Compile();                                                          \
      op_info.setOp(op);                                                     \
      recipe = op_info.GetRecipe();                                          \
    }                                                                        \
                                                                             \
    std::map<std::string, uint64_t> tensors;                                 \
    tensors["x"] = reinterpret_cast<uint64_t>(x.data<T>());                  \
    tensors["y"] = reinterpret_cast<uint64_t>(y.data<T>());                  \
    tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());          \
                                                                             \
    RecipeRunner runner(recipe);                                             \
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);       \
                                                                             \
    return;                                                                  \
  }

#define BINARY_KERNEL(kernel_func)                                      \
  template <typename T, typename Context>                               \
  void kernel_func##Kernel(const Context& dev_ctx,                      \
                           const phi::DenseTensor& x,                   \
                           const phi::DenseTensor& y,                   \
                           phi::DenseTensor* out) {                     \
    int axis = -1;                                                      \
    custom_kernel::kernel_func##RawKernel<T>(dev_ctx, x, y, axis, out); \
  }

// #define PRINT_MACRO_HELPER(args...) #args
// #define PRINT_MACRO(x) #x "=" PRINT_MACRO_HELPER(x)
// #pragma message(PRINT_MACRO(BINARY_RAW_KERNEL(Add, add)))
// #pragma message(PRINT_MACRO(BINARY_KERNEL(Add)))

BINARY_RAW_KERNEL(Add, add);
BINARY_KERNEL(Add);

BINARY_RAW_KERNEL(Div, div);
BINARY_KERNEL(Div);

BINARY_RAW_KERNEL(Max, max);
BINARY_KERNEL(Max);

BINARY_RAW_KERNEL(Min, min);
BINARY_KERNEL(Min);

BINARY_RAW_KERNEL(Mod, mod);
BINARY_KERNEL(Mod);

BINARY_RAW_KERNEL(Mult, mult);
BINARY_KERNEL(Mult);

BINARY_RAW_KERNEL(Pow, pow);
BINARY_KERNEL(Pow);

BINARY_RAW_KERNEL(Sub, sub);
BINARY_KERNEL(Sub);

template <typename T, typename Context>
void PowKernelScalar(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::Scalar& factor_scalar,
                     phi::DenseTensor* out) {
  phi::DenseTensor y;
  phi::DenseTensorMeta y_meta({x.dtype(), {1}});
  y.set_meta(y_meta);
  std::vector<int64_t> shape_vec = {1};
  phi::IntArray scalar_shape(shape_vec);
  custom_kernel::FullKernel<float, Context>(
      dev_ctx, scalar_shape, factor_scalar, x.dtype(), &y);
  custom_kernel::PowKernel<T, Context>(dev_ctx, x, y, out);
}

}  // namespace custom_kernel

#define HPU_KERNEL_REGISTER(kernel_name, kernel_func, ...) \
  PD_REGISTER_PLUGIN_KERNEL(kernel_name,                   \
                            intel_hpu,                     \
                            ALL_LAYOUT,                    \
                            custom_kernel::kernel_func,    \
                            __VA_ARGS__) {}

#define PD_REGISTER_PLUGIN_KERNEL_FPx3(OP_NAME, GUID) \
  HPU_KERNEL_REGISTER(GUID##_raw,                     \
                      OP_NAME##RawKernel,             \
                      float,                          \
                      phi::dtype::float16,            \
                      phi::dtype::bfloat16)           \
  HPU_KERNEL_REGISTER(                                \
      GUID, OP_NAME##Kernel, float, phi::dtype::float16, phi::dtype::bfloat16)

#define PD_REGISTER_PLUGIN_KERNEL_FPx2(OP_NAME, GUID)        \
  HPU_KERNEL_REGISTER(GUID##_raw, OP_NAME##RawKernel, float) \
  HPU_KERNEL_REGISTER(GUID, OP_NAME##Kernel, float, phi::dtype::bfloat16)

PD_REGISTER_PLUGIN_KERNEL_FPx3(Add, add);
PD_REGISTER_PLUGIN_KERNEL_FPx3(Max, maximum);
PD_REGISTER_PLUGIN_KERNEL_FPx3(Min, minimum);
PD_REGISTER_PLUGIN_KERNEL_FPx3(Mult, multiply);
PD_REGISTER_PLUGIN_KERNEL_FPx3(Pow, elementwise_pow);
PD_REGISTER_PLUGIN_KERNEL_FPx3(Sub, subtract);
PD_REGISTER_PLUGIN_KERNEL_FPx3(Div, divide);
PD_REGISTER_PLUGIN_KERNEL_FPx2(Mod, remainder);

PD_REGISTER_PLUGIN_KERNEL(pow,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::PowKernelScalar,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}