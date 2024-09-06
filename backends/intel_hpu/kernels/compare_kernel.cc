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

struct CompareParams {
  std::string op;
};

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out);

template <typename T, typename Context>
void EqualRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void GreaterEqualRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           int axis,
                           phi::DenseTensor* out);

class Compare : public HpuOperator {
 public:
  Compare() : HpuOperator("compare") {}
  void AddNode(ConvertTensors& ct, CompareParams& params) {
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

    std::string guid = params.op + "_fwd_" + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     guid.c_str(),
                                     params.op.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void NotEqualRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  phi::DenseTensor tmp;
  phi::DenseTensorMeta meta({x.dtype(), x.dims()});
  tmp.set_meta(meta);
  custom_kernel::EqualRawKernel<T, Context>(dev_ctx, x, y, axis, &tmp);
  custom_kernel::LogicalNotKernel<T, Context>(dev_ctx, tmp, out);
}

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::NotEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void EqualRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params;
  params.op = "equal";
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "EqualRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Compare op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  custom_kernel::EqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessThanRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  phi::DenseTensor tmp;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  tmp.set_meta(meta);
  custom_kernel::GreaterEqualRawKernel<T, Context>(dev_ctx, x, y, -1, &tmp);
  custom_kernel::LogicalNotKernel<T, Context>(dev_ctx, tmp, out);
}

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::LessThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessEqualRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params;
  params.op = "less_equal";
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "LessEqualRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Compare op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  custom_kernel::LessEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterThanRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          int axis,
                          phi::DenseTensor* out) {
  phi::DenseTensor tmp;
  phi::DenseTensorMeta meta({x.dtype(), x.dims()});
  tmp.set_meta(meta);
  custom_kernel::LessEqualRawKernel<T, Context>(dev_ctx, x, y, -1, &tmp);
  custom_kernel::LogicalNotKernel<T, Context>(dev_ctx, tmp, out);
}

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  custom_kernel::GreaterThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterEqualRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           int axis,
                           phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params;
  params.op = "greater_equal";
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "GreaterEqualRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Compare op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  custom_kernel::GreaterEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(not_equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::NotEqualKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(not_equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::NotEqualRawKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EqualKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EqualRawKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_than,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessThanKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_than_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessThanRawKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualRawKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_than,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_than_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanRawKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualRawKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}
