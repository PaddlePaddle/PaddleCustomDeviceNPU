// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
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
#include "utils/utils.h"

namespace custom_kernel {

struct CompareParams {
  char op[MAX_OPNAME_LEN];
};

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

    std::string guid =
        std::string(params.op) + "_fwd_" + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     guid.c_str(),
                                     params.op,
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

class CompareCast : public HpuOperator {
 public:
  CompareCast() : HpuOperator("compare") {}
  void AddNode(ConvertTensors& ct, CompareParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> x_i64;
    x_i64.push_back(createTensor(inputs[0].dims.size(),
                                 inputs[0].type,
                                 inputs[0].dims,
                                 true,
                                 inputs[0].name));
    std::vector<synTensor> x_i32;
    auto x_cast = createTensor(
        inputs[0].dims.size(), syn_type_int32, inputs[0].dims, false, "x_cast");
    x_i32.push_back(x_cast);

    std::string guid_cast = "cast_i64_to_i32";
    synStatus status = synNodeCreate(graphHandle_,
                                     x_i64.data(),
                                     x_i32.data(),
                                     x_i64.size(),
                                     x_i32.size(),
                                     nullptr,
                                     0,
                                     guid_cast.c_str(),
                                     "cast_x",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (compare/cast_x) failed = ",
             status);

    std::vector<synTensor> y_i64;
    y_i64.push_back(createTensor(inputs[1].dims.size(),
                                 inputs[1].type,
                                 inputs[1].dims,
                                 true,
                                 inputs[1].name));

    std::vector<synTensor> y_i32;
    auto y_cast = createTensor(
        inputs[1].dims.size(), syn_type_int32, inputs[1].dims, false, "y_cast");
    y_i32.push_back(y_cast);

    status = synNodeCreate(graphHandle_,
                           y_i64.data(),
                           y_i32.data(),
                           y_i64.size(),
                           y_i32.size(),
                           nullptr,
                           0,
                           guid_cast.c_str(),
                           "cast_y",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (compare/cast_x) failed = ",
             status);

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(x_cast);
    syn_inputs.push_back(y_cast);

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    std::string guid = std::string(params.op) + "_fwd_i32";

    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           syn_outputs.data(),
                           syn_inputs.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           guid.c_str(),
                           params.op,
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (compare) failed = ",
             status);
  }
};

class CompareNotEqual : public HpuOperator {
 public:
  CompareNotEqual() : HpuOperator("compare") {}
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

    std::vector<synTensor> equal_out;
    equal_out.push_back(createTensor(outputs[0].dims.size(),
                                     outputs[0].type,
                                     outputs[0].dims,
                                     false,
                                     "equal_out"));

    std::string guid_eq = "equal_fwd_i64";

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     equal_out.data(),
                                     syn_inputs.size(),
                                     equal_out.size(),
                                     nullptr,
                                     0,
                                     guid_eq.c_str(),
                                     "equal",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    std::string guid_not = "not_fwd_i8";

    status = synNodeCreate(graphHandle_,
                           equal_out.data(),
                           syn_outputs.data(),
                           equal_out.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           guid_not.c_str(),
                           "not",
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
  VLOG(6) << "call HPU NotEqualRawKernel";
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params = {};
  snprintf(params.op, MAX_OPNAME_LEN, "%s", "not_equal");
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "NotEqualRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (x.dtype() == phi::DataType::INT64) {
      CompareNotEqual op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    } else {
      Compare op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    }
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
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
  VLOG(6) << "call HPU EqualRawKernel";
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params = {};
  snprintf(params.op, MAX_OPNAME_LEN, "%s", "equal");
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
  VLOG(6) << "call HPU LessThanRawKernel";
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params = {};
  snprintf(params.op, MAX_OPNAME_LEN, "%s", "less");
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "LessThanRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (x.dtype() == phi::DataType::INT64) {
      CompareCast op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    } else {
      Compare op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    }
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
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
  VLOG(6) << "call HPU LessEqualRawKernel";
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params = {};
  snprintf(params.op, MAX_OPNAME_LEN, "%s", "less_equal");
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "LessEqualRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (x.dtype() == phi::DataType::INT64) {
      CompareCast op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    } else {
      Compare op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    }
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
  VLOG(6) << "call HPU GreaterThanRawKernel";
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params = {};
  snprintf(params.op, MAX_OPNAME_LEN, "%s", "greater");
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "GreaterThanRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (x.dtype() == phi::DataType::INT64) {
      CompareCast op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    } else {
      Compare op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    }
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
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
  VLOG(6) << "call HPU GreaterEqualRawKernel";
  dev_ctx.template Alloc<bool>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(out, false);

  CompareParams params = {};
  snprintf(params.op, MAX_OPNAME_LEN, "%s", "greater_equal");
  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, CompareParams>(
      "GreaterEqualRawKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (x.dtype() == phi::DataType::INT64) {
      CompareCast op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    } else {
      Compare op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    }
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
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(not_equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::NotEqualRawKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EqualKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EqualRawKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_than,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessThanKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_than_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessThanRawKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(less_equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualRawKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_than,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_than_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanRawKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_equal,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(greater_equal_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualRawKernel,
                          phi::dtype::bfloat16,
                          float,
                          uint8_t,
                          int32_t,
                          int64_t,
                          bool) {}
