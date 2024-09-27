// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
namespace {
std::vector<int64_t> CalSections(const std::vector<int64_t>& input_shape,
                                 int axis,
                                 const std::vector<int64_t>& origin_sections) {
  std::vector<int64_t> sections(origin_sections);

  int64_t unknown_dim_index = -1;
  int64_t num_of_unknown_dim = 0;
  int64_t sum_of_known_dims = 0;

  for (size_t i = 0; i < sections.size(); ++i) {
    if (sections[i] == -1) {
      num_of_unknown_dim++;
      unknown_dim_index = i;
    } else {
      sum_of_known_dims += sections[i];
    }
  }

  if (num_of_unknown_dim == 0) {
    return sections;
  }

  PADDLE_ENFORCE_EQ(
      num_of_unknown_dim,
      1,
      phi::errors::InvalidArgument("Split noly support at "
                                   "most 1 unknown dim, but got: %ld",
                                   num_of_unknown_dim));
  sections[unknown_dim_index] = input_shape[axis] - sum_of_known_dims;
  return sections;
}
}  // namespace

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::IntArray& num_or_sections,
                 const phi::Scalar& axis_scalar,
                 std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("split");
  auto origin_sections = num_or_sections.GetData();
  PADDLE_ENFORCE_GT(
      origin_sections.size(),
      0,
      phi::errors::InvalidArgument("Split should set num_or_sections."));

  if (LaunchAOTKernel()) {
    int axis = axis_scalar.to<int>();
    const auto& input_shape = phi::vectorize(x.dims());
    if (axis < 0) {
      axis += input_shape.size();
    }
    auto sections = CalSections(input_shape, axis, origin_sections);
    PADDLE_ENFORCE_EQ(outs.size(),
                      sections.size(),
                      phi::errors::InvalidArgument(
                          "out size %zu should equal to sections size %zu.",
                          outs.size(),
                          sections.size()));

    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    std::vector<phi::DenseTensor> outputs;
    for (size_t i = 0; i < outs.size(); ++i) {
      dev_ctx.template Alloc<T>(outs[i]);
      outputs.emplace_back(
          MaybeCreateOrTrans64To32bits(dev_ctx, *(outs[i]), false));
    }

    int64_t start = 0;
    std::vector<int64_t> axes = {axis};
    for (size_t i = 0; i < sections.size(); ++i) {
      start += (i == 0) ? 0 : sections[i - 1];
      std::vector<int64_t> starts = {start};
      custom_kernel::SliceBase(dev_ctx, input_x, axes, starts, &(outputs[i]));
      MaybeTransResult(dev_ctx, outputs[i], outs[i]);
    }

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    std::vector<std::string> names;
    names.reserve(outs.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
      dev_ctx.template Alloc<T>(outs[i]);
      names.emplace_back(std::string("out") + std::to_string(i));
      values.emplace_back(outs[i]);
    }
    output_names["Out"] = names;
    outputs["Out"] = values;

    GcuAttributeMap attrs;
    attrs["axis"] = axis_scalar.to<int>();

    std::vector<int> sections = GetIntList(num_or_sections.GetData());
    attrs["sections"] = sections;
    attrs["num"] = 0;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "split", dev_ctx);
  }
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        int num,
                        const phi::Scalar& axis_scalar,
                        std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("split_with_num");
  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec =
      std::vector<int64_t>(num, input_axis_dim / num);
  phi::IntArray sections(sections_vec);
  custom_kernel::SplitKernel<T, Context>(
      dev_ctx, x, sections, axis_scalar, outs);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(split,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SplitKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(split_with_num,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SplitWithNumKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
