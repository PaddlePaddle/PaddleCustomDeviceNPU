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
class SetTensorValue : public HpuOperator {
 public:
  SetTensorValue(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype,
               synSliceParams params) {
    assert(ins.size() == 2 && "input size should be 2");
    assert(outs.size() == 1 && "output size should be 1");

    synSectionHandle section = createSection();
    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "input", section),
        createTensor(ins[1].size(), datatype, ins[1], true, "value")};
    synTensor outputs[outs.size()] = {createTensor(
        outs[0].size(), datatype, outs[0], true, "output", section)};

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
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (SetTensorValue) failed = ",
             status);
  }
  std::string pName_;
};

template <typename T, typename Context>
void SetTensorValueKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& value,
                          const phi::IntArray& starts,
                          const phi::IntArray& ends,
                          const phi::IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          phi::DenseTensor* out) {
  auto starts_v = starts.GetData();
  auto ends_v = ends.GetData();

  PADDLE_ENFORCE_EQ(
      starts_v.size(),
      axes.size(),
      phi::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends_v.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  const auto& in_dims = x.dims();

  PADDLE_ENFORCE_EQ(x.data<T>(),
                    out->data<T>(),
                    phi::errors::InvalidArgument(
                        "The input ptr must be equal to output ptr."));
  // ToDo: handle decrease_axes and none_axes in future

  synSliceParams params = {{0}};
  for (int i = 0; i < in_dims.size(); i++) {
    params.axes[i] = i;
    params.steps[i] = 1;
    params.starts[i] = 0;
    params.ends[i] = in_dims[in_dims.size() - 1 - i];
  }
  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    params.starts[in_dims.size() - 1 - axes[i]] = starts[i];
    params.ends[in_dims.size() - 1 - axes[i]] = ends[i];
  }

  std::vector<int64_t> input_dim = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> value_dim = phi::vectorize<int64_t>(value.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synSliceParams>(
      "slice_insert", {input_dim, value_dim}, &params);

  auto recipe = op_info.GetRecipe();
  if (recipe == nullptr) {
    // compile
    SetTensorValue op("slice_insert", "SliceInsert");
    op.AddNode(
        {input_dim, value_dim}, {outputs_dim}, op_info.datatype_, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  // runtime
  std::map<std::string, uint64_t> tensors;
  tensors["input"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["value"] = reinterpret_cast<uint64_t>(value.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t>(out->data<T>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

// template <typename T, typename Context>
// void SetValueKernel(const Context& dev_ctx,
//                        const phi::DenseTensor& x,
//                        const phi::IntArray& starts,
//                        const phi::IntArray& ends,
//                        const phi::IntArray& steps,
//                        const std::vector<int64_t>& axes,
//                        const std::vector<int64_t>& decrease_axes,
//                        const std::vector<int64_t>& none_axes,
//                        const std::vector<int64_t>& shape,
//                        const std::vector<phi::Scalar>& values,
//                        phi::DenseTensor* out) {
//   std::vector<T> assgin_values;
//   assgin_values.reserve(values.size());
//   for (const auto& val : values) {
//     assgin_values.push_back(val.to<T>());
//   }
//   phi::DenseTensor value_tensor;
//   value_tensor.Resize(phi::make_ddim(shape));
//   custom_kernel::TensorFromVector(
//       dev_ctx, assgin_values, dev_ctx, &value_tensor);
//   value_tensor.Resize(phi::make_ddim(shape));

//   custom_kernel::SetTensorValueKernel<T, Context>(dev_ctx,
//                                                      x,
//                                                      value_tensor,
//                                                      starts,
//                                                      ends,
//                                                      steps,
//                                                      axes,
//                                                      decrease_axes,
//                                                      none_axes,
//                                                      out);
// }

//

}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(set_value,
//                           intel_hpu,
//                           ALL_LAYOUT,
//                           custom_kernel::SetValueKernel,
//                           float,
//                           phi::dtype::float16,
//                           phi::dtype::bfloat16) {
// }

PD_REGISTER_PLUGIN_KERNEL(set_value_with_tensor,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SetTensorValueKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int64_t) {}
