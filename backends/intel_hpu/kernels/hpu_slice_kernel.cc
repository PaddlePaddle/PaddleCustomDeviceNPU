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
#include "utils/utills.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "paddle/phi/extension.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {

class SliceOperator : public HpuOperator {
 public:
  SliceOperator(std::string guid_prefix, std::string node_name)
    : HpuOperator(guid_prefix), pName_(node_name) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype,
               synSliceParamsV2 params) {
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
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
  std::string pName_;
};


template <typename T = int64_t>
inline phi::DDim GetDecreasedDims(const phi::DDim slice_dims,
                                  const std::vector<T>& decrease_axes,
                                  std::vector<T>* infer_flags = nullptr) {
  phi::DDim decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        PADDLE_ENFORCE_EQ(decreased_dims[axis],
                          1,
                          phi::errors::InvalidArgument(
                              "Decrease dim should be 1, but now received %d",
                              decreased_dims[axis]));
      }
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }

    decreased_dims = phi::make_ddim(new_shape);
  }
  return decreased_dims;
}

template <typename T, typename Context>
void SliceRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& axes,
                    const phi::IntArray& starts_array,
                    const phi::IntArray& ends_array,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    phi::DenseTensor* out) {

  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  std::vector<int64_t> strides(axes.size(), 1);

  PADDLE_ENFORCE_EQ(starts.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Infer output dims
  const auto& in_dims = x.dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start + 1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }
  
  phi::funcs::UpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = phi::funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = custom_kernel::GetDecreasedDims(slice_dims, decrease_axis);
  out->Resize(out_dims);

  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  
  synSliceParamsV2 params;
  for (size_t i = 0; i < in_dims.size(); i++)
  {
    params.axes[i] = i;
    params.steps[i] = 1;
    params.starts[i] = 0;
    params.ends[i] = in_dims[in_dims.size() - 1 - i];
  }
  for (size_t i = 0; i < axes.size(); i++)
  {
    params.starts[in_dims.size() - 1 - axes[i]] = starts[i];
    params.ends[in_dims.size() - 1 - axes[i]] = ends[i];
  }
/*
  for (size_t i = 0; i < in_dims.size(); i++)
  {     
    std::cout << "**************************    " << std::endl;
    std::cout << i << ": " << params.axes[i] << ", " << params.starts[i] << ", " << params.ends[i] << ", " << params.steps[i] << std::endl;
  }
    params.axes  [0] = 0;    params.axes  [1] = 1;    params.axes  [2] = 2;    params.axes  [3] = 3;
    params.starts[0] = 0;    params.starts[1] = 2;    params.starts[2] = 0;    params.starts[3] = 1;
    params.ends  [0] = 6;    params.ends  [1] = 4;    params.ends  [2] = 3;    params.ends  [3] = 3;
    params.steps [0] = 1;    params.steps [1] = 1;    params.steps [2] = 1;    params.steps [3] = 1;
*/
  std::vector<int64_t> inputs_dim = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> outputs_dim = phi::vectorize<int64_t>(out->dims());
  
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synSliceParamsV2>("slice", {inputs_dim}, &params);

  auto recipe = op_info.GetRecipe();
  if(recipe == nullptr){
    // compile
    SliceOperator op("slice", "Slice");
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SliceRawKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
