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

#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

struct SqueezeParams {
  synSqueezeParams params;
};

class Squeeze : public HpuOperator {
 public:
  Squeeze() : HpuOperator("squeeze") {}

  void AddNode(ConvertTensors& ct, SqueezeParams& params) {
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

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "squeeze",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = ", status);
  }
};

class SqueezeNull : public HpuOperator {
 public:
  SqueezeNull() : HpuOperator("squeeze") {}

  void AddNode(ConvertTensors& ct) {
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

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     "squeeze",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = ", status);
  }
};

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call intel_hpu SqueezeKernel";
  dev_ctx.template Alloc<T>(out);

  std::vector<int32_t> axes(axes_int_array.GetData().begin(),
                            axes_int_array.GetData().end());

  PADDLE_ENFORCE_LT(
      axes.size(),
      2,
      phi::errors::InvalidArgument(
          "Intel HPU only support axis.size() = 0 or 1 at present."));

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  synRecipeHandle recipe = nullptr;

  if (axes.size() == 0) {
    OpCacheOperator op_info;
    std::vector<DIMS> inputs_dims = ct.GetDims();
    op_info.prepareOpInfo<T, nullptr_t>("SqueezeKernel", inputs_dims, nullptr);
    recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      SqueezeNull op;

      op.AddNode(ct);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }
  } else {
    int32_t dim = axes[0];

    if (dim < 0) {
      dim += x.dims().size();
    }

    OpCacheOperator op_info;
    SqueezeParams params;
    params.params.axis = static_cast<int32_t>(x.dims().size()) - 1 - dim;
    std::vector<DIMS> inputs_dims = ct.GetDims();
    op_info.prepareOpInfo<T, SqueezeParams>(
        "SqueezeKernel", inputs_dims, &params);
    recipe = op_info.GetRecipe();
    if (recipe == nullptr) {
      Squeeze op;

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
void SqueezeWithXShapeKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::IntArray& axes_int_array,
                             phi::DenseTensor* out,
                             phi::DenseTensor* xshape) {
  custom_kernel::SqueezeKernel<T, Context>(dev_ctx, x, axes_int_array, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squeeze,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_with_xshape,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeWithXShapeKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          int32_t,
                          int64_t) {}
