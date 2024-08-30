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
#include "perf_lib_layer_params.h"
#include "utils/utills.h"

namespace custom_kernel {
struct RangeParams {
  ns_RangeKernel::Params params;
};

class Range : public HpuOperator {
 public:
  Range() : HpuOperator("range") {}
  void AddNode(ConvertTensors& ct, RangeParams& params) {
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

    std::string guid = guid_ + "_" + SynDataTypeToStr(outputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid.c_str(),
                                     "Range",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

void GetSize(int32_t start, int32_t end, int32_t step, int64_t* size) {
  PADDLE_ENFORCE_NE(step,
                    0,
                    phi::errors::InvalidArgument("The step of range op should "
                                                 "not be 0."));
  if (start < end) {
    PADDLE_ENFORCE_GT(
        step,
        0,
        phi::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }
  *size = std::ceil(std::abs((end - start) / step));
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const phi::DenseTensor& start_t,
                        const phi::DenseTensor& end_t,
                        const phi::DenseTensor& step_t,
                        phi::DenseTensor* out) {
  phi::DenseTensor n;
  n.Resize(start_t.dims());
  T* n_data = dev_ctx.template HostAlloc<T>(&n);

  TensorCopy(dev_ctx, start_t, true, &n, phi::CPUPlace());
  T start = n_data[0];

  TensorCopy(dev_ctx, end_t, true, &n, phi::CPUPlace());
  T end = n_data[0];

  TensorCopy(dev_ctx, step_t, true, &n, phi::CPUPlace());
  T step = n_data[0];

  int64_t size = 0;
  GetSize(static_cast<int32_t>(start),
          static_cast<int32_t>(end),
          static_cast<int32_t>(step),
          &size);

  out->Resize(phi::make_ddim({size}));
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(out, false);

  RangeParams params;
  if (std::is_same<T, phi::dtype::bfloat16>::value ||
      std::is_same<T, phi::dtype::float16>::value ||
      std::is_same<T, float>::value) {
    params.params.start.f = static_cast<float>(start);
    params.params.limit.f = static_cast<float>(end);
    params.params.delta.f = static_cast<float>(step);
  } else {
    params.params.start.i = static_cast<float>(start);
    params.params.limit.i = static_cast<float>(end);
    params.params.delta.i = static_cast<float>(step);
  }

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, RangeParams>("ArangeKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Range op;

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
void ArangeKernel(const Context& dev_ctx,
                  const phi::Scalar& start,
                  const phi::Scalar& end,
                  const phi::Scalar& step,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(out, false);

  RangeParams params;
  if (std::is_same<T, phi::dtype::bfloat16>::value ||
      std::is_same<T, phi::dtype::float16>::value ||
      std::is_same<T, float>::value) {
    params.params.start.f = start.to<float>();
    params.params.limit.f = end.to<float>();
    params.params.delta.f = step.to<float>();
  } else {
    params.params.start.i = start.to<int>();
    params.params.limit.i = end.to<int>();
    params.params.delta.i = step.to<int>();
  }

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, RangeParams>("ArangeKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Range op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(arange_tensor,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeTensorKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          float) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(arange,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          float) {}
