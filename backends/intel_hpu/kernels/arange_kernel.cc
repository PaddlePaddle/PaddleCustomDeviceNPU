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
struct RangeParams {
  ns_RangeKernel::Params params;
};

class Range : public HpuOperator {
 public:
  Range() : HpuOperator("range") {}
  void AddNode(ConvertTensors& ct, RangeParams& params) {
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));

    std::string guid = guid_ + "_" + SynDataTypeToStr(outputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     syn_outputs.data(),
                                     0,
                                     1,
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

class RangeI64 : public HpuOperator {
 public:
  RangeI64() : HpuOperator("range") {}
  void AddNode(ConvertTensors& ct, RangeParams& params) {
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> range_out;
    range_out.push_back(createTensor(outputs[0].dims.size(),
                                     syn_type_int32,
                                     outputs[0].dims,
                                     false,
                                     "range_out"));
    std::string guid = guid_ + "_i32";
    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     range_out.data(),
                                     0,
                                     1,
                                     &params.params,
                                     sizeof(params.params),
                                     guid.c_str(),
                                     "Range",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = ", status);

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));
    std::string guid_cast = "cast_i32_to_i64";

    status = synNodeCreate(graphHandle_,
                           range_out.data(),
                           syn_outputs.data(),
                           range_out.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           guid_cast.c_str(),
                           "Cast",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (range/cast) failed = ",
             status);
  }
};

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  PADDLE_ENFORCE_NE(
      step,
      0,
      phi::errors::InvalidArgument("The step of range op should not be 0."));

  if (start < end) {
    PADDLE_ENFORCE_GT(
        step,
        0,
        phi::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }

  if (start > end) {
    PADDLE_ENFORCE_LT(step,
                      0,
                      phi::errors::InvalidArgument(
                          "The step should be less than 0 while start > end."));
  }

  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

void GetSize(phi::dtype::float16 start,
             phi::dtype::float16 end,
             phi::dtype::float16 step,
             int64_t* size) {
  PADDLE_ENFORCE_NE(static_cast<float>(step),
                    0,
                    phi::errors::InvalidArgument("The step of range op should "
                                                 "not be 0."));
  if (static_cast<float>(start) < static_cast<float>(end)) {
    PADDLE_ENFORCE_GT(
        static_cast<float>(step),
        0,
        phi::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }
  *size =
      std::ceil(std::abs((static_cast<float>(end) - static_cast<float>(start)) /
                         static_cast<float>(step)));
}

void GetSize(phi::dtype::bfloat16 start,
             phi::dtype::bfloat16 end,
             phi::dtype::bfloat16 step,
             int64_t* size) {
  PADDLE_ENFORCE_NE(static_cast<float>(step),
                    0,
                    phi::errors::InvalidArgument("The step of range op should "
                                                 "not be 0."));
  if (static_cast<float>(start) < static_cast<float>(end)) {
    PADDLE_ENFORCE_GT(
        static_cast<float>(step),
        0,
        phi::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }
  *size =
      std::ceil(std::abs((static_cast<float>(end) - static_cast<float>(start)) /
                         static_cast<float>(step)));
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const phi::DenseTensor& start_t,
                        const phi::DenseTensor& end_t,
                        const phi::DenseTensor& step_t,
                        phi::DenseTensor* out) {
  VLOG(6) << "call HPU ArangeTensorKernel";
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
  GetSize(start, end, step, &size);

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
    params.params.start.i = static_cast<int32_t>(start);
    params.params.limit.i = static_cast<int32_t>(end);
    params.params.delta.i = static_cast<int32_t>(step);
  }

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, RangeParams>("ArangeKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (std::is_same<T, int64_t>::value) {
      RangeI64 op;
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    } else {
      Range op;
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
void ArangeKernel(const Context& dev_ctx,
                  const phi::Scalar& start,
                  const phi::Scalar& end,
                  const phi::Scalar& step,
                  phi::DenseTensor* out) {
  VLOG(6) << "call HPU ArangeKernel";
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
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          float) {}
