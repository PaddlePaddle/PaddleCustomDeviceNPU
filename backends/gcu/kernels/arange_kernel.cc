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

template <typename T, typename Context>
void ArangeKernel(const Context& dev_ctx,
                  const phi::Scalar& start,
                  const phi::Scalar& end,
                  const phi::Scalar& step,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("arange");
  //   VLOG(6) << "[HOST_KERNEL] Impl on host for arange";
  T start_value = start.to<T>();
  T end_value = end.to<T>();
  T step_value = step.to<T>();
  int64_t size = 0;
  GetSize(start_value, end_value, step_value, &size);
  out->Resize(phi::make_ddim({size}));
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor output_t =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    topsatenTensor output = CreateTopsatenTensor(output_t);
    topsatenScalar_t start_s = ScalarToTopsatenScalar(start);
    topsatenScalar_t end_s = ScalarToTopsatenScalar(end);
    topsatenScalar_t step_s = ScalarToTopsatenScalar(step);
    topsatenDataType_t d_type = DataTypeToTopsatenDataType(output_t.dtype());
    topsatenLayoutType_t layout = TOPSATEN_LAYOUT_STRIDED;
    // pinMemory only support false
    bool use_pin_mem = false;
    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "topsatenArange", output_t, start, end, step, output_t.dtype());
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenArange,
                                        dev_ctx,
                                        abstract_info,
                                        output,
                                        start_s,
                                        end_s,
                                        step_s,
                                        d_type,
                                        layout,
                                        use_pin_mem);
    MaybeTransResult(dev_ctx, output_t, out);

  } else {  // kernel impl base on JIT
    VLOG(6) << "[HOST_KERNEL] Impl on host for arange";
    std::vector<T, PinnedAllocatorForSTL<T>> odata;
    T value = start_value;
    for (int64_t i = 0; i < size; ++i) {
      odata.push_back(value);
      value += step_value;
    }

    TensorFromVector(dev_ctx, odata, dev_ctx, out);
    //   dev_ctx.Wait();
  }
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const phi::DenseTensor& start_t,
                        const phi::DenseTensor& end_t,
                        const phi::DenseTensor& step_t,
                        phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("arange_tensor");
  //   VLOG(6) << "[HOST_KERNEL] Impl on host for arange_tensor";
  phi::DenseTensor n;
  n.Resize(start_t.dims());
  T* n_data = dev_ctx.template HostAlloc<T>(&n);

  TensorCopy(dev_ctx, start_t, true, &n, phi::CPUPlace());
  T start = n_data[0];

  TensorCopy(dev_ctx, end_t, true, &n, phi::CPUPlace());
  T end = n_data[0];

  TensorCopy(dev_ctx, step_t, true, &n, phi::CPUPlace());
  T step = n_data[0];

  custom_kernel::ArangeKernel<T, Context>(
      dev_ctx, phi::Scalar(start), phi::Scalar(end), phi::Scalar(step), out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(arange_tensor,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeTensorKernel,
                          int,
                          int64_t,
                          float,
                          double) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(arange,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ArangeKernel,
                          int,
                          int64_t,
                          float,
                          double) {}
