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
#include "paddle/phi/common/amp_type_traits.h"

namespace custom_kernel {
template <typename T>
inline void UniformRealDistribution(T* data,
                                    const int64_t& size,
                                    const float& min,
                                    const float& max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  using UniformDataType = typename phi::dtype::MPTypeTrait<T>::Type;
  std::uniform_real_distribution<UniformDataType> dist(
      static_cast<UniformDataType>(min), static_cast<UniformDataType>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <typename T, typename Context>
void UniformRawKernel(const Context& dev_ctx,
                      const phi::IntArray& shape,
                      phi::DataType dtype,
                      const phi::Scalar& min,
                      const phi::Scalar& max,
                      int seed,
                      int diag_num,
                      int diag_step,
                      float diag_val,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("uniform_raw");
  ContextPinnedGuard<Context> ctx_pinned_guard(dev_ctx);
  VLOG(6) << "[HOST_KERNEL] Impl on host for uniform_raw";
  VLOG(6) << "Enter UniformRawKernel with min:" << min.ToString()
          << ", max:" << max.ToString() << ", seed:" << seed
          << ", diag_num:" << diag_num << ", diag_step:" << diag_step
          << ", diag_val:" << diag_val << ", shape:" << out->dims()
          << ", dtype:" << phi::DataTypeToString(dtype);
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  // 1. CPU implement
  phi::DenseTensor cpu_out;
  phi::DenseTensorMeta cpu_out_meta = {out->dtype(), out->dims()};
  cpu_out.set_meta(cpu_out_meta);
  T* cpu_data = dev_ctx.template HostAlloc<T>(&cpu_out);

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  UniformRealDistribution<T>(
      cpu_data, size, min.to<float>(), max.to<float>(), engine);

  if (diag_num > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        phi::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            size));
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      cpu_data[pos] = diag_val;
    }
  }

  // 2. CPU Copy to GCU
  TensorCopy(dev_ctx, cpu_out, false, out);
}

template <typename T, typename Context>
void UniformKernel(const Context& dev_ctx,
                   const phi::IntArray& shape,
                   phi::DataType dtype,
                   const phi::Scalar& min,
                   const phi::Scalar& max,
                   int seed,
                   phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("uniform");

  if (LaunchAOTKernel()) {
    out->Resize(common::make_ddim(shape.GetData()));
    dev_ctx.template Alloc<T>(out);
    double from = static_cast<double>(min.to<float>());
    double to = static_cast<double>(max.to<float>());

    std::pair<uint64_t, uint64_t> seed_offset;
    auto gen = dev_ctx.GetGenerator();
    seed_offset.first = (seed != 0) ? seed : (gen->GetCurrentSeed());
    seed_offset.second = 0;

    LAUNCH_TOPSATENOP(topsatenRngUniform, dev_ctx, *out, from, to, seed_offset);

  } else {  // kernel impl base on JIT
    VLOG(6) << "[HOST_KERNEL] Impl on host for uniform";
    custom_kernel::UniformRawKernel<T>(
        dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(uniform,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::UniformKernel,
                          float,
                          phi::dtype::float16) {}
