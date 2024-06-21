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
#include <vector>

#include "funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T>
inline void UniformRealDistribution(T* data,
                                    const int64_t& size,
                                    const float& min,
                                    const float& max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <typename T, typename Context>
void UniformRandomRawKernel(const Context& dev_ctx,
                            const phi::IntArray& shape,
                            phi::DataType dtype,
                            const phi::Scalar& min,
                            const phi::Scalar& max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA UniformRandomRawKernel";
  out->Resize(phi::make_ddim(shape.GetData()));
  VLOG(4) << out->dims();
  T* data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  // 1.CPU implement
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
  // 2. CPU copy to SDAA
  phi::Copy(dev_ctx, cpu_out, out->place(), false, out);
}

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const phi::IntArray& shape,
                         phi::DataType dtype,
                         const phi::Scalar& min,
                         const phi::Scalar& max,
                         int seed,
                         phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA UniformRandomKernel";
  custom_kernel::UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(uniform_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::UniformRandomRawKernel,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(
    uniform, sdaa, ALL_LAYOUT, custom_kernel::UniformRandomKernel, float) {}
