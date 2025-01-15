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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

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
  out->Resize(phi::make_ddim(shape.GetData()));
  VLOG(4) << out->dims();
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

  // 2. CPU Copy to NPU
  TensorCopy(dev_ctx, cpu_out, true, out);
}

template <typename T, typename Context>
void AclopUniformKernel(const Context& dev_ctx,
                        const phi::IntArray& shape,
                        phi::DataType dtype,
                        const phi::Scalar& min,
                        const phi::Scalar& max,
                        int seed,
                        phi::DenseTensor* out) {
  custom_kernel::UniformRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}
template <typename T, typename Context>
void UniformKernel(const Context& dev_ctx,
                   const phi::IntArray& shape,
                   phi::DataType dtype,
                   const phi::Scalar& min,
                   const phi::Scalar& max,
                   int seed,
                   phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnInplaceUniform,
                   (custom_kernel::AclopUniformKernel<T, Context>(
                       dev_ctx, shape, dtype, min, max, seed, out)));
  out->Resize(phi::make_ddim(shape.GetData()));
  dev_ctx.template Alloc<T>(out);
  uint64_t offset = 0;
  uint64_t seed_uint = seed;
  double from = min.to<double>();
  double to = max.to<double>();
  EXEC_NPU_CMD(aclnnInplaceUniform, dev_ctx, *out, from, to, seed_uint, offset);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    uniform_raw, npu, ALL_LAYOUT, custom_kernel::UniformRawKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    uniform, npu, ALL_LAYOUT, custom_kernel::UniformKernel, float) {}
