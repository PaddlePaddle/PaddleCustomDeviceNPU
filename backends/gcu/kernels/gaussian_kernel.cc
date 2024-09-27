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

template <typename T, typename Context>
void GaussianKernel(const Context& ctx,
                    const phi::IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("gaussian");

  if (LaunchAOTKernel()) {
    out->Resize(common::make_ddim(shape.GetData()));
    ctx.template Alloc<T>(out);
    double d_mean = static_cast<double>(mean);
    double d_std = static_cast<double>(std);

    std::pair<uint64_t, uint64_t> seed_offset;
    auto gen = ctx.GetGenerator();
    seed_offset.first = (seed != 0) ? seed : (gen->GetCurrentSeed());
    seed_offset.second = 0;

    LAUNCH_TOPSATENOP(topsatenNormal, ctx, *out, d_mean, d_std, seed_offset);

  } else {  // kernel impl base on JIT
    ContextPinnedGuard<Context> ctx_pinned_guard(ctx);
    VLOG(6) << "[HOST_KERNEL] Impl on host for gaussian";
    VLOG(6) << "Enter GaussianKernel with mean:" << mean << ", std:" << std
            << ", seed:" << seed << ", dtype:" << phi::DataTypeToString(dtype);
    ctx.template Alloc<T>(out);

    phi::DenseTensor cpu_tensor;
    phi::DenseTensorMeta cpu_meta = {out->dtype(), out->dims()};
    cpu_tensor.set_meta(cpu_meta);
    T* cpu_data = ctx.template HostAlloc<T>(&cpu_tensor);
    std::normal_distribution<typename phi::dtype::MPTypeTrait<T>::Type> dist(
        mean, std);

    int64_t size = out->numel();

    std::shared_ptr<std::mt19937_64> engine;
    if (seed) {
      engine = std::make_shared<std::mt19937_64>();
      engine->seed(seed);
    } else {
      engine = ctx.GetGenerator()->GetCPUEngine();
    }

    for (int64_t i = 0; i < size; ++i) {
      cpu_data[i] = static_cast<T>(dist(*engine));
    }
    TensorCopy(ctx, cpu_tensor, false, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gaussian,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GaussianKernel,
                          float,
                          phi::dtype::float16) {}
