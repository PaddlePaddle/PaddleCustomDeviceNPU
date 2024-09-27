// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void EinsumKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& inputs,
                  const std::string& equation,
                  phi::DenseTensor* out,
                  std::vector<phi::DenseTensor*> cache,
                  std::vector<phi::DenseTensor*> xshape UNUSED) {
  PADDLE_GCU_KERNEL_TRACE("einsum");
  if (LaunchAOTKernel()) {
    ContextPinnedGuard<Context> ctx_pinned_guard(dev_ctx);
    //   topsaten::topsatenEinSum is not support yet.
    //   dev_ctx.template Alloc<T>(out);
    //   auto out_tensor = CreateTopsatenTensor(*out);

    //   std::vector<topsatenTensor> in_tensors;
    //   for (const auto& in : inputs) {
    //     in_tensors.emplace_back(CreateTopsatenTensor(*in));
    //   }
    //   auto stream = static_cast<topsStream_t>(dev_ctx.stream());
    //   VLOG(3) << "EinsumKernel, use topsatenEinSum, input size:" <<
    //   in_tensors.size()
    //           << ", stream: " << stream;

    //   ATEN_OP_CALL_MAYBE_SYNC(topsaten::topsatenEinSum(out_tensor,
    //   in_tensors,
    //                                                    equation.c_str(),
    //                                                    stream),
    //                           dev_ctx);

    // Directly convert the input to call the CPU implementation.
    VLOG(6) << "[CPU_KERNEL] Call CPU kernel for einsum(float16)";
    PADDLE_ENFORCE_GE(
        inputs.size(),
        1,
        phi::errors::InvalidArgument(
            "Inputs size expected >= 1, but got %zu. Please check input value.",
            inputs.size()));
    // Only float16 is supported, other data types will fallback to CPU.
    PADDLE_ENFORCE_EQ(inputs[0]->dtype(),
                      phi::DataType::FLOAT16,
                      phi::errors::InvalidArgument(
                          "Only float16 is supported, but got %s.",
                          phi::DataTypeToString(inputs[0]->dtype()).c_str()));
    std::vector<phi::DenseTensor> inputs_gcu_tmp(inputs.size());
    std::vector<phi::DenseTensor> cache_gcu_tmp(cache.size());
    std::vector<phi::DenseTensor> inputs_cpu(inputs.size());
    std::vector<phi::DenseTensor> cache_cpu(cache.size());
    std::vector<phi::DenseTensor> cache_out_gcu(cache.size());

    std::vector<const phi::DenseTensor*> inputs_f32(inputs.size(), nullptr);
    std::vector<phi::DenseTensor*> cache_f32(cache.size(), nullptr);
    phi::DenseTensor out_cpu_f32;

    // convert inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] != nullptr) {
        phi::DenseTensorMeta gcu_meta(phi::DataType::FLOAT32,
                                      inputs[i]->dims());
        inputs_gcu_tmp[i].set_meta(gcu_meta);
        if (inputs[i]->initialized()) {
          custom_kernel::Cast(dev_ctx,
                              *(inputs[i]),
                              phi::DataType::FLOAT32,
                              &inputs_gcu_tmp[i]);
          TensorCopy(dev_ctx,
                     inputs_gcu_tmp[i],
                     false,
                     &inputs_cpu[i],
                     phi::CPUPlace());
        }
        inputs_f32[i] = &inputs_cpu[i];
      }
    }

    // convert cache
    for (size_t i = 0; i < cache.size(); ++i) {
      if (cache[i] != nullptr) {
        phi::DenseTensorMeta gcu_meta(phi::DataType::FLOAT32, cache[i]->dims());
        cache_gcu_tmp[i].set_meta(gcu_meta);
        if (cache[i]->initialized()) {
          custom_kernel::Cast(
              dev_ctx, *(cache[i]), phi::DataType::FLOAT32, &cache_gcu_tmp[i]);
          TensorCopy(
              dev_ctx, cache_gcu_tmp[i], false, &cache_cpu[i], phi::CPUPlace());
        }
        cache_f32[i] = &cache_cpu[i];
      }
    }

    // Wait for conversion
    dev_ctx.Wait();

    // call the CPU implementation
    phi::CPUContext dev_ctx_cpu;
    dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
    dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));
    phi::DenseTensorMeta cpu_meta(phi::DataType::FLOAT32, out->dims());
    out_cpu_f32.set_meta(cpu_meta);
    phi::EinsumKernel<float, phi::CPUContext>(
        dev_ctx_cpu, inputs_f32, equation, &out_cpu_f32, cache_f32, xshape);
    dev_ctx.Wait();

    // convert result
    phi::DenseTensor out_gcu_f32;
    TensorCopy(dev_ctx, out_cpu_f32, false, &out_gcu_f32);
    custom_kernel::Cast(dev_ctx, out_gcu_f32, phi::DataType::FLOAT16, out);

    // convert cache
    for (size_t i = 0; i < cache.size(); ++i) {
      if (cache[i] != nullptr && cache[i]->initialized()) {
        TensorCopy(dev_ctx, *(cache_f32[i]), false, &cache_out_gcu[i]);
        custom_kernel::Cast(
            dev_ctx, cache_out_gcu[i], phi::DataType::FLOAT16, cache[i]);
      }
    }
    dev_ctx.Wait();

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(einsum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::EinsumKernel,
                          //   float,
                          //   int32_t,
                          phi::dtype::float16) {}
