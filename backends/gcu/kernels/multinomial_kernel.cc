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
void MultinomialKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::Scalar& num,
                       bool replacement,
                       phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("multinomial");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<int64_t>(out);

    // std::pair<uint64_t, uint64_t> seed_offset;
    // auto gen = dev_ctx.GetGenerator();
    // seed_offset.first = gen->GetCurrentSeed();
    // seed_offset.second = 0;
    // auto num_samples = num.to<int64_t>();
    // phi::DenseTensor output =
    //     MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    // LAUNCH_TOPSATENOP(topsatenMultinomial,
    //                   dev_ctx,
    //                   output,
    //                   x,
    //                   num_samples,
    //                   replacement,
    //                   seed_offset);
    // MaybeTransResult(dev_ctx, output, out);

    // Directly convert the input to call the CPU implementation.
    // Only float16 is supported, other data types will fallback to CPU.
    VLOG(6) << "[CPU_KERNEL] Call CPU kernel for multinomial(float16)";
    PADDLE_ENFORCE_EQ(
        x.dtype(),
        phi::DataType::FLOAT16,
        phi::errors::InvalidArgument("Only float16 is supported, but got % s.",
                                     phi::DataTypeToString(x.dtype()).c_str()));

    phi::DenseTensor x_gcu_f32;
    phi::DenseTensor x_cpu_f32;
    phi::DenseTensor out_cpu_int64;

    // convert input
    phi::DenseTensorMeta gcu_meta = x.meta();
    gcu_meta.dtype = phi::DataType::FLOAT32;
    x_gcu_f32.set_meta(gcu_meta);
    custom_kernel::Cast(dev_ctx, x, phi::DataType::FLOAT32, &x_gcu_f32);
    TensorCopy(dev_ctx, x_gcu_f32, false, &x_cpu_f32, phi::CPUPlace());

    // Wait for conversion
    dev_ctx.Wait();

    // call the CPU implementation
    phi::CPUContext dev_ctx_cpu;
    dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
    dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));
    dev_ctx_cpu.SetHostGenerator(dev_ctx.GetHostGenerator());
    out_cpu_int64.set_meta(out->meta());
    phi::MultinomialKernel<float, phi::CPUContext>(
        dev_ctx_cpu, x_cpu_f32, num, replacement, &out_cpu_int64);
    dev_ctx.Wait();

    // copy result
    TensorCopy(dev_ctx, out_cpu_int64, false, out);
    dev_ctx.Wait();

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multinomial,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MultinomialKernel,
                          //   float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
