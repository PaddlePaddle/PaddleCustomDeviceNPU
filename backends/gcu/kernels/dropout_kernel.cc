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
#include "kernels/funcs/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename Context>
inline void GetSeedDataAndIncrement(
    const Context& dev_ctx,
    const paddle::optional<phi::DenseTensor>& seed_tensor,
    const bool is_fix_seed,
    const int seed_val,
    const int offset,
    uint64_t* seed_data,
    uint64_t* increment) {
  auto gen_custom = dev_ctx.GetGenerator();
  if (seed_tensor) {
    phi::DenseTensor seed_cpu_tensor;
    TensorCopy(
        dev_ctx, seed_tensor.get(), true, &seed_cpu_tensor, phi::CustomPlace());
    *seed_data = static_cast<uint64_t>(seed_cpu_tensor.data<int>()[0]);
    *increment = offset;
  } else if (!is_fix_seed) {
    auto seed_offset = gen_custom->IncrementOffset(offset);
    *seed_data = seed_offset.first;
    *increment = seed_offset.second;
    VLOG(6) << "DefaultCustomDeviceGenerator status, seed:" << seed_offset.first
            << ", offset:" << seed_offset.second;
  } else {
    *seed_data = seed_val;
    *increment = offset;
  }
}

template <typename Context>
inline std::pair<uint64_t, uint64_t> GetSeedOffset(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& seed_tensor,
    int seed,
    bool fix_seed) {
  // Refer to the implementation of GPU dropout at:
  // paddle/phi/kernels/funcs/dropout_impl.cu.h
  int64_t x_numel = x.numel();
  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  topsDeviceProp_t prop = GetDeviceProp(device_id);
  size_t grid_size = prop.multiProcessorCount;
  size_t block_size = prop.maxThreadsPerBlock;
  size_t kVecSize = 4;
  int offset =
      ((x_numel - 1) / (grid_size * block_size * kVecSize) + 1) * kVecSize;
  uint64_t seed_data = 0;
  uint64_t increment = 0;
  GetSeedDataAndIncrement<Context>(
      dev_ctx, seed_tensor, fix_seed, seed, offset, &seed_data, &increment);
  return std::pair<uint64_t, uint64_t>(seed_data, increment);
}

template <typename T, typename Context>
void DropoutKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const paddle::optional<phi::DenseTensor>& seed_tensor,
                   const phi::Scalar& p,
                   bool is_test,
                   const std::string& mode,
                   int seed,
                   bool fix_seed,
                   phi::DenseTensor* out,
                   phi::DenseTensor* mask) {
  PADDLE_GCU_KERNEL_TRACE("dropout");
  dev_ctx.template Alloc<T>(out);
  if (mask) {
    dev_ctx.template Alloc<uint8_t>(mask);
  }

  if (LaunchAOTKernel()) {
    auto dropout_prob = p.to<double>();
    auto keep_scale = phi::Scalar(static_cast<float>(1.0 - dropout_prob));
    if (is_test) {
      if (mode == "upscale_in_train") {
        TensorCopy(dev_ctx, x, false, out);
      } else {
        LAUNCH_TOPSATENOP(topsatenMul, dev_ctx, *out, x, keep_scale);
      }
      return;
    }

    auto seed_offset =
        GetSeedOffset<Context>(dev_ctx, x, seed_tensor, seed, fix_seed);
    VLOG(6) << "DropoutKernel status, seed:" << seed_offset.first
            << ", offset:" << seed_offset.second;
    auto tmp_out = custom_kernel::TensorEmpty(dev_ctx, out->meta());
    LAUNCH_TOPSATENOP(topsatenNativeDropout,
                      dev_ctx,
                      *out,
                      tmp_out,
                      x,
                      dropout_prob,
                      !is_test,
                      seed_offset);
    if (mode != "upscale_in_train") {
      LAUNCH_TOPSATENOP(topsatenMul, dev_ctx, *out, *out, keep_scale);
    }

    if (!is_test && (mask != nullptr)) {
      custom_kernel::Cast(dev_ctx, tmp_out, mask->dtype(), mask);
    }

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};
    output_names["Mask"] = {"mask"};

    TensorValueMap outputs;
    outputs["Out"] = {out};
    outputs["Mask"] = {mask};

    auto dropout_prob = p.to<float>();
    auto seed_offset =
        GetSeedOffset<Context>(dev_ctx, x, seed_tensor, seed, fix_seed);
    int seed_data = seed_offset.first;

    GcuAttributeMap attrs;
    attrs["dropout_prob"] = dropout_prob;
    attrs["dropout_implementation"] = mode;
    attrs["seed"] = seed_data;
    attrs["is_test"] = is_test;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "dropout", dev_ctx);
  }
}

template <typename T, typename Context>
void DropoutGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& mask,
                       const phi::DenseTensor& dout,
                       const phi::Scalar& p,
                       bool is_test,
                       const std::string& mode,
                       phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("dropout_grad");
  dev_ctx.template Alloc<T>(dx);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Mask"] = {"mask"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["Mask"] = {const_cast<DenseTensor*>(&mask)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    auto dropout_prob = p.to<float>();

    GcuAttributeMap attrs;
    attrs["dropout_prob"] = dropout_prob;
    attrs["is_test"] = is_test;
    attrs["dropout_implementation"] = mode;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "dropout_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(dropout,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DropoutKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_PLUGIN_KERNEL(dropout_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DropoutGradKernel,
                          float,
                          phi::dtype::float16) {}
