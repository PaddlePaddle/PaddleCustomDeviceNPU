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
namespace {
int64_t GridSampleInterpolationMode(const std::string& mode) {
  const std::unordered_map<std::string, int64_t> kInterpolationModeMap = {
      // aten  0:bilinear, 1:nearest, 2:bicubic
      {"bilinear", 0},
      {"nearest", 1},
      {"bicubic", 2},
  };
  if (kInterpolationModeMap.count(mode) == 0) {
    PADDLE_THROW(phi::errors::Unimplemented("Unsupported interpolation mode %s",
                                            mode.c_str()));
  }
  return kInterpolationModeMap.at(mode);
}

int64_t GridSamplePaddingMode(const std::string& mode) {
  const std::unordered_map<std::string, int64_t> kPaddingModeMap = {
      // aten  0:zero, 1:border, 2:reflect
      {"zeros", 0},
      {"border", 1},
      {"reflection", 2},
  };
  if (kPaddingModeMap.count(mode) == 0) {
    PADDLE_THROW(phi::errors::Unimplemented("Unsupported padding mode %s",
                                            mode.c_str()));
  }
  return kPaddingModeMap.at(mode);
}
}  // namespace

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("grid_sample");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    // Aten's support for topsatenGridSampler is not yet complete.
    THROW_AOT_UNIMPLEMENTED();

    int64_t interpolation_mode = GridSampleInterpolationMode(mode);
    int64_t outside_padding_mode = GridSamplePaddingMode(padding_mode);
    LAUNCH_TOPSATENOP(topsatenGridSampler,
                      dev_ctx,
                      *out,
                      x,
                      grid,
                      interpolation_mode,
                      outside_padding_mode,
                      align_corners);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Grid"] = {"grid"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Grid"] = {const_cast<DenseTensor*>(&grid)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["mode"] = mode;
    attrs["padding_mode"] = padding_mode;
    attrs["align_corners"] = align_corners;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "grid_sampler",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(grid_sample,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
