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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

const std::map<std::string, uint64_t> interpolationModeMap = {{"bilinear", 0},
                                                              {"nearest", 1}};

const std::map<std::string, uint64_t> paddlingModeMap = {
    {"zeros", 0}, {"border", 1}, {"reflection", 2}};

template <typename T, typename Context>
void AclopGridSampleKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& grid,
                           const std::string& mode,
                           const std::string& padding_mode,
                           bool align_corners,
                           phi::DenseTensor* out) {
  if (x.dims().size() == 4) {
    // GridSample2D
    NpuOpRunner grid_sample_runner;
    grid_sample_runner.SetType("GridSampler2D")
        .AddInput(x)
        .AddInput(grid)
        .AddAttr("interpolation_mode", mode)
        .AddAttr("padding_mode", padding_mode)
        .AddAttr("align_corners", align_corners)
        .AddOutput(*out)
        .Run(dev_ctx.stream());
  } else if (x.dims().size() == 5) {
    // GridSample3D
    NpuOpRunner grid_sample_runner;
    grid_sample_runner.SetType("GridSampler3D")
        .AddInput(x)
        .AddInput(grid)
        .AddAttr("interpolation_mode", mode)
        .AddAttr("padding_mode", padding_mode)
        .AddAttr("align_corners", align_corners)
        .AddOutput(*out)
        .Run(dev_ctx.stream());
  }
}

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      phi::DenseTensor* out) {
  // GridSample2D
  if (x.dims().size() == 4) {
    const int n = grid.dims()[0];
    const int out_h = grid.dims()[1];
    const int out_w = grid.dims()[2];
    const int c = x.dims()[1];

    out->Resize(phi::make_ddim({n, c, out_h, out_w}));
    dev_ctx.template Alloc<T>(out);
    DO_COMPATIBILITY(
        aclnnGridSampler2D,
        (custom_kernel::AclopGridSampleKernel<T, Context>(
            dev_ctx, x, grid, mode, padding_mode, align_corners, out)));
    uint64_t mode_to_int, paddle_mode_to_int;
    auto find_it = interpolationModeMap.find(mode);
    if (find_it != interpolationModeMap.end()) {
      mode_to_int = find_it->second;
    }
    find_it = paddlingModeMap.find(padding_mode);
    if (find_it != paddlingModeMap.end()) {
      paddle_mode_to_int = find_it->second;
    }
    EXEC_NPU_CMD(aclnnGridSampler2D,
                 dev_ctx,
                 x,
                 grid,
                 mode_to_int,
                 paddle_mode_to_int,
                 align_corners,
                 *out);
  } else if (x.dims().size() == 5) {
    // GridSample3D
    const int n = grid.dims()[0];
    const int out_d = grid.dims()[1];
    const int out_h = grid.dims()[2];
    const int out_w = grid.dims()[3];
    const int c = x.dims()[1];

    out->Resize(phi::make_ddim({n, c, out_d, out_h, out_w}));
    dev_ctx.template Alloc<T>(out);
    DO_COMPATIBILITY(
        aclnnGridSampler3D,
        (custom_kernel::AclopGridSampleKernel<T, Context>(
            dev_ctx, x, grid, mode, padding_mode, align_corners, out)));
    uint64_t mode_to_int, paddle_mode_to_int;
    auto find_it = interpolationModeMap.find(mode);
    if (find_it != interpolationModeMap.end()) {
      mode_to_int = find_it->second;
    }
    find_it = paddlingModeMap.find(padding_mode);
    if (find_it != paddlingModeMap.end()) {
      paddle_mode_to_int = find_it->second;
    }
    EXEC_NPU_CMD(aclnnGridSampler3D,
                 dev_ctx,
                 x,
                 grid,
                 mode_to_int,
                 paddle_mode_to_int,
                 align_corners,
                 *out);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "the input x must be 4D/5D tensor of grid_sample, but got [%d]D tensor",
        x.dims().size()));
  }
}

template <typename T, typename Context>
void AclopGridSampleGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& grid,
                               const phi::DenseTensor& out_grad,
                               const std::string& mode,
                               const std::string& padding_mode,
                               bool align_corners,
                               phi::DenseTensor* x_grad,
                               phi::DenseTensor* grid_grad) {
  if (x.dims().size() == 4) {
    const int n = grid.dims()[0];
    const int out_h = grid.dims()[1];
    const int out_w = grid.dims()[2];
    const int c = x.dims()[1];
    const int in_h = x.dims()[2];
    const int in_w = x.dims()[3];

    x_grad->Resize({n, c, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(grid_grad);

      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler2DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(*grid_grad)
          .Run(dev_ctx.stream());
    } else {
      phi::DenseTensor tmp_grid_grad;
      tmp_grid_grad.Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(&tmp_grid_grad);
      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler2DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(tmp_grid_grad)
          .Run(dev_ctx.stream());
    }
  } else if (x.dims().size() == 5) {
    const int n = grid.dims()[0];
    const int out_d = grid.dims()[1];
    const int out_h = grid.dims()[2];
    const int out_w = grid.dims()[3];
    const int c = x.dims()[1];
    const int in_d = x.dims()[2];
    const int in_h = x.dims()[3];
    const int in_w = x.dims()[4];

    x_grad->Resize({n, c, in_d, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_d, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(grid_grad);

      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler3DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(*grid_grad)
          .Run(dev_ctx.stream());
    } else {
      phi::DenseTensor tmp_grid_grad;
      tmp_grid_grad.Resize({n, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(&tmp_grid_grad);
      NpuOpRunner grid_sample_runner;
      grid_sample_runner.SetType("GridSampler3DGrad")
          .AddInput(out_grad)
          .AddInput(x)
          .AddInput(grid)
          .AddAttr("interpolation_mode", mode)
          .AddAttr("padding_mode", padding_mode)
          .AddAttr("align_corners", align_corners)
          .AddOutput(*x_grad)
          .AddOutput(tmp_grid_grad)
          .Run(dev_ctx.stream());
    }
  }
}

template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& grid,
                          const phi::DenseTensor& out_grad,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          phi::DenseTensor* x_grad,
                          phi::DenseTensor* grid_grad) {
  std::array<bool, 2> output_mask = {true, true};
  if (x.dims().size() == 4) {
    DO_COMPATIBILITY(
        aclnnGridSampler2DBackward,
        (custom_kernel::AclopGridSampleGradKernel<T, Context>(dev_ctx,
                                                              x,
                                                              grid,
                                                              out_grad,
                                                              mode,
                                                              padding_mode,
                                                              align_corners,
                                                              x_grad,
                                                              grid_grad)));
    const int n = grid.dims()[0];
    const int out_h = grid.dims()[1];
    const int out_w = grid.dims()[2];
    const int c = x.dims()[1];
    const int in_h = x.dims()[2];
    const int in_w = x.dims()[3];

    x_grad->Resize({n, c, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(grid_grad);

      uint64_t mode_to_int, paddle_mode_to_int;
      auto find_it = interpolationModeMap.find(mode);
      if (find_it != interpolationModeMap.end()) {
        mode_to_int = find_it->second;
      }
      find_it = paddlingModeMap.find(padding_mode);
      if (find_it != paddlingModeMap.end()) {
        paddle_mode_to_int = find_it->second;
      }
      EXEC_NPU_CMD(aclnnGridSampler2DBackward,
                   dev_ctx,
                   out_grad,
                   x,
                   grid,
                   mode_to_int,
                   paddle_mode_to_int,
                   align_corners,
                   output_mask,
                   *x_grad,
                   *grid_grad);
    } else {
      phi::DenseTensor tmp_grid_grad;
      tmp_grid_grad.Resize({n, out_h, out_w, 2});
      dev_ctx.template Alloc<T>(&tmp_grid_grad);
      uint64_t mode_to_int, paddle_mode_to_int;
      auto find_it = interpolationModeMap.find(mode);
      if (find_it != interpolationModeMap.end()) {
        mode_to_int = find_it->second;
      }
      find_it = paddlingModeMap.find(padding_mode);
      if (find_it != paddlingModeMap.end()) {
        paddle_mode_to_int = find_it->second;
      }
      EXEC_NPU_CMD(aclnnGridSampler2DBackward,
                   dev_ctx,
                   out_grad,
                   x,
                   grid,
                   mode_to_int,
                   paddle_mode_to_int,
                   align_corners,
                   output_mask,
                   *x_grad,
                   tmp_grid_grad);
    }
  } else if (x.dims().size() == 5) {
    DO_COMPATIBILITY(
        aclnnGridSampler3DBackward,
        (custom_kernel::AclopGridSampleGradKernel<T, Context>(dev_ctx,
                                                              x,
                                                              grid,
                                                              out_grad,
                                                              mode,
                                                              padding_mode,
                                                              align_corners,
                                                              x_grad,
                                                              grid_grad)));
    const int n = grid.dims()[0];
    const int out_d = grid.dims()[1];
    const int out_h = grid.dims()[2];
    const int out_w = grid.dims()[3];
    const int c = x.dims()[1];
    const int in_d = x.dims()[2];
    const int in_h = x.dims()[3];
    const int in_w = x.dims()[4];

    x_grad->Resize({n, c, in_d, in_h, in_w});
    dev_ctx.template Alloc<T>(x_grad);

    if (grid_grad != nullptr) {
      grid_grad->Resize({n, out_d, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(grid_grad);

      uint64_t mode_to_int, paddle_mode_to_int;
      auto find_it = interpolationModeMap.find(mode);
      if (find_it != interpolationModeMap.end()) {
        mode_to_int = find_it->second;
      }
      find_it = paddlingModeMap.find(padding_mode);
      if (find_it != paddlingModeMap.end()) {
        paddle_mode_to_int = find_it->second;
      }
      EXEC_NPU_CMD(aclnnGridSamplerd3DBackward,
                   dev_ctx,
                   out_grad,
                   x,
                   grid,
                   mode_to_int,
                   paddle_mode_to_int,
                   align_corners,
                   output_mask,
                   *x_grad,
                   *grid_grad);
    } else {
      phi::DenseTensor tmp_grid_grad;
      tmp_grid_grad.Resize({n, out_h, out_w, 3});
      dev_ctx.template Alloc<T>(&tmp_grid_grad);
      uint64_t mode_to_int, paddle_mode_to_int;
      auto find_it = interpolationModeMap.find(mode);
      if (find_it != interpolationModeMap.end()) {
        mode_to_int = find_it->second;
      }
      find_it = paddlingModeMap.find(padding_mode);
      if (find_it != paddlingModeMap.end()) {
        paddle_mode_to_int = find_it->second;
      }
      EXEC_NPU_CMD(aclnnGridSamplerd3DBackward,
                   dev_ctx,
                   out_grad,
                   x,
                   grid,
                   mode_to_int,
                   paddle_mode_to_int,
                   align_corners,
                   output_mask,
                   *x_grad,
                   tmp_grid_grad);
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "the input x must be 4D/5D tensor of grid_sample, but got [%d]D tensor",
        x.dims().size()));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(grid_sample,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(grid_sample_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleGradKernel,
                          float,
                          phi::dtype::float16) {}
