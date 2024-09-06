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

template <typename Context>
void ReduceMean(const Context& dev_ctx,
                const phi::DenseTensor* x,
                phi::DenseTensor* y,
                const std::vector<int>& dim,
                bool keep_dims = true) {
  NpuOpRunner runner;
  runner.SetType("ReduceMean")
      .AddInput(*x)
      .AddInput(dev_ctx, std::move(dim))
      .AddOutput(*y)
      .AddAttrs({{"keep_dims", keep_dims}})
      .Run(dev_ctx.stream());
}

template <typename Context>
void ReduceSum(const Context& dev_ctx,
               const phi::DenseTensor* x,
               phi::DenseTensor* y,
               const std::vector<int>& dim,
               bool keep_dims = true) {
  NpuOpRunner runner;
  runner.SetType("ReduceSum")
      .AddInput(*x)
      .AddInput(dev_ctx, std::move(dim))
      .AddOutput(*y)
      .AddAttrs({{"keep_dims", keep_dims}})
      .Run(dev_ctx.stream());
}

template <typename Context>
void Add(const Context& dev_ctx,
         const phi::DenseTensor* x,
         const phi::DenseTensor* y,
         phi::DenseTensor* z) {
  const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
  runner.Run(dev_ctx.stream());
}

template <typename Context>
void Sub(const Context& dev_ctx,
         const phi::DenseTensor* x,
         const phi::DenseTensor* y,
         phi::DenseTensor* z) {
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(dev_ctx.stream());
}

template <typename Context>
void Mul(const Context& dev_ctx,
         const phi::DenseTensor* x,
         const phi::DenseTensor* y,
         phi::DenseTensor* z) {
  const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
  runner.Run(dev_ctx.stream());
}

template <typename Context>
void Div(const Context& dev_ctx,
         const phi::DenseTensor* x,
         const phi::DenseTensor* y,
         phi::DenseTensor* z) {
  const auto& runner = NpuOpRunner("Div", {*x, *y}, {*z}, {});
  runner.Run(dev_ctx.stream());
}

template <typename Context>
void DivNoNan(const Context& dev_ctx,
              const phi::DenseTensor* x,
              const phi::DenseTensor* y,
              phi::DenseTensor* z) {
  const auto& runner = NpuOpRunner("DivNoNan", {*x, *y}, {*z}, {});
  runner.Run(dev_ctx.stream());
}

template <typename Context>
void Adds(const Context& dev_ctx,
          const phi::DenseTensor* x,
          float scalar,
          phi::DenseTensor* y) {
  const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
  runner.Run(dev_ctx.stream());
}

template <typename Context>
void Transpose(const Context& dev_ctx,
               const phi::DenseTensor* x,
               phi::DenseTensor* y,
               const std::vector<int>& axis) {
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(*x)
      .AddInput(dev_ctx, std::move(axis))
      .AddOutput(*y)
      .Run(dev_ctx.stream());
}

template <typename Context>
void Sqrt(const Context& dev_ctx,
          const phi::DenseTensor* x,
          phi::DenseTensor* y) {
  const auto& runner = NpuOpRunner("Sqrt", {*x}, {*y}, {});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
phi::DenseTensor ReduceMeanToNG(const Context& dev_ctx,
                                const phi::DenseTensor* x,
                                const phi::DataLayout& data_layout,
                                const int64_t N,
                                const int64_t C,
                                const int64_t L,
                                const int G) {
  phi::DenseTensor y;
  if (data_layout == phi::DataLayout::kNCHW) {
    phi::DenseTensorMeta y_meta = {x->dtype(), {N, G, 1}};
    y.set_meta(y_meta);
    dev_ctx.template Alloc<T>(&y);
    //  shape of x is [N, G, C*H*W/G]
    ReduceMean<Context>(dev_ctx, x, &y, std::vector<int>{2});
  } else {
    phi::DenseTensorMeta y_meta = {x->dtype(), {N, 1, G}};
    y.set_meta(y_meta);
    dev_ctx.template Alloc<T>(&y);
    //  shape of x is [N, C*H*W/G, G]
    phi::DenseTensor x_trans;
    phi::DenseTensorMeta x_trans_meta = {x->dtype(), {N, G, C * L / G}};
    x_trans.set_meta(x_trans_meta);
    dev_ctx.template Alloc<T>(&x_trans);
    Transpose<Context>(dev_ctx, x, &x_trans, std::vector<int>{0, 2, 1});
    ReduceMean<Context>(dev_ctx, &x_trans, &y, std::vector<int>{2});
  }
  return y;
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void AclopGroupNormKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const paddle::optional<phi::DenseTensor>& scale,
                          const paddle::optional<phi::DenseTensor>& bias,
                          float epsilon,
                          int groups,
                          const std::string& data_layout,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean,
                          phi::DenseTensor* variance) {
  auto x_dims = phi::vectorize(x.dims());
  const phi::DataLayout data_layout_data =
      common::StringToDataLayout(data_layout);

  if (x_dims.size() > 3) {
    phi::DenseTensor x_tmp(x);
    std::vector<int64_t> x_tmp_dims;
    if (data_layout_data == phi::DataLayout::kNCHW) {
      x_tmp_dims = {x_dims[0],
                    x_dims[1],
                    std::accumulate(x_dims.cbegin() + 2,
                                    x_dims.cend(),
                                    1,
                                    std::multiplies<int64_t>())};
    } else {
      x_tmp_dims = {x_dims[0],
                    std::accumulate(x_dims.cbegin() + 1,
                                    x_dims.cend() - 1,
                                    1,
                                    std::multiplies<int64_t>()),
                    x_dims.back()};
    }
    x_tmp.Resize(phi::make_ddim(x_tmp_dims));
    custom_kernel::AclopGroupNormKernel<T, Context>(dev_ctx,
                                                    x_tmp,
                                                    scale,
                                                    bias,
                                                    epsilon,
                                                    groups,
                                                    data_layout,
                                                    y,
                                                    mean,
                                                    variance);
    y->Resize(x.dims());
    return;
  }

  auto stream = dev_ctx.stream();

  phi::DenseTensor xnorm;
  phi::DenseTensorMeta xnorm_meta = {x.dtype(), x.dims()};
  xnorm.set_meta(xnorm_meta);
  dev_ctx.template Alloc<T>(&xnorm);
  if (data_layout_data != phi::DataLayout::kNCHW) {
    xnorm.Resize({x.dims()[0], x.dims()[2], x.dims()[1]});
    Transpose<Context>(dev_ctx, &x, &xnorm, std::vector<int>{0, 2, 1});
  } else {
    TensorCopy(dev_ctx, x, false, &xnorm, dev_ctx.GetPlace());
  }
  auto N = xnorm.dims()[0];
  auto C = xnorm.dims()[1];
  auto L = xnorm.dims()[2];
  xnorm.Resize({N * groups, C * L / groups});
  std::vector<int> axis = {1};
  auto reduce_dim = mean->dims();

  dev_ctx.template Alloc<T>(y);
  mean->Resize({N * groups, 1});
  dev_ctx.template Alloc<T>(mean);
  variance->Resize({N * groups, 1});
  dev_ctx.template Alloc<T>(variance);

  // mean
  ReduceMean<Context>(dev_ctx, &xnorm, mean, axis);
  // x - mean
  Sub<Context>(dev_ctx, &xnorm, mean, &xnorm);

  // var
  phi::DenseTensor sqr;
  phi::DenseTensorMeta sqr_meta = {x.dtype(), xnorm.dims()};
  sqr.set_meta(sqr_meta);
  dev_ctx.template Alloc<T>(&sqr);
  Mul<Context>(dev_ctx, &xnorm, &xnorm, &sqr);
  ReduceMean<Context>(dev_ctx, &sqr, variance, axis);

  // sqrt(var + epsilon)
  phi::DenseTensor std;
  phi::DenseTensorMeta std_meta = {x.dtype(), variance->dims()};
  std.set_meta(std_meta);
  dev_ctx.template Alloc<T>(&std);
  Adds<Context>(dev_ctx, variance, epsilon, &std);
  Sqrt<Context>(dev_ctx, &std, &std);

  y->Resize(xnorm.dims());
  Div<Context>(dev_ctx, &xnorm, &std, y);
  y->Resize({N, C, L});

  if (scale) {
    phi::DenseTensor scale_tensor = scale.get();
    phi::DenseTensor scale_t(scale_tensor);
    scale_t.Resize({1, C, 1});
    Mul<Context>(dev_ctx, y, &scale_t, y);
  }
  if (bias) {
    phi::DenseTensor bias_tensor = bias.get();
    phi::DenseTensor bias_t(bias_tensor);
    bias_t.Resize({1, C, 1});
    Add<Context>(dev_ctx, y, &bias_t, y);
  }
  if (data_layout_data != phi::DataLayout::kNCHW) {
    phi::DenseTensor y_out;
    phi::DenseTensorMeta y_out_meta = {
        y->dtype(), {y->dims()[0], y->dims()[2], y->dims()[1]}};
    y_out.set_meta(y_out_meta);
    dev_ctx.template Alloc<T>(&y_out);
    Transpose<Context>(dev_ctx, y, &y_out, std::vector<int>{0, 2, 1});
    // assign y_out to y
    Transpose<Context>(dev_ctx, &y_out, y, std::vector<int>{0, 1, 2});
    y->Resize(x.dims());
  }
  mean->Resize(reduce_dim);
  variance->Resize(reduce_dim);
}

template <typename T, typename Context>
void AclnnGroupNormKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const paddle::optional<phi::DenseTensor>& scale,
                          const paddle::optional<phi::DenseTensor>& bias,
                          float epsilon,
                          int groups,
                          const std::string& data_layout,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean,
                          phi::DenseTensor* variance) {
  auto x_dims = phi::vectorize(x.dims());
  const phi::DataLayout data_layout_data =
      common::StringToDataLayout(data_layout);

  if (x_dims.size() > 3) {
    phi::DenseTensor x_tmp(x);
    std::vector<int64_t> x_tmp_dims;
    if (data_layout_data == phi::DataLayout::kNCHW) {
      x_tmp_dims = {x_dims[0],
                    x_dims[1],
                    std::accumulate(x_dims.cbegin() + 2,
                                    x_dims.cend(),
                                    1,
                                    std::multiplies<int64_t>())};
    } else {
      x_tmp_dims = {x_dims[0],
                    std::accumulate(x_dims.cbegin() + 1,
                                    x_dims.cend() - 1,
                                    1,
                                    std::multiplies<int64_t>()),
                    x_dims.back()};
    }
    x_tmp.Resize(phi::make_ddim(x_tmp_dims));
    custom_kernel::AclnnGroupNormKernel<T, Context>(dev_ctx,
                                                    x_tmp,
                                                    scale,
                                                    bias,
                                                    epsilon,
                                                    groups,
                                                    data_layout,
                                                    y,
                                                    mean,
                                                    variance);
    y->Resize(x.dims());
    return;
  }

  auto stream = dev_ctx.stream();

  phi::DenseTensor xnorm;
  phi::DenseTensorMeta xnorm_meta = {x.dtype(), x.dims()};
  xnorm.set_meta(xnorm_meta);
  dev_ctx.template Alloc<T>(&xnorm);
  if (data_layout_data != phi::DataLayout::kNCHW) {
    xnorm.Resize({x.dims()[0], x.dims()[2], x.dims()[1]});
    Transpose<Context>(dev_ctx, &x, &xnorm, std::vector<int>{0, 2, 1});
  } else {
    TensorCopy(dev_ctx, x, false, &xnorm, dev_ctx.GetPlace());
  }
  auto N = xnorm.dims()[0];
  auto C = xnorm.dims()[1];
  auto L = xnorm.dims()[2];
  auto reduce_dim = mean->dims();
  y->Resize(xnorm.dims());
  dev_ctx.template Alloc<T>(y);
  mean->Resize({N, groups});
  dev_ctx.template Alloc<T>(mean);
  variance->Resize({N, groups});
  dev_ctx.template Alloc<T>(variance);

  phi::DenseTensor scale_tensor, bias_tensor;
  if (scale) {
    scale_tensor = scale.get();
  } else {
    scale_tensor.Resize({C});
    FillNpuTensorWithConstant<T>(&scale_tensor, dev_ctx, static_cast<T>(1));
  }
  if (bias) {
    bias_tensor = bias.get();
  } else {
    bias_tensor.Resize({C});
    FillNpuTensorWithConstant<T>(&bias_tensor, dev_ctx, static_cast<T>(0));
  }
  double eps = static_cast<double>(epsilon);
  EXEC_NPU_CMD(aclnnGroupNorm,
               dev_ctx,
               xnorm,
               scale_tensor,
               bias_tensor,
               N,
               C,
               L,
               groups,
               eps,
               *y,
               *mean,
               *variance);  // variance: (N, group)

  const auto& inv_runner = NpuOpRunner("Inv", {*variance}, {*variance}, {});
  inv_runner.Run(stream);
  custom_kernel::SquareKernel<T, Context>(dev_ctx, *variance, variance);

  phi::DenseTensor eps_tensor;
  phi::DenseTensorMeta eps_tensor_meta = {variance->dtype(), variance->dims()};
  eps_tensor.set_meta(eps_tensor_meta);
  dev_ctx.template Alloc<T>(&eps_tensor);

  phi::DenseTensor ones_tensor;
  phi::DenseTensorMeta ones_tensor_meta = {variance->dtype(), variance->dims()};
  ones_tensor.set_meta(ones_tensor_meta);
  dev_ctx.template Alloc<T>(&ones_tensor);

  const auto& runner_one_dx =
      NpuOpRunner("OnesLike", {*variance}, {ones_tensor}, {});
  runner_one_dx.Run(stream);

  const auto& muls_runner =
      NpuOpRunner("Muls", {ones_tensor}, {eps_tensor}, {{"value", epsilon}});
  muls_runner.Run(stream);

  custom_kernel::SubtractKernel<T, Context>(
      dev_ctx, *variance, eps_tensor, variance);
  if (data_layout_data != phi::DataLayout::kNCHW) {
    phi::DenseTensor y_out;
    phi::DenseTensorMeta y_out_meta = {
        y->dtype(), {y->dims()[0], y->dims()[2], y->dims()[1]}};
    y_out.set_meta(y_out_meta);
    dev_ctx.template Alloc<T>(&y_out);
    Transpose<Context>(dev_ctx, y, &y_out, std::vector<int>{0, 2, 1});
    // assign y_out to y
    Transpose<Context>(dev_ctx, &y_out, y, std::vector<int>{0, 1, 2});
    y->Resize(x.dims());
  }
  mean->Resize(reduce_dim);
  variance->Resize(reduce_dim);
}

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const paddle::optional<phi::DenseTensor>& scale,
                     const paddle::optional<phi::DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean,
                     phi::DenseTensor* variance) {
  DO_COMPATIBILITY(aclnnGroupNorm,
                   (custom_kernel::AclopGroupNormKernel<T, Context>(dev_ctx,
                                                                    x,
                                                                    scale,
                                                                    bias,
                                                                    epsilon,
                                                                    groups,
                                                                    data_layout,
                                                                    y,
                                                                    mean,
                                                                    variance)));
  AclnnGroupNormKernel<T, Context>(
      dev_ctx, x, scale, bias, epsilon, groups, data_layout, y, mean, variance);
}

template <typename T, typename Context>
void GroupNormGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const paddle::optional<phi::DenseTensor>& scale,
                         const paddle::optional<phi::DenseTensor>& bias,
                         const phi::DenseTensor& y,
                         const phi::DenseTensor& mean,
                         const phi::DenseTensor& variance,
                         const phi::DenseTensor& d_y,
                         float epsilon,
                         int groups,
                         const std::string& data_layout,
                         phi::DenseTensor* d_x,
                         phi::DenseTensor* d_scale,
                         phi::DenseTensor* d_bias) {
  auto x_dims = phi::vectorize(x.dims());
  const phi::DataLayout data_layout_data = StringToDataLayout(data_layout);

  if (x_dims.size() > 3) {
    phi::DenseTensor x_tmp(x), y_tmp(y), dy_tmp(d_y);
    std::vector<int64_t> x_tmp_dims;
    if (data_layout_data == phi::DataLayout::kNCHW) {
      x_tmp_dims = {x_dims[0],
                    x_dims[1],
                    std::accumulate(x_dims.cbegin() + 2,
                                    x_dims.cend(),
                                    1,
                                    std::multiplies<int64_t>())};
    } else {
      x_tmp_dims = {x_dims[0],
                    std::accumulate(x_dims.cbegin() + 1,
                                    x_dims.cend() - 1,
                                    1,
                                    std::multiplies<int64_t>()),
                    x_dims.back()};
    }
    x_tmp.Resize(phi::make_ddim(x_tmp_dims));
    y_tmp.Resize(phi::make_ddim(x_tmp_dims));
    dy_tmp.Resize(phi::make_ddim(x_tmp_dims));
    custom_kernel::GroupNormGradKernel<T, Context>(dev_ctx,
                                                   x_tmp,
                                                   scale,
                                                   bias,
                                                   y_tmp,
                                                   mean,
                                                   variance,
                                                   dy_tmp,
                                                   epsilon,
                                                   groups,
                                                   data_layout,
                                                   d_x,
                                                   d_scale,
                                                   d_bias);
    d_x->Resize(x.dims());
    return;
  }

  auto stream = dev_ctx.stream();

  phi::DenseTensor xnorm;
  phi::DenseTensorMeta xnorm_meta = {y.dtype(), y.dims()};
  xnorm.set_meta(xnorm_meta);
  dev_ctx.template Alloc<T>(&xnorm);

  auto scale_tensor = scale.get();
  auto bias_tensor = bias.get();
  phi::DenseTensor scale_share(scale_tensor);
  phi::DenseTensor bias_share(bias_tensor);

  int64_t N = y.dims()[0];
  int64_t C, L;
  phi::DDim scale_bias_dim;
  if (data_layout_data == phi::DataLayout::kNCHW) {
    C = y.dims()[1];
    L = y.dims()[2];
    scale_bias_dim = phi::make_ddim({1, C, 1});
  } else {
    C = y.dims()[2];
    L = y.dims()[1];
    scale_bias_dim = phi::make_ddim({1, 1, C});
  }
  scale_share.Resize(scale_bias_dim);
  bias_share.Resize(scale_bias_dim);

  Sub<Context>(dev_ctx, &y, &bias_share, &xnorm);
  DivNoNan<Context>(dev_ctx, &xnorm, &scale_share, &xnorm);

  if (d_bias) {
    dev_ctx.template Alloc<T>(d_bias);
    if (data_layout_data == phi::DataLayout::kNCHW) {
      ReduceSum<Context>(dev_ctx, &d_y, d_bias, std::vector<int>{0, 2}, false);
    } else {
      ReduceSum<Context>(dev_ctx, &d_y, d_bias, std::vector<int>{0, 1}, false);
    }
  }
  if (d_scale) {
    dev_ctx.template Alloc<T>(d_scale);
    phi::DenseTensor dy_xnorm;
    phi::DenseTensorMeta dy_xnorm_meta = {y.dtype(), d_y.dims()};
    dy_xnorm.set_meta(dy_xnorm_meta);
    dev_ctx.template Alloc<T>(&dy_xnorm);
    Mul<Context>(dev_ctx, &d_y, &xnorm, &dy_xnorm);
    if (data_layout_data == phi::DataLayout::kNCHW) {
      ReduceSum<Context>(dev_ctx, &dy_xnorm, d_scale, std::vector<int>{0, 2});
    } else {
      ReduceSum<Context>(dev_ctx, &dy_xnorm, d_scale, std::vector<int>{0, 1});
    }
  }

  //  std = Sqrt(var+epsilon), init shape = [ N, G ]
  phi::DenseTensor std;
  phi::DenseTensorMeta std_meta = {y.dtype(), variance.dims()};
  std.set_meta(std_meta);
  dev_ctx.template Alloc<T>(&std);
  Adds<Context>(dev_ctx, &variance, epsilon, &std);
  Sqrt<Context>(dev_ctx, &std, &std);
  //  d_xnorm_std = dy_proc * scale / std
  phi::DenseTensor d_xnorm_std;
  phi::DenseTensorMeta d_xnorm_std_meta = {y.dtype(), y.dims()};
  d_xnorm_std.set_meta(d_xnorm_std_meta);
  dev_ctx.template Alloc<T>(&d_xnorm_std);
  Mul<Context>(dev_ctx, &d_y, &scale_share, &d_xnorm_std);

  if (data_layout_data == phi::DataLayout::kNCHW) {
    xnorm.Resize({N, groups, C * L / groups});
    d_xnorm_std.Resize({N, groups, C * L / groups});
    std.Resize({N, groups, 1});
  } else {
    xnorm.Resize({N, C * L / groups, groups});
    d_xnorm_std.Resize({N, C * L / groups, groups});
    std.Resize({N, 1, groups});
  }
  Div<Context>(dev_ctx, &d_xnorm_std, &std, &d_xnorm_std);

  d_x->Resize(xnorm.dims());
  dev_ctx.template Alloc<T>(d_x);
  Mul<Context>(dev_ctx, &d_xnorm_std, &xnorm, d_x);
  phi::DenseTensor dx1 = ReduceMeanToNG<T, Context>(
      dev_ctx, d_x, data_layout_data, N, C, L, groups);
  Mul<Context>(dev_ctx, &dx1, &xnorm, d_x);

  phi::DenseTensor dx2 = ReduceMeanToNG<T, Context>(
      dev_ctx, &d_xnorm_std, data_layout_data, N, C, L, groups);

  Sub<Context>(dev_ctx, &d_xnorm_std, d_x, d_x);
  Sub<Context>(dev_ctx, d_x, &dx2, d_x);

  d_x->Resize(y.dims());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(group_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GroupNormKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}

PD_REGISTER_PLUGIN_KERNEL(group_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GroupNormGradKernel,
                          float,
                          phi::dtype::float16) {}
