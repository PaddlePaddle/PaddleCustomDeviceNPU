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

void GetSeed(const phi::DeviceContext& dev_ctx,
             const paddle::optional<phi::DenseTensor>& seed_tensor,
             int seed,
             bool fix_seed,
             int* seed1,
             int* seed2) {
  if (seed_tensor) {
    MemCpyD2H(nullptr, seed1, seed_tensor->data(), sizeof(int));
  } else if (!fix_seed) {
    // use cpu engine to generate a seed for npu.
    auto offset = 0;
    auto& engine = *dev_ctx.GetGenerator()->GetCPUEngine();
    *seed1 = static_cast<int>(engine());
  } else {
    *seed1 = seed;
  }
  *seed2 = 0;
}

template <typename T, typename Context>
void AclopDropoutRawKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& seed_tensor,
    const phi::Scalar& p,
    bool is_test,
    const std::string& mode,
    int seed,
    bool fix_seed,
    phi::DenseTensor* out,
    phi::DenseTensor* mask) {
  if (x.dims().size() == 0) {
    phi::DenseTensor tmp_x(x);
    tmp_x.Resize(phi::make_ddim({1}));
    out->Resize(phi::make_ddim({1}));
    ::custom_kernel::AclopDropoutRawKernel<T, Context>(dev_ctx,
                                                       tmp_x,
                                                       seed_tensor,
                                                       p,
                                                       is_test,
                                                       mode,
                                                       seed,
                                                       fix_seed,
                                                       out,
                                                       mask);
    out->Resize(phi::make_ddim({}));
    return;
  }
  const bool is_upscale = (mode == "upscale_in_train");
  auto dropout_prob = p.to<float>();

  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (!is_test) {
    // mask used in `DropOutGenMask` NPU OP is different from
    // the output `Mask`.
    uint32_t length = (x.numel() + 128 - 1) / 128 * 128;
    mask->Resize({length / 8});
    dev_ctx.template Alloc<uint8_t>(mask);
  }

  if (dropout_prob == 0.0f) {
    TensorCopy(dev_ctx, x, false, out);
    FillNpuTensorWithConstant<uint8_t>(
        mask, dev_ctx, static_cast<uint8_t>(255));
    return;
  }

  if (dropout_prob == 1.) {
    const auto& runner_zeros_out = NpuOpRunner("ZerosLike", {*out}, {*out});
    runner_zeros_out.Run(stream);
    ACL_CHECK(aclrtMemsetAsync(mask->data(),
                               mask->numel() * sizeof(uint8_t),
                               0,
                               mask->numel() * sizeof(uint8_t),
                               stream));
    return;
  }

  // only achieve the default `upscale_in_train` method
  if (!is_test) {
    int seed1 = 0;
    int seed2 = 0;
    GetSeed(dev_ctx, seed_tensor, seed, fix_seed, &seed1, &seed2);
    phi::DenseTensor seed1_tensor, seed2_tensor;
    custom_kernel::TensorFromVector<int32_t>(
        dev_ctx, {seed1}, dev_ctx, &seed1_tensor);
    custom_kernel::TensorFromVector<int32_t>(
        dev_ctx, {seed2}, dev_ctx, &seed2_tensor);

    if (x.dtype() == phi::DataType::FLOAT64) {
      float keep_prob = static_cast<float>(1. - dropout_prob);

      // transform x
      phi::DenseTensor tmp_x;
      phi::DenseTensorMeta tmp_x_meta = {phi::DataType::FLOAT32, x.dims()};
      tmp_x.set_meta(tmp_x_meta);
      dev_ctx.template Alloc<float>(&tmp_x);
      NpuOpRunner runner;
      runner.SetType("Cast")
          .AddInput(x)
          .AddOutput(tmp_x)
          .AddAttr("dst_type", ACL_FLOAT)
          .Run(stream);
      phi::DenseTensor tmp_out;
      phi::DenseTensorMeta tmp_out_meta = {phi::DataType::FLOAT32, out->dims()};
      tmp_out.set_meta(tmp_out_meta);
      dev_ctx.template Alloc<float>(&tmp_out);

      if (x.dims().size() == 1) {
        // DropOutDoMask will get error result when input
        // is 1-D. Make it become 2-D.
        std::vector<int> vec_dim = phi::vectorize<int>(x.dims());
        tmp_x.Resize(phi::make_ddim({vec_dim[0], 1}));
        tmp_out.Resize(phi::make_ddim({vec_dim[0], 1}));
      }

      NpuOpRunner runner_gen_mask;
      runner_gen_mask.SetType("StatelessDropOutGenMask")
          .AddInput(dev_ctx, phi::vectorize(tmp_out.dims()))
          .AddInput(dev_ctx, std::vector<float>({keep_prob}))
          .AddInput(seed1_tensor)
          .AddInput(seed2_tensor)
          .AddOutput(*mask);
      // dropout用双stream性能会更好，后续要持续优化
      runner_gen_mask.Run(stream);
      // SecondaryStream::Instance().RecordBefore(dev_ctx.stream());

      phi::DenseTensor keep_prob_tensor;
      phi::DenseTensorMeta keep_prob_tensor_meta = {phi::DataType::FLOAT32,
                                                    {1}};
      keep_prob_tensor.set_meta(keep_prob_tensor_meta);
      keep_prob_tensor.Resize(phi::DDim({}));
      C_Device_st device{dev_ctx.GetPlace().GetDeviceId()};
      AsyncMemCpyH2D(&device,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     dev_ctx.template Alloc<float>(&keep_prob_tensor),
                     &keep_prob,
                     sizeof(float));
      NpuOpRunner runner_dropout;
      runner_dropout.SetType("DropOutDoMask")
          .AddInput(tmp_x)
          .AddInput(*mask)
          .AddInput(keep_prob_tensor)
          .AddOutput(tmp_out);
      runner_dropout.Run(stream);

      NpuOpRunner runner_cast;
      runner_cast.SetType("Cast")
          .AddInput(tmp_out)
          .AddOutput(*out)
          .AddAttr("dst_type", ACL_DOUBLE)
          .Run(stream);
    } else {
      T keep_prob = static_cast<T>(1. - dropout_prob);

      phi::DenseTensor tmp_x(x);
      phi::DenseTensor tmp_out(*out);

      if (x.dims().size() == 1) {
        // DropOutDoMask will get error result when input
        // is 1-D. Make it become 2-D.
        std::vector<int> vec_dim = phi::vectorize<int>(x.dims());
        tmp_x.Resize(phi::make_ddim({vec_dim[0], 1}));
        tmp_out.Resize(phi::make_ddim({vec_dim[0], 1}));
      }

      // TODO(pangyoki): `keep_prob` used in `DropOutGenMask` NPU
      // OP must be a scalar with shape[0]. At present, the shape
      // of the `prob` Tensor of this OP is forced to be set to 0
      // in `npu_op_runner.cc`, which needs to be optimized later.
      NpuOpRunner runner_gen_mask;
      runner_gen_mask.SetType("StatelessDropOutGenMask")
          .AddInput(dev_ctx, phi::vectorize(tmp_out.dims()))
          .AddInput(dev_ctx, std::vector<float>({keep_prob}))
          .AddInput(seed1_tensor)
          .AddInput(seed2_tensor)
          .AddOutput(*mask);
      // dropout用双stream性能会更好，后续要持续优化！！！！
      // 当前双stream有问题，与wangran沟通，先改成单stream
      runner_gen_mask.Run(stream);
      // SecondaryStream::Instance().RecordBefore(dev_ctx.stream());

      phi::DenseTensor keep_prob_tensor;
      phi::DenseTensorMeta keep_prob_tensor_meta = {x.dtype(), {1}};
      keep_prob_tensor.set_meta(keep_prob_tensor_meta);
      keep_prob_tensor.Resize(phi::DDim({}));

      C_Device_st device{dev_ctx.GetPlace().GetDeviceId()};
      AsyncMemCpyH2D(&device,
                     static_cast<C_Stream>(dev_ctx.stream()),
                     dev_ctx.template Alloc<T>(&keep_prob_tensor),
                     &keep_prob,
                     sizeof(T));
      if (!is_upscale) {
        const auto& muls_runner =
            NpuOpRunner("OnesLike", {keep_prob_tensor}, {keep_prob_tensor});
        muls_runner.Run(stream);
      }
      NpuOpRunner runner_dropout;
      runner_dropout.SetType("DropOutDoMask")
          .AddInput(tmp_x)
          .AddInput(*mask)
          .AddInput(keep_prob_tensor)
          .AddOutput(tmp_out);
      runner_dropout.Run(stream);
    }
  } else {
    if (!is_upscale) {
      const auto& muls_runner =
          NpuOpRunner("Muls", {x}, {*out}, {{"value", 1 - dropout_prob}});
      muls_runner.Run(stream);
      return;
    }
    TensorCopy(dev_ctx, x, false, out);
  }
}

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const paddle::optional<phi::DenseTensor>& seed_tensor,
                      const phi::Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      phi::DenseTensor* out,
                      phi::DenseTensor* mask) {
  DO_COMPATIBILITY(
      aclnnDropoutGenMask,
      (custom_kernel::AclopDropoutRawKernel<T, Context>(dev_ctx,
                                                        x,
                                                        seed_tensor,
                                                        p,
                                                        is_test,
                                                        mode,
                                                        seed,
                                                        fix_seed,
                                                        out,
                                                        mask)));
  DO_COMPATIBILITY(
      aclnnDropoutDoMask,
      (custom_kernel::AclopDropoutRawKernel<T, Context>(dev_ctx,
                                                        x,
                                                        seed_tensor,
                                                        p,
                                                        is_test,
                                                        mode,
                                                        seed,
                                                        fix_seed,
                                                        out,
                                                        mask)));
  if (x.dims().size() == 0) {
    phi::DenseTensor tmp_x(x);
    tmp_x.Resize(phi::make_ddim({1}));
    out->Resize(phi::make_ddim({1}));
    ::custom_kernel::DropoutRawKernel<T, Context>(dev_ctx,
                                                  tmp_x,
                                                  seed_tensor,
                                                  p,
                                                  is_test,
                                                  mode,
                                                  seed,
                                                  fix_seed,
                                                  out,
                                                  mask);
    out->Resize(phi::make_ddim({}));
    return;
  }
  const bool is_upscale = (mode == "upscale_in_train");
  auto dropout_prob = p.to<double>();

  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  if (!is_test) {
    // mask used in `DropOutGenMask` NPU OP is different from
    // the output `Mask`.
    uint32_t length = (x.numel() + 128 - 1) / 128 * 128;
    mask->Resize({length / 8});
    dev_ctx.template Alloc<uint8_t>(mask);
  }

  if (dropout_prob == 0.0) {
    TensorCopy(dev_ctx, x, false, out);
    FillNpuTensorWithConstant<uint8_t>(
        mask, dev_ctx, static_cast<uint8_t>(255));
    return;
  }

  if (dropout_prob == 1.) {
    const auto& runner_zeros_out = NpuOpRunner("ZerosLike", {*out}, {*out});
    runner_zeros_out.Run(stream);
    ACL_CHECK(aclrtMemsetAsync(mask->data(),
                               mask->numel() * sizeof(uint8_t),
                               0,
                               mask->numel() * sizeof(uint8_t),
                               stream));
    return;
  }

  // only achieve the default `upscale_in_train` method
  if (!is_test) {
    int seed1 = 0;
    int seed2 = 0;
    GetSeed(dev_ctx, seed_tensor, seed, fix_seed, &seed1, &seed2);
    phi::DenseTensor seed1_tensor, seed2_tensor;
    custom_kernel::TensorFromVector<int32_t>(
        dev_ctx, {seed1}, dev_ctx, &seed1_tensor);
    custom_kernel::TensorFromVector<int32_t>(
        dev_ctx, {seed2}, dev_ctx, &seed2_tensor);

    if (x.dtype() == phi::DataType::FLOAT64) {
      // transform x
      phi::DenseTensor tmp_x;
      phi::DenseTensorMeta tmp_x_meta = {phi::DataType::FLOAT32, x.dims()};
      tmp_x.set_meta(tmp_x_meta);
      dev_ctx.template Alloc<float>(&tmp_x);
      NpuOpRunner runner;
      runner.SetType("Cast")
          .AddInput(x)
          .AddOutput(tmp_x)
          .AddAttr("dst_type", ACL_FLOAT)
          .Run(stream);
      phi::DenseTensor tmp_out;
      phi::DenseTensorMeta tmp_out_meta = {phi::DataType::FLOAT32, out->dims()};
      tmp_out.set_meta(tmp_out_meta);
      dev_ctx.template Alloc<float>(&tmp_out);

      if (x.dims().size() == 1) {
        // DropOutDoMask will get error result when input
        // is 1-D. Make it become 2-D.
        std::vector<int> vec_dim = phi::vectorize<int>(x.dims());
        tmp_x.Resize(phi::make_ddim({vec_dim[0], 1}));
        tmp_out.Resize(phi::make_ddim({vec_dim[0], 1}));
      }

      auto out_shape = phi::vectorize(tmp_out.dims());
      EXEC_NPU_CMD(aclnnDropoutGenMask,
                   dev_ctx,
                   out_shape,
                   dropout_prob,
                   seed1,
                   seed2,
                   *mask);

      EXEC_NPU_CMD(
          aclnnDropoutDoMask, dev_ctx, tmp_x, *mask, dropout_prob, tmp_out);

      NpuOpRunner runner_cast;
      runner_cast.SetType("Cast")
          .AddInput(tmp_out)
          .AddOutput(*out)
          .AddAttr("dst_type", ACL_DOUBLE)
          .Run(stream);
    } else {
      T keep_prob = static_cast<T>(1. - dropout_prob);

      phi::DenseTensor tmp_x(x);
      phi::DenseTensor tmp_out(*out);

      if (x.dims().size() == 1) {
        // DropOutDoMask will get error result when input
        // is 1-D. Make it become 2-D.
        std::vector<int> vec_dim = phi::vectorize<int>(x.dims());
        tmp_x.Resize(phi::make_ddim({vec_dim[0], 1}));
        tmp_out.Resize(phi::make_ddim({vec_dim[0], 1}));
      }

      auto out_shape = phi::vectorize(tmp_out.dims());
      EXEC_NPU_CMD(aclnnDropoutGenMask,
                   dev_ctx,
                   out_shape,
                   dropout_prob,
                   seed1,
                   seed2,
                   *mask);

      EXEC_NPU_CMD(
          aclnnDropoutDoMask, dev_ctx, tmp_x, *mask, dropout_prob, tmp_out);
    }

    if (!is_upscale) {
      auto revert_scale = static_cast<T>(1.0 - dropout_prob);
      aclDataType acl_data_type = ConvertToNpuDtype(x.dtype());
      static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
      aclScalar* acl_scalar_revert_scale =
          aclCreateScalar(&revert_scale, acl_data_type);
      EXEC_NPU_CMD(aclnnInplaceMuls, dev_ctx, *out, acl_scalar_revert_scale);
    }
  } else {
    if (!is_upscale) {
      auto down_scale = static_cast<T>(1.0 - dropout_prob);
      aclDataType acl_data_type = ConvertToNpuDtype(x.dtype());
      static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
      aclScalar* acl_scalar_down_scale =
          aclCreateScalar(&down_scale, acl_data_type);
      EXEC_NPU_CMD(aclnnMuls, dev_ctx, x, acl_scalar_down_scale, *out);
      return;
    }
    TensorCopy(dev_ctx, x, false, out);
  }
}

template <typename T, typename Context>
void AclopDropoutGradRawKernel(const Context& dev_ctx,
                               const phi::DenseTensor& mask,
                               const phi::DenseTensor& dout,
                               const phi::Scalar& p,
                               bool is_test,
                               const std::string& mode,
                               phi::DenseTensor* dx) {
  if (dout.dims().size() == 0) {
    phi::DenseTensor tmp_dout(dout);
    tmp_dout.Resize(phi::make_ddim({1}));
    dx->Resize(phi::make_ddim({1}));
    ::custom_kernel::AclopDropoutGradRawKernel<T, Context>(
        dev_ctx, mask, tmp_dout, p, is_test, mode, dx);
    dx->Resize(phi::make_ddim({}));
    return;
  }

  const bool is_upscale = (mode == "upscale_in_train");
  auto dropout_prob = p.to<float>();

  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  if (dropout_prob == 1.) {
    const auto& runner_zeros = NpuOpRunner("ZerosLike", {*dx}, {*dx});
    runner_zeros.Run(stream);
    return;
  }

  dropout_prob = is_upscale ? dropout_prob : 0.0f;
  if (dx->dtype() == phi::DataType::FLOAT64) {
    phi::DenseTensor keep_prob_tensor;
    phi::DenseTensorMeta keep_prob_tensor_meta = {phi::DataType::FLOAT32, {1}};
    keep_prob_tensor.set_meta(keep_prob_tensor_meta);
    FillNpuTensorWithConstant<float>(
        &keep_prob_tensor, dev_ctx, static_cast<float>(1 - dropout_prob));
    // transform dx
    phi::DenseTensor tmp_dx;
    phi::DenseTensorMeta tmp_dx_meta = {phi::DataType::FLOAT32, dx->dims()};
    tmp_dx.set_meta(tmp_dx_meta);
    dev_ctx.template Alloc<float>(&tmp_dx);
    // transform dout
    phi::DenseTensor tmp_dout;
    phi::DenseTensorMeta tmp_dout_meta = {phi::DataType::FLOAT32, dout.dims()};
    tmp_dout.set_meta(tmp_dout_meta);
    dev_ctx.template Alloc<float>(&tmp_dout);
    NpuOpRunner runner1;
    runner1.SetType("Cast")
        .AddInput(dout)
        .AddOutput(tmp_dout)
        .AddAttr("dst_type", ACL_FLOAT)
        .Run(stream);
    NpuOpRunner runner_dropout;
    runner_dropout.SetType("DropOutDoMask")
        .AddInput(tmp_dout)
        .AddInput(mask)
        .AddInput(keep_prob_tensor)
        .AddOutput(tmp_dx)
        .Run(stream);
    NpuOpRunner runner2;
    runner2.SetType("Cast")
        .AddInput(tmp_dx)
        .AddOutput(*dx)
        .AddAttr("dst_type", ACL_DOUBLE)
        .Run(stream);
  } else {
    phi::DenseTensor keep_prob_tensor;
    phi::DenseTensorMeta keep_prob_tensor_meta = {dx->dtype(), {1}};
    keep_prob_tensor.set_meta(keep_prob_tensor_meta);
    FillNpuTensorWithConstant<T>(
        &keep_prob_tensor, dev_ctx, static_cast<T>(1 - dropout_prob));

    NpuOpRunner runner_dropout;
    runner_dropout.SetType("DropOutDoMask")
        .AddInput(dout)
        .AddInput(mask)
        .AddInput(keep_prob_tensor)
        .AddOutput(*dx);
    runner_dropout.Run(stream);
  }
  return;
}

template <typename T, typename Context>
void DropoutGradRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& mask,
                          const phi::DenseTensor& dout,
                          const phi::Scalar& p,
                          bool is_test,
                          const std::string& mode,
                          phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnDropoutDoMask,
                   (custom_kernel::AclopDropoutGradRawKernel<T, Context>(
                       dev_ctx, mask, dout, p, is_test, mode, dx)));
  if (dout.dims().size() == 0) {
    phi::DenseTensor tmp_dout(dout);
    tmp_dout.Resize(phi::make_ddim({1}));
    dx->Resize(phi::make_ddim({1}));
    ::custom_kernel::DropoutGradRawKernel<T, Context>(
        dev_ctx, mask, tmp_dout, p, is_test, mode, dx);
    dx->Resize(phi::make_ddim({}));
    return;
  }

  const bool is_upscale = (mode == "upscale_in_train");
  auto dropout_prob = p.to<double>();

  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  if (dropout_prob == 1.) {
    const auto& runner_zeros = NpuOpRunner("ZerosLike", {*dx}, {*dx});
    runner_zeros.Run(stream);
    return;
  }

  if (dx->dtype() == phi::DataType::FLOAT64) {
    // transform dx
    phi::DenseTensor tmp_dx;
    phi::DenseTensorMeta tmp_dx_meta = {phi::DataType::FLOAT32, dx->dims()};
    tmp_dx.set_meta(tmp_dx_meta);
    dev_ctx.template Alloc<float>(&tmp_dx);
    // transform dout
    phi::DenseTensor tmp_dout;
    phi::DenseTensorMeta tmp_dout_meta = {phi::DataType::FLOAT32, dout.dims()};
    tmp_dout.set_meta(tmp_dout_meta);
    dev_ctx.template Alloc<float>(&tmp_dout);

    NpuOpRunner runner1;
    runner1.SetType("Cast")
        .AddInput(dout)
        .AddOutput(tmp_dout)
        .AddAttr("dst_type", ACL_FLOAT)
        .Run(stream);

    EXEC_NPU_CMD(
        aclnnDropoutDoMask, dev_ctx, tmp_dout, mask, dropout_prob, tmp_dx);

    NpuOpRunner runner2;
    runner2.SetType("Cast")
        .AddInput(tmp_dx)
        .AddOutput(*dx)
        .AddAttr("dst_type", ACL_DOUBLE)
        .Run(stream);
  } else {
    EXEC_NPU_CMD(aclnnDropoutDoMask, dev_ctx, dout, mask, dropout_prob, *dx);
  }

  if (!is_upscale) {
    auto revert_scale = static_cast<T>(1.0 - dropout_prob);
    aclDataType acl_data_type = ConvertToNpuDtype(dx->dtype());
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    aclScalar* acl_scalar_revert_scale =
        aclCreateScalar(&revert_scale, acl_data_type);
    EXEC_NPU_CMD(aclnnInplaceMuls, dev_ctx, *dx, acl_scalar_revert_scale);
  }
  return;
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(dropout,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DropoutRawKernel,
                          float,
                          phi::dtype::float16,
                          double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_PLUGIN_KERNEL(dropout_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DropoutGradRawKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
