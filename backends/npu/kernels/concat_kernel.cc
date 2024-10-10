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

/// If a variable is a empty variable, that name will be used.
constexpr char kEmptyVarName[] = "@EMPTY@";

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename T, typename Context>
void AclopConcatKernel(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& ins,
                       const phi::Scalar& axis_scalar,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));

  std::vector<phi::DenseTensor> inputs;
  std::vector<std::string> names;
  names.push_back("concat_dim");

  if (out->dtype() != phi::DataType::FLOAT64) {
    for (size_t i = 0; i < ins.size(); ++i) {
      if (ins[i] && ins[i]->numel() > 0) {
        inputs.push_back(*ins[i]);
        names.push_back("x" + std::to_string(i));
      } else {
        continue;
      }
    }
    if (inputs.size() == 1) {
      out->ResizeAndAllocate(inputs[0].dims());
      TensorCopy(dev_ctx, inputs[0], true, out);
      return;
    }
    NpuOpRunner runner;
    runner.SetType("Concat")
        .AddInput(dev_ctx, std::move(std::vector<int>(1, axis)))
        .AddInputs(inputs)
        .AddOutput(*out)
        .AddAttr("N", static_cast<int>(inputs.size()));
    runner.AddInputNames(names);
    runner.Run(stream);

  } else {
    // TODO(songkai05): In CANN512, Concat doesn't support dtype double,
    // so cast double to float32 temporarily until it supports double.
    for (size_t i = 0; i < ins.size(); ++i) {
      if (ins[i] && ins[i]->numel() > 0) {
        phi::DenseTensor tmp;
        phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, ins[i]->dims()};
        tmp.set_meta(meta);
        dev_ctx.template Alloc<float>(&tmp);
        const auto& cast_runner =
            NpuOpRunner("Cast", {*ins[i]}, {tmp}, {{"dst_type", ACL_FLOAT}});
        cast_runner.Run(stream);
        dev_ctx.Wait();
        inputs.push_back(tmp);
        names.push_back("x" + std::to_string(i));
      } else {
        continue;
      }
    }

    phi::DenseTensor out_fp32;
    phi::DenseTensorMeta meta_fp32 = {phi::DataType::FLOAT32, out->dims()};
    out_fp32.set_meta(meta_fp32);
    dev_ctx.template Alloc<float>(&out_fp32);

    if (inputs.size() == 1) {
      int index = std::stoi(names[0].substr(1, names[0].size() - 1));
      out->ResizeAndAllocate(ins[index]->dims());
      TensorCopy(dev_ctx, *ins[index], true, out);
      return;
    }

    NpuOpRunner runner;
    runner.SetType("Concat")
        .AddInput(dev_ctx, std::move(std::vector<int>(1, axis)))
        .AddInputs(inputs)
        .AddOutput(out_fp32)
        .AddAttr("N", static_cast<int>(inputs.size()));
    runner.AddInputNames(names);

    runner.Run(stream);
    const auto& cast_out =
        NpuOpRunner("Cast", {out_fp32}, {*out}, {{"dst_type", ACL_DOUBLE}});
    cast_out.Run(stream);
  }
}

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnCat,
                   (custom_kernel::AclopConcatKernel<T, Context>(
                       dev_ctx, ins, axis_scalar, out)));
  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));
  std::vector<const phi::DenseTensor*> inputs;
  for (size_t i = 0; i < ins.size(); ++i) {
    if (ins[i] && ins[i]->numel() > 0) {
      inputs.push_back(ins[i]);
    } else {
      continue;
    }
  }
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnCat, dev_ctx, inputs, axis, *out);
}

template <typename T, typename Context>
void AclopConcatGradKernel(const Context& dev_ctx,
                           const std::vector<const phi::DenseTensor*>& ins,
                           const phi::DenseTensor& dout,
                           const phi::Scalar& axis_scalar,
                           std::vector<phi::DenseTensor*> outs) {
  auto stream = dev_ctx.stream();

  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));

  int offset = 0;
  for (size_t j = 0; j < outs.size(); ++j) {
    if (outs[j] && outs[j]->numel() != 0UL) {
      dev_ctx.template Alloc<T>(outs[j]);

      std::vector<int> offsets;
      std::vector<int> sizes;
      for (int dim = 0; dim < ins[j]->dims().size(); ++dim) {
        if (dim == axis) {
          offsets.push_back(offset);
          sizes.push_back(ins[j]->dims()[dim]);
        } else {
          offsets.push_back(0);
          sizes.push_back(ins[j]->dims()[dim]);
        }
      }
      NpuOpRunner runner;
      runner.SetType("Slice")
          .AddInput(dout)
          .AddInput(dev_ctx, std::move(offsets))
          .AddInput(dev_ctx, std::move(sizes))
          .AddOutput(*outs[j])
          .Run(stream);
    }
    if (ins[j]->numel() != 0UL) {
      offset += ins[j]->dims()[axis];
    }
  }
}

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& ins,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  DO_COMPATIBILITY(aclnnSliceV2,
                   (custom_kernel::AclopConcatGradKernel<T, Context>(
                       dev_ctx, ins, dout, axis_scalar, outs)));
  auto stream = dev_ctx.stream();

  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(static_cast<int64_t>(axis),
                     static_cast<int64_t>(ins[0]->dims().size()));

  std::vector<int64_t> axes_t;
  axes_t.push_back(axis);

  int offset = 0;
  for (size_t j = 0; j < outs.size(); ++j) {
    if (outs[j] && outs[j]->numel() != 0UL) {
      dev_ctx.template Alloc<T>(outs[j]);

      std::vector<int64_t> starts_array;
      starts_array.push_back(offset);
      std::vector<int64_t> ends_array;
      ends_array.push_back(ins[j]->dims()[axis] + offset);

      std::vector<int64_t> steps;
      for (int i = 0; i < axes_t.size(); i++) {
        steps.push_back(1.0);
      }

      EXEC_NPU_CMD(aclnnSliceV2,
                   dev_ctx,
                   dout,
                   starts_array,
                   ends_array,
                   axes_t,
                   steps,
                   *outs[j]);
    }
    if (ins[j]->numel() != 0UL) {
      offset += ins[j]->dims()[axis];
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(concat_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
