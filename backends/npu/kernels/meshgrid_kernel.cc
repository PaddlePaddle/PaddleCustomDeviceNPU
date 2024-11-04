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

template <typename T, typename Context>
void AclopMeshgridKernel(const Context& dev_ctx,
                         const std::vector<const phi::DenseTensor*>& ins,
                         std::vector<phi::DenseTensor*> outs) {
  auto stream = dev_ctx.stream();

  int64_t size = ins.size();
  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; i++) {
    switch (ins[i]->dims().size()) {
      case 0:
        shape[i] = 1;
        break;
      case 1:
        shape[i] = ins[i]->dims()[0];
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Expected scalar or 1D tensor in the tensor list but got tensor "
            "%d: ",
            i));
    }
  }

  for (int64_t i = 0; i < size; i++) {
    std::vector<int64_t> view_shape(size, 1);
    view_shape[i] = shape[i];

    phi::DDim out_dims_reshape = phi::make_ddim(view_shape);
    phi::DenseTensor reshape_ins_tensor(*ins[i]);
    reshape_ins_tensor.Resize(out_dims_reshape);

    phi::DDim out_dims = phi::make_ddim(shape);
    outs[i]->Resize(out_dims);
    dev_ctx.template Alloc<T>(outs[i]);

    NpuOpRunner runner;
    runner.SetType("BroadcastTo")
        .AddInput(reshape_ins_tensor)
        .AddInput(dev_ctx, std::move(shape))
        .AddOutput(*(outs[i]))
        .Run(stream);
  }
}

template <typename T, typename Context>
void MeshgridKernel(const Context& dev_ctx,
                    const std::vector<const phi::DenseTensor*>& ins,
                    std::vector<phi::DenseTensor*> outs) {
  PADDLE_ENFORCE_EQ(
      (ins.size() > 1) && (ins.size() < 7),
      true,
      phi::errors::InvalidArgument(
          "Excepted Tensor numbers between 2 and 6, but only received d% .",
          ins.size()));

  DO_COMPATIBILITY(
      aclnnExpand,
      (custom_kernel::AclopMeshgridKernel<T, Context>(dev_ctx, ins, outs)));

  int64_t size = ins.size();
  std::vector<int64_t> shape(size);

  for (int64_t i = 0; i < size; i++) {
    switch (ins[i]->dims().size()) {
      case 0:
        shape[i] = 1;
        break;
      case 1:
        shape[i] = ins[i]->dims()[0];
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Expected scalar or 1D tensor in the tensor list but got tensor "
            "%d: ",
            i));
    }
  }

  for (int64_t i = 0; i < size; i++) {
    std::vector<int64_t> view_shape(size, 1);
    view_shape[i] = shape[i];

    phi::DDim out_dims_reshape = phi::make_ddim(view_shape);
    phi::DenseTensor reshape_ins_tensor(*ins[i]);
    reshape_ins_tensor.Resize(out_dims_reshape);

    phi::DDim out_dims = phi::make_ddim(shape);
    outs[i]->Resize(out_dims);
    dev_ctx.template Alloc<T>(outs[i]);

    EXEC_NPU_CMD(aclnnExpand, dev_ctx, reshape_ins_tensor, shape, *(outs[i]));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(meshgrid,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeshgridKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
