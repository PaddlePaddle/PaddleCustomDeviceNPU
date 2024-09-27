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

template <typename T, typename Context>
void AccuracyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& infer_out,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& label,
                       phi::DenseTensor* accuracy,
                       phi::DenseTensor* correct,
                       phi::DenseTensor* total) {
  PADDLE_GCU_KERNEL_TRACE("accuracy");
  dev_ctx.template Alloc<float>(accuracy);
  dev_ctx.template Alloc<int>(correct);
  dev_ctx.template Alloc<int>(total);

  if (LaunchAOTKernel()) {
    PADDLE_ENFORCE_EQ(
        infer_out.dims().size(),
        2,
        phi::errors::InvalidArgument(
            "Rank(Input) of AccuracyOp must be 2, with shape "
            "[sample_number, class_dim], But received rank(Input) is %d",
            infer_out.dims().size()));

    auto num_samples = static_cast<int64_t>(infer_out.dims()[0]);
    if (num_samples == 0) {
      return;
    }

    auto indices_i32 = MaybeCreateOrTrans64To32bits(dev_ctx, indices);
    auto label_i32 = MaybeCreateOrTrans64To32bits(dev_ctx, label);

    phi::DenseTensorMeta equal_out_meta = {phi::DataType::BOOL,
                                           infer_out.dims()};
    auto equal_out = custom_kernel::TensorEmpty(dev_ctx, equal_out_meta);

    LAUNCH_TOPSATENOP(topsatenEq, dev_ctx, equal_out, indices_i32, label_i32);
    auto equal_out_f =
        custom_kernel::Cast(dev_ctx, equal_out, phi::DataType::FLOAT32);

    // correct: reduce_max + reduce_sum
    phi::DenseTensorMeta correct_max_meta = {phi::DataType::FLOAT32,
                                             phi::make_ddim({num_samples})};
    auto correct_max = custom_kernel::TensorEmpty(dev_ctx, correct_max_meta);
    int axis = 1;
    bool keep_dim = false;
    LAUNCH_TOPSATENOP(
        topsatenMax, dev_ctx, correct_max, equal_out_f, axis, keep_dim);

    phi::DenseTensorMeta correct_sum_meta = {phi::DataType::FLOAT32,
                                             correct->dims()};
    auto correct_sum = custom_kernel::TensorEmpty(dev_ctx, correct_sum_meta);
    LAUNCH_TOPSATENOP(
        topsatenSum, dev_ctx, correct_sum, correct_max, phi::DataType::FLOAT32);
    custom_kernel::Cast(dev_ctx, correct_sum, phi::DataType::INT32, correct);

    // total
    FillGcuTensorWithConstant<int>(
        total, dev_ctx, static_cast<int>(num_samples));

    // accuracy
    phi::DenseTensorMeta total_f_meta = {phi::DataType::FLOAT32, total->dims()};
    auto total_f = custom_kernel::TensorEmpty(dev_ctx, total_f_meta);
    FillGcuTensorWithConstant<float>(
        &total_f, dev_ctx, static_cast<float>(num_samples));
    LAUNCH_TOPSATENOP(topsatenDiv, dev_ctx, *accuracy, correct_sum, total_f);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Out"] = {"infer_out"};
    input_names["Indices"] = {"indices"};
    input_names["Label"] = {"label"};

    TensorValueMap inputs;
    inputs["Out"] = {const_cast<DenseTensor*>(&infer_out)};
    inputs["Indices"] = {const_cast<DenseTensor*>(&indices)};
    inputs["Label"] = {const_cast<DenseTensor*>(&label)};

    TensorNameMap output_names;
    output_names["Accuracy"] = {"accuracy"};
    output_names["Correct"] = {"correct"};
    output_names["Total"] = {"total"};

    TensorValueMap outputs;
    outputs["Accuracy"] = {accuracy};
    outputs["Correct"] = {correct};
    outputs["Total"] = {total};

    GcuRunner(
        input_names, inputs, output_names, outputs, {}, "accuracy", dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(accuracy,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AccuracyRawKernel,
                          float,
                          phi::dtype::float16,
                          int) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
