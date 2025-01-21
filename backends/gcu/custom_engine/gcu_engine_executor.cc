// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "custom_engine/gcu_engine_executor.h"

namespace custom_engine {
void GCUEngineExecutor::Init() {
  tensor_args_device_.resize(tensor_args_.size());
}

void GCUEngineExecutor::Run(const phi::CustomContext &dev_ctx) {
  VLOG(3) << "=== GCUEngineExecutor Run ===";
  std::vector<void *> dev_inputs;
  dev_inputs.reserve(tensor_args_.size());
  std::vector<void *> dev_outputs;
  dev_outputs.resize(return_tensor_.size());

  for (size_t i = 0; i < tensor_args_.size(); ++i) {
    auto input = tensor_args_[i];
    PADDLE_ENFORCE_NE(
        input, nullptr, phi::errors::InvalidArgument("inputs is null"));

    if (input->initialized()) {
      phi::DenseTensor *tensor = &(tensor_args_device_[i]);
      if (input->place().GetType() != phi::AllocationType::CUSTOM) {
        custom_kernel::TensorCopy(dev_ctx, *input, false, tensor);
      } else {
        *tensor = *input;
      }
      auto device_tensor = tensor_args_device_[i];
      dev_inputs.emplace_back(device_tensor.data());
      VLOG(6) << "GCUEngineExecutor::Run, Inputs[" << i
              << "] addr:" << device_tensor.data() << ", capacity is "
              << device_tensor.capacity() << ", type:" << device_tensor.dtype()
              << ", place:" << device_tensor.place()
              << ", ddim:" << device_tensor.dims().to_str();
    } else {
      VLOG(6) << "GCUEngineExecutor::Run, inputs[" << i
              << "] is not initialized.";
    }
  }

  for (size_t i = 0; i < return_tensor_.size(); ++i) {
    auto *tensor = return_tensor_[i];
    PADDLE_ENFORCE_NE(
        tensor, nullptr, phi::errors::InvalidArgument("outputs is null"));
    dev_ctx.Alloc(tensor, tensor->dtype());
    dev_outputs[i] = tensor->data();

    VLOG(6) << "GCUEngineExecutor::Run, outputs[" << i
            << "] addr:" << tensor->data() << ", capacity is "
            << tensor->capacity() << ", type:" << tensor->dtype()
            << ", place:" << tensor->place()
            << ", ddim:" << tensor->dims().to_str();
  }

  auto tops_stream = static_cast<topsStream_t>(dev_ctx.stream());
  VLOG(6) << "GCUEngineExecutor Run on stream:" << tops_stream
          << ", tops_exec_:" << tops_exec_;

  static double total_time_cost = 0;
  static int32_t exec_count = 0;
  auto start_time = custom_kernel::GetCurrentTimestap();

  RT_CHECK(topsLaunchExecutable(tops_exec_,
                                nullptr,
                                dev_inputs.data(),
                                dev_inputs.size(),
                                nullptr,
                                nullptr,
                                dev_outputs.data(),
                                dev_outputs.size(),
                                nullptr,
                                nullptr,
                                tops_stream));

  if (VLOG_IS_ON(6)) {
    auto time_cost = custom_kernel::GetTimeCostInMs(
        start_time, custom_kernel::GetCurrentTimestap());
    total_time_cost += time_cost;

    VLOG(6) << "exec_count: " << ++exec_count << ", time_cost: " << time_cost
            << ", total_time_cost: " << total_time_cost;
  }
  return;
}

}  // namespace custom_engine
