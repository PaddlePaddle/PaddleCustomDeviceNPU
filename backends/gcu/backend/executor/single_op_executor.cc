/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "backend/executor/single_op_executor.h"

#include <algorithm>
#include <chrono>  // NOLINT [build/c++11]
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "backend/executor/tops_compiler.h"
#include "common/gcu_funcs.h"
#include "common/utils.h"
#include "runtime/runtime.h"

namespace backend {
using TensorPtr = std::shared_ptr<phi::DenseTensor>;
using phi::DenseTensor;
using DenseTensor = phi::DenseTensor;

SingleOpGcuExecutor::SingleOpGcuExecutor(
    const std::string& op_type,
    topsExecutable_t exec,
    const std::vector<GcuNode>& input_nodes,
    const std::vector<GcuNode>& output_nodes)
    : op_type_(op_type) {
  PADDLE_ENFORCE_NOT_NULL(
      exec, phi::errors::InvalidArgument("Expect executable is not null."));
  tops_exec_ = exec;
  input_nodes_ = input_nodes;
  output_nodes_ = output_nodes;
}

SingleOpGcuExecutor::SingleOpGcuExecutor(
    const std::string& op_type,
    const std::vector<GcuNode>& input_nodes,
    const std::vector<GcuNode>& output_nodes)
    : op_type_(op_type) {
  input_nodes_ = input_nodes;
  output_nodes_ = output_nodes;
}

void SingleOpGcuExecutor::ReleaseResource() {
  if (tops_exec_ != nullptr) {
    RT_CHECK(topsDestroyExecutable(tops_exec_));
    tops_exec_ = nullptr;
  }
}

void SingleOpGcuExecutor::RunGcuOp(const phi::CustomContext* device_context,
                                   const std::vector<DenseTensor*>& inputs,
                                   const std::vector<DenseTensor*>& outputs,
                                   bool tensor_split) {
  std::vector<void*> dev_inputs;
  dev_inputs.reserve(inputs.size());
  std::vector<void*> dev_outputs;
  dev_outputs.resize(outputs.size());

  static DenseTensor tmp_out_tensor;
  static std::once_flag alloc_flags;
  std::call_once(alloc_flags, [&]() {
    const phi::DenseTensorMeta meta(phi::DataType::FLOAT32,
                                    phi::make_ddim({1}));
    tmp_out_tensor.set_meta(meta);
    device_context->Alloc<float>(&tmp_out_tensor);
  });

  std::vector<DenseTensor*> real_inputs;
  std::vector<DenseTensor*> real_outputs;
  real_inputs.reserve(inputs.size());
  real_outputs.reserve(outputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    PADDLE_ENFORCE_NE(
        inputs[i], nullptr, phi::errors::InvalidArgument("inputs is null"));
    GcuNode input_node(*inputs[i]);
    PADDLE_ENFORCE_EQ(
        input_node,
        input_nodes_[i],
        phi::errors::InvalidArgument(
            "input %zu desc not equal cached, input desc: %s vs cached desc %s",
            i,
            input_node.to_str(),
            input_nodes_[i].to_str()));
    auto* tensor = inputs[i];

    if (tensor->initialized()) {
      dev_inputs.emplace_back(tensor->data());
      real_inputs.emplace_back(tensor);
      VLOG(6) << "op_type: " << op_type_ << ", inputs[" << i
              << "] addr:" << tensor->data() << ", capacity is "
              << tensor->capacity() << ", type:" << tensor->dtype()
              << ", place:" << tensor->place()
              << ", ddim:" << tensor->dims().to_str();
    } else {
      VLOG(6) << "op_type: " << op_type_ << ", inputs[" << i
              << "] is not initialized.";
    }
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    PADDLE_ENFORCE_NE(
        outputs[i], nullptr, phi::errors::InvalidArgument("outputs is null"));
    GcuNode output_node(*outputs[i]);
    PADDLE_ENFORCE_EQ(
        output_node,
        output_nodes_[i],
        phi::errors::InvalidArgument("output %zu desc not equal cached, "
                                     "input desc: %s vs cached desc %s",
                                     i,
                                     output_node.to_str(),
                                     output_nodes_[i].to_str()));
    auto* tensor = outputs[i];
    if (tensor->initialized()) {
      dev_outputs[i] = outputs[i]->data();
      real_outputs.emplace_back(outputs[i]);
    } else {
      dev_outputs[i] = tmp_out_tensor.data();
      real_outputs.emplace_back(&tmp_out_tensor);
    }
    VLOG(6) << "op_type: " << op_type_ << ", outputs[" << i
            << "] addr:" << dev_outputs[i] << ", capacity is "
            << tensor->capacity() << ", type:" << tensor->dtype()
            << ", place:" << tensor->place()
            << ", ddim:" << tensor->dims().to_str();
  }

  auto tops_stream = static_cast<topsStream_t>(device_context->stream());

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
}

}  // namespace backend
