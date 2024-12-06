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

#pragma once
#include <tops/tops_ext.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/executor/gcu_node.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/core/dense_tensor.h"

using DenseTensor = phi::DenseTensor;
using DDim = phi::DDim;
using DataType = phi::DataType;
using DataLayout = phi::DataLayout;

namespace backend {

class SingleOpGcuExecutor {
 public:
  SingleOpGcuExecutor(const std::string& op_type,
                      topsExecutable_t exec,
                      const std::vector<GcuNode>& input_nodes,
                      const std::vector<GcuNode>& output_nodes);
  SingleOpGcuExecutor(const std::string& op_type,
                      const std::vector<GcuNode>& input_nodes,
                      const std::vector<GcuNode>& output_nodes);
  SingleOpGcuExecutor() = delete;
  ~SingleOpGcuExecutor() = default;
  SingleOpGcuExecutor(const SingleOpGcuExecutor& exec) = default;
  SingleOpGcuExecutor& operator=(const SingleOpGcuExecutor& exec) = default;
  void ReleaseResource();
  void RunGcuOp(const phi::CustomContext* device_context,
                const std::vector<DenseTensor*>& inputs,
                const std::vector<DenseTensor*>& outputs,
                bool tensor_split);

 private:
  const std::string op_type_;
  topsExecutable_t tops_exec_ = nullptr;
  std::vector<GcuNode> input_nodes_;
  std::vector<GcuNode> output_nodes_;
};

class SingleOpGcuExecutorManager {
 public:
  ~SingleOpGcuExecutorManager() { ReleaseAll(); }

  void ReleaseAll() {
    for (const auto& p : single_executors_) {
      p.second->ReleaseResource();
    }
    single_executors_.clear();
  }

  void Add(const std::string& key,
           const std::shared_ptr<SingleOpGcuExecutor>& exec) {
    single_executors_[key] = exec;
  }

  std::shared_ptr<SingleOpGcuExecutor> Find(const std::string& key) {
    if (single_executors_.count(key) == 0) {
      return nullptr;
    }
    auto exec = single_executors_[key];
    PADDLE_ENFORCE_NE(
        exec, nullptr, phi::errors::NotFound("buffered exec is nullptr"));
    return exec;
  }

 public:
  static SingleOpGcuExecutorManager* GetInstance() {
    static SingleOpGcuExecutorManager manager;
    return &manager;
  }

 private:
  std::map<std::string, std::shared_ptr<SingleOpGcuExecutor>> single_executors_;
};

}  // namespace backend
