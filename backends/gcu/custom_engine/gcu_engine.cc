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

#include "custom_engine/gcu_engine.h"

namespace custom_engine {

GCUEngine::GCUEngine(const std::string &engine_key,
                     topsExecutable_t tops_exec,
                     const std::vector<const phi::DenseTensor *> &tensor_args,
                     const std::vector<phi::DenseTensor *> &return_tensor)
    : engine_key_(engine_key),
      tops_exec_(tops_exec),
      tensor_args_(tensor_args),
      return_tensor_(return_tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tops_exec_,
      phi::errors::InvalidArgument("Expect executable is not null."));
}

void GCUEngine::Init(const std::string &engine_key,
                     topsExecutable_t tops_exec,
                     const std::vector<const phi::DenseTensor *> &tensor_args,
                     const std::vector<phi::DenseTensor *> &return_tensor) {
  engine_key_ = engine_key;
  tops_exec_ = tops_exec;
  tensor_args_ = tensor_args;
  return_tensor_ = return_tensor;
  executor_ = std::make_shared<GCUEngineExecutor>(
      tops_exec, tensor_args, return_tensor);
}

void GCUEngine::Run(const phi::CustomContext &dev_ctx) {
  VLOG(3) << "=== GCUEngine Run ===";
  executor_->Run(dev_ctx);
}

}  // namespace custom_engine
