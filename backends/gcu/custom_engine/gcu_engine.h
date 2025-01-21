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

#pragma once

#include <glog/logging.h>

#include <vector>

#include "custom_engine/gcu_engine_executor.h"
#include "custom_engine/ir_translator/utils/utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace custom_engine {

class GCUEngine {
 public:
  GCUEngine() = default;
  GCUEngine(const std::string &engine_key,
            topsExecutable_t tops_exec,
            const std::vector<const phi::DenseTensor *> &tensor_args,
            const std::vector<phi::DenseTensor *> &return_tensor);
  ~GCUEngine() {
    if (tops_exec_ != nullptr) {
      RT_CHECK(topsDestroyExecutable(tops_exec_));
      tops_exec_ = nullptr;
      VLOG(3) << "Release topsExecutable and destory GCUEngine " << engine_key_;
    }
  }

  void Init(const std::string &engine_key,
            topsExecutable_t tops_exec,
            const std::vector<const phi::DenseTensor *> &tensor_args,
            const std::vector<phi::DenseTensor *> &return_tensor);

  void Run(const phi::CustomContext &dev_ctx);

 private:
  std::string engine_key_;
  topsExecutable_t tops_exec_ = nullptr;
  std::vector<const phi::DenseTensor *> tensor_args_;
  std::vector<phi::DenseTensor *> return_tensor_;
  std::shared_ptr<GCUEngineExecutor> executor_ = nullptr;
};

}  // namespace custom_engine
