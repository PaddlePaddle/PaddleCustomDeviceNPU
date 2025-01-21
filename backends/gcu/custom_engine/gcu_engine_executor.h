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

#include "custom_engine/ir_translator/utils/utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace custom_engine {

class GCUEngineExecutor {
 public:
  GCUEngineExecutor(topsExecutable_t tops_exec,
                    const std::vector<const phi::DenseTensor *> &tensor_args,
                    const std::vector<phi::DenseTensor *> &return_tensor)
      : tops_exec_(tops_exec),
        tensor_args_(tensor_args),
        return_tensor_(return_tensor) {
    Init();
  }
  ~GCUEngineExecutor() {}

  void Init();

  void Run(const phi::CustomContext &dev_ctx);

 private:
  topsExecutable_t tops_exec_ = nullptr;  // Not owned
  std::vector<const phi::DenseTensor *> tensor_args_;
  std::vector<phi::DenseTensor *> return_tensor_;

  std::vector<phi::DenseTensor> tensor_args_device_;
};

}  // namespace custom_engine
