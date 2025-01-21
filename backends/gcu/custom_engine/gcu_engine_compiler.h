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

#include "custom_engine/gcu_engine.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/pir/include/core/operation.h"

namespace custom_engine {

class GCUEngineCompiler {
 public:
  GCUEngineCompiler(
      const phi::KernelContext& kernel_context,
      pir::Operation* op,
      const std::vector<pir::Value>& engine_inputs,
      const std::vector<pir::Value>& engine_outputs,
      const std::unordered_map<pir::Value, std::vector<phi::DenseTensor*>>&
          engine_value_to_tensors,
      const std::unordered_map<pir::Value, std::vector<std::string>>&
          engine_value_to_var_names,
      const std::string& engine_key = "GCUEngineCompiler_default");
  ~GCUEngineCompiler() {}

  // NOTES: This function applies for memory and is released by the caller.
  void Compile(GCUEngine* gcu_engine);

 private:
  class GCUEngineCompilerImpl;
  std::shared_ptr<GCUEngineCompilerImpl> impl_ = nullptr;
};

}  // namespace custom_engine
