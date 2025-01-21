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

#include <vector>

#include "custom_engine/ir_translator/translator_registry.h"

namespace custom_engine {

static GcuOpPtr TranslateYield(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  size_t output_num = gcu_op_inputs.size();
  if (output_num > 1) {
    std::vector<builder::Op> outputs;
    for (size_t i = 0; i < output_num; ++i) {
      outputs.emplace_back(*(gcu_op_inputs[i][0]));
    }
    builder::Op result = builder::Tuple(outputs);
    return std::make_shared<GcuOp>(result);
  } else if (output_num == 1) {
    return gcu_op_inputs[0][0];
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet("Not support now."));
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(cf_yield, custom_engine::TranslateYield)
