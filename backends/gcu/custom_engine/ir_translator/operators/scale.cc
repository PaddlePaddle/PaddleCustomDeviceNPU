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

static GcuOpPtr TranslateScale(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = gcu_op_inputs[0][0];
  const auto &attrs = op->attributes();
  bool bias_after_scale =
      attrs.at("bias_after_scale").dyn_cast<pir::BoolAttribute>().data();

  builder::Op scale_op;
  if (gcu_op_inputs.size() == 2) {  // with scale tensor
    scale_op = *(gcu_op_inputs[1][0]);
  } else {
    float scale = attrs.at("scale").dyn_cast<::pir::FloatAttribute>().data();
    scale_op = builder::FullLike(*input, scale);
  }
  float bias = attrs.at("bias").dyn_cast<pir::FloatAttribute>().data();
  auto bias_op = builder::FullLike(*input, bias);
  if (bias_after_scale) {
    return std::make_shared<GcuOp>((*input) * scale_op + bias_op);
  } else {
    return std::make_shared<GcuOp>(((*input) + bias_op) * scale_op);
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_scale, custom_engine::TranslateScale)
REGISTER_OP_TRANSLATOR(pd_op_scale_, custom_engine::TranslateScale)
