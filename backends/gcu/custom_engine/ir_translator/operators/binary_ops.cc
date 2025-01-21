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

#define DEFINE_BINARY_TRANS_FUNC(func)                                   \
  static GcuOpPtr TranslateBinaryOps##func(                              \
      GcuBuilderPtr gcu_builder,                                         \
      const pir::Operation *op,                                          \
      const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {         \
    PADDLE_ENFORCE_EQ(gcu_op_inputs.size(),                              \
                      2,                                                 \
                      common::errors::PreconditionNotMet(                \
                          "Intput op num check failed, op: %s, num:%zu", \
                          std::string(#func).c_str(),                    \
                          gcu_op_inputs.size()));                        \
    auto lhs = *(gcu_op_inputs[0][0]);                                   \
    auto rhs = *(gcu_op_inputs[1][0]);                                   \
    return std::make_shared<GcuOp>(builder::func(lhs, rhs));             \
  }

DEFINE_BINARY_TRANS_FUNC(Add)
DEFINE_BINARY_TRANS_FUNC(Sub)
DEFINE_BINARY_TRANS_FUNC(Mul)
DEFINE_BINARY_TRANS_FUNC(Div)
DEFINE_BINARY_TRANS_FUNC(Greater)
DEFINE_BINARY_TRANS_FUNC(GreaterEqual)
DEFINE_BINARY_TRANS_FUNC(Less)
DEFINE_BINARY_TRANS_FUNC(LessEqual)

#undef DEFINE_BINARY_TRANS_FUNC

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_add, custom_engine::TranslateBinaryOpsAdd)
REGISTER_OP_TRANSLATOR(pd_op_add_, custom_engine::TranslateBinaryOpsAdd)
REGISTER_OP_TRANSLATOR(pd_op_subtract, custom_engine::TranslateBinaryOpsSub)
REGISTER_OP_TRANSLATOR(pd_op_multiply, custom_engine::TranslateBinaryOpsMul)
REGISTER_OP_TRANSLATOR(pd_op_divide, custom_engine::TranslateBinaryOpsDiv)
REGISTER_OP_TRANSLATOR(pd_op_greater_than,
                       custom_engine::TranslateBinaryOpsGreater)
REGISTER_OP_TRANSLATOR(pd_op_greater_equal,
                       custom_engine::TranslateBinaryOpsGreaterEqual)
REGISTER_OP_TRANSLATOR(pd_op_less_than, custom_engine::TranslateBinaryOpsLess)
REGISTER_OP_TRANSLATOR(pd_op_less_equal,
                       custom_engine::TranslateBinaryOpsLessEqual)
