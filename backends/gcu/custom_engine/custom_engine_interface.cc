// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "custom_engine/custom_engine_interface.h"

#include "custom_engine/custom_engine_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"

namespace paddle {
namespace dialect {

void RegisterCustomEngineOp() {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Dialect *custom_engine_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::CustomEngineDialect>();
  PADDLE_ENFORCE_NOT_NULL(custom_engine_dialect,
                          "Failed to register CustomEngineDialect.");
  ctx->RegisterOpInfo(custom_engine_dialect,
                      pir::TypeId::get<paddle::dialect::CustomEngineOp>(),
                      paddle::dialect::CustomEngineOp::name(),
                      paddle::dialect::CustomEngineOp::interface_set(),
                      paddle::dialect::CustomEngineOp::GetTraitSet(),
                      paddle::dialect::CustomEngineOp::attributes_num,
                      paddle::dialect::CustomEngineOp::attributes_name,
                      paddle::dialect::CustomEngineOp::VerifySigInvariants,
                      paddle::dialect::CustomEngineOp::VerifyRegionInvariants);
  VLOG(3) << "Register CustomEngineOp successfully.";
}

}  // namespace dialect
}  // namespace paddle
