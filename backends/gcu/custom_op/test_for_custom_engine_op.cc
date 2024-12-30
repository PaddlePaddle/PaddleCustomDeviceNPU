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
#include "paddle/pir/include/core/program.h"

namespace {
void TestCustomEngineOp() {
  VLOG(0) << "Start to run TestCustomEngineOp.";
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  pir::FloatAttribute fp_attr = builder.float_attr(2.0f);
  pir::Float32Type fp32_type = builder.float32_type();
  pir::DenseTensorType const_out_type =
      pir::DenseTensorType::get(ctx,
                                pir::Float32Type::get(ctx),
                                phi::DDim(std::vector<int64_t>{2, 2}.data(), 2),
                                phi::DataLayout::kNCHW,
                                phi::LoD(),
                                0);

  auto const_op1 =
      builder.Build<pir::ConstantOp>(builder.float_attr(2.0f), const_out_type);
  auto const_op2 =
      builder.Build<pir::ConstantOp>(builder.float_attr(6.0f), const_out_type);

  auto buildin_combine_op = builder.Build<pir::CombineOp>(
      std::vector<pir::Value>{const_op1.result(0), const_op2.result(0)});

  pir::OpInfo custom_engine_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::CustomEngineOp::name());

  std::vector<pir::Type> out_types;
  out_types.emplace_back(
      pir::DenseTensorType::get(ctx,
                                pir::Float32Type::get(ctx),
                                phi::DDim(std::vector<int64_t>{2, 2}.data(), 2),
                                phi::DataLayout::kNCHW,
                                phi::LoD(),
                                0));
  pir::Type out_vector_type = pir::VectorType::get(ctx, out_types);
  std::vector<pir::Type> output_types = {out_vector_type};

  pir::AttributeMap attribute_map;
  std::vector<pir::Attribute> val;
  val.emplace_back(pir::StrAttribute::get(ctx, "input_0"));
  val.emplace_back(pir::StrAttribute::get(ctx, "input_1"));
  attribute_map.insert({"input_names", pir::ArrayAttribute::get(ctx, val)});
  std::vector<pir::Attribute> out_val;
  out_val.emplace_back(pir::StrAttribute::get(ctx, "output_0"));
  out_val.emplace_back(pir::StrAttribute::get(ctx, "output_1"));
  attribute_map.insert(
      {"output_names", pir::ArrayAttribute::get(ctx, out_val)});

  pir::Operation* op1 = pir::Operation::Create({buildin_combine_op.result(0)},
                                               attribute_map,
                                               output_types,
                                               custom_engine_op_info);

  // Test custom operation printer
  std::stringstream ss1;
  op1->Print(ss1);
  VLOG(0) << "TestCustomEngineOp op1:" << ss1.str();

  builder.Insert(op1);

  auto op2 = builder.Build<paddle::dialect::CustomEngineOp>(
      buildin_combine_op.result(0),
      std::vector<std::string>{"input_0", "input_1"},
      std::vector<std::string>{"output_0"},
      std::vector<std::vector<int64_t>>{{2, 2}},
      std::vector<phi::DataType>{phi::DataType::FLOAT32});

  // Test custom operation printer
  std::stringstream ss2;
  op2->Print(ss2);
  VLOG(0) << "TestCustomEngineOp op2:" << ss2.str();

  PADDLE_ENFORCE_EQ(
      block->size(), 5, "Block size should be 5, bet get %zu.", block->size());

  VLOG(0) << "TestCustomEngineOp program:\n" << program;
  VLOG(0) << "Run TestCustomEngineOp successfully.";
}

void RunTestCustomEngineOp() {
  paddle::dialect::RegisterCustomEngineOp();
  TestCustomEngineOp();
}
}  // namespace

std::vector<paddle::Tensor> TestForCustomEngineOp(const paddle::Tensor& x) {
  RunTestCustomEngineOp();
  return {x};
}

std::vector<std::vector<int64_t>> TestForCustomEngineOpInferShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> TestForCustomEngineOpInferDtype(
    const paddle::DataType& x_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(test_for_custom_engine_op)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(TestForCustomEngineOp))
    .SetInferShapeFn(PD_INFER_SHAPE(TestForCustomEngineOpInferShape))
    .SetInferDtypeFn(
        PD_INFER_DTYPE(TestForCustomEngineOpInferDtype));  // neccessary if the
                                                           // op has muti_inputs
