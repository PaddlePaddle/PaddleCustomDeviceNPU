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
#include "custom_engine/gcu_engine.h"
#include "custom_engine/gcu_engine_compiler.h"
#include "paddle/fluid/framework/new_executor/instruction/custom_engine_instruction.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation_utils.h"

namespace {
using DenseTensorType = pir::DenseTensorType;
using AllocatedDenseTensorType = paddle::dialect::AllocatedDenseTensorType;
using SelectedRowsType = paddle::dialect::SelectedRowsType;
using AllocatedSelectedRowsType = paddle::dialect::AllocatedSelectedRowsType;
using DenseTensorArrayType = paddle::dialect::DenseTensorArrayType;
using AllocatedDenseTensorArrayType =
    paddle::dialect::AllocatedDenseTensorArrayType;
using SparseCooTensorType = paddle::dialect::SparseCooTensorType;
using SparseCsrTensorType = paddle::dialect::SparseCsrTensorType;

template <class IrType1, class IrType2>
static pir::Type CreatType(pir::Type type,
                           const phi::Place& place,
                           pir::Type out_dtype,
                           pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

static pir::Type BuildOutputType(pir::Type type,
                                 const phi::Place& place,
                                 pir::IrContext* ctx) {
  if (type.isa<DenseTensorType>()) {
    auto out_dtype = type.dyn_cast<DenseTensorType>().dtype();
    return CreatType<DenseTensorType, AllocatedDenseTensorType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<SelectedRowsType>()) {
    auto out_dtype = type.dyn_cast<SelectedRowsType>().dtype();
    return CreatType<SelectedRowsType,
                     paddle::dialect::AllocatedSelectedRowsType>(
        type, place, out_dtype, ctx);
  } else if (type.isa<DenseTensorArrayType>()) {
    auto array_type = type.dyn_cast<DenseTensorArrayType>();
    return AllocatedDenseTensorArrayType::get(ctx,
                                              place,
                                              array_type.dtype(),
                                              array_type.dims(),
                                              array_type.data_layout());
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "BuildOutputType only support DenseTensorType, SelectedRowsType, "
        "and DenseTensorArrayType"));
  }
}

void PushBackOutputTypes(pir::IrContext* ctx,
                         pir::Operation* op_item,
                         const pir::Type& origin_type,
                         const phi::Place& out_place,
                         const phi::KernelKey& kernel_key,
                         std::vector<pir::Type>* op_output_types) {
  auto result_type = origin_type;
  if (!result_type) {
    op_output_types->push_back(result_type);
  } else if (result_type.isa<DenseTensorType>() ||
             result_type.isa<SelectedRowsType>() ||
             result_type.isa<DenseTensorArrayType>() ||
             result_type.isa<SparseCooTensorType>() ||
             result_type.isa<SparseCsrTensorType>()) {
    op_output_types->push_back(BuildOutputType(result_type, out_place, ctx));

  } else if (result_type.isa<pir::VectorType>()) {
    std::vector<pir::Type> vec_inner_types;
    auto base_types = result_type.dyn_cast<pir::VectorType>().data();
    for (auto& base_type : base_types) {
      if (base_type) {
        if (base_type.isa<DenseTensorType>() ||
            base_type.isa<SelectedRowsType>()) {
          vec_inner_types.push_back(BuildOutputType(base_type, out_place, ctx));
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "only support dense tensor and selected rows in vector type "
              "for now"));
        }
      } else {
        // NOTE(phlrain), kernel not support a nullptr in output
        pir::Type fp32_dtype = pir::Float32Type::get(ctx);
        phi::DDim dims = {};
        phi::DataLayout data_layout = phi::DataLayout::NCHW;
        phi::LegacyLoD lod = {{}};
        size_t offset = 0;
        auto dense_tensor_dtype = DenseTensorType::get(
            ctx, fp32_dtype, dims, data_layout, lod, offset);
        auto allocated_dense_tensor_dtype =
            AllocatedDenseTensorType::get(ctx, out_place, dense_tensor_dtype);
        vec_inner_types.push_back(allocated_dense_tensor_dtype);
      }
    }

    pir::Type t1 = pir::VectorType::get(ctx, vec_inner_types);
    op_output_types->push_back(t1);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Result type only support DenseTensorType, SelectedRowType, "
        "SparseCooTensorType, SparseCsrTensorType and "
        "VectorType"));
  }
}
}  // namespace

C_Status RegisterCustomEngineOp() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Dialect* custom_engine_dialect =
      ctx->GetOrRegisterDialect<paddle::dialect::CustomEngineDialect>();
  PADDLE_ENFORCE_NOT_NULL(custom_engine_dialect,
                          "Failed to register CustomEngineDialect.");
  ctx->RegisterOpInfo(custom_engine_dialect,
                      pir::TypeId::get<custom_engine::CustomEngineOp>(),
                      custom_engine::CustomEngineOp::name(),
                      custom_engine::CustomEngineOp::interface_set(),
                      custom_engine::CustomEngineOp::GetTraitSet(),
                      custom_engine::CustomEngineOp::attributes_num,
                      custom_engine::CustomEngineOp::attributes_name,
                      custom_engine::CustomEngineOp::VerifySigInvariants,
                      custom_engine::CustomEngineOp::VerifyRegionInvariants);
  VLOG(3) << "Register CustomEngineOp successfully.";
  return C_SUCCESS;
}

C_Status CustomEngineOpLower(C_CustomEngineLowerParams* lower_param) {
  VLOG(3) << "Enter CustomEngineOpLower.";
  // get lower params
  pir::IrContext* ctx =
      reinterpret_cast<pir::IrContext*>(lower_param->ir_context);
  pir::Operation* op_item =
      reinterpret_cast<pir::Operation*>(lower_param->operation);
  phi::KernelKey* kernel_key =
      reinterpret_cast<phi::KernelKey*>(lower_param->kernel_key);
  phi::Place* place = reinterpret_cast<phi::Place*>(lower_param->place);
  std::unordered_map<pir::Operation*, pir::Operation*>* map_op_pair =
      reinterpret_cast<std::unordered_map<pir::Operation*, pir::Operation*>*>(
          lower_param->map_op_pair);
  std::unordered_map<pir::Value, pir::Value>* map_value_pair =
      reinterpret_cast<std::unordered_map<pir::Value, pir::Value>*>(
          lower_param->map_value_pair);
  pir::Block* block = reinterpret_cast<pir::Block*>(lower_param->block);

  // Prepare output types
  std::vector<pir::Type> op_output_types;

  for (size_t i = 0; i < op_item->num_results(); ++i) {
    phi::Place out_place = phi::TransToPhiPlace(kernel_key->backend());
    PushBackOutputTypes(ctx,
                        op_item,
                        op_item->result(i).type(),
                        out_place,
                        *kernel_key,
                        &op_output_types);
  }

  // Prepare input
  std::vector<pir::Value> vec_inputs;

  for (size_t i = 0; i < op_item->num_operands(); ++i) {
    auto cur_in = op_item->operand_source(i);
    PADDLE_ENFORCE_EQ(
        map_value_pair->count(cur_in),
        true,
        common::errors::PreconditionNotMet(
            "[%d]'s input of [%s] op MUST in map pair", i, op_item->name()));

    auto new_in = map_value_pair->at(cur_in);

    vec_inputs.push_back(new_in);
  }

  // Prepare attr
  std::unordered_map<std::string, pir::Attribute> op_attribute;
  auto op_attr_map = op_item->attributes();
  for (auto& map_item : op_attr_map) {
    op_attribute.emplace(map_item.first, map_item.second);
  }
  op_attribute["op_name"] = pir::StrAttribute::get(ctx, op_item->name());

  pir::OpInfo custom_engine_op_info =
      ctx->GetRegisteredOpInfo(custom_engine::CustomEngineOp::name());

  pir::Operation* op = pir::Operation::Create(
      vec_inputs, op_attribute, op_output_types, custom_engine_op_info, 1);
  op->set_attribute("origin_id", pir::Int64Attribute::get(ctx, op->id()));
  VLOG(3) << "CustomEngineOpLower create custom_engine_op";

  VLOG(3) << "CustomEngineOpLower get op_item subgraph block.";
  pir::Region& op_item_region = op_item->region(0);
  PADDLE_ENFORCE_EQ(
      op_item_region.empty(),
      false,
      ::common::errors::Unavailable(
          "Required CustomEngineOp's region must not be emptpy."));
  pir::Block* sub_graph_block = &(op_item_region.front());

  VLOG(3) << "CustomEngineOpLower set new op subgraph block.";
  pir::Region& region = op->region(0);
  if (region.empty()) {
    region.emplace_back();
  }
  pir::Block* op_block = &(region.front());

  // process subgraph block
  paddle::dialect::ProcessBlock(
      *place, sub_graph_block, op_block, ctx, map_op_pair, map_value_pair);

  if (VLOG_IS_ON(3)) {
    std::stringstream ss;
    ss << "CustomEngineOpLower new op:";
    op->Print(ss);
    VLOG(3) << ss.str();
  }

  (*map_op_pair)[op_item] = op;

  // only deal with single output
  if (op_item->num_results() > 0) {
    for (size_t i = 0; i < op_item->num_results(); ++i) {
      (*map_value_pair)[op_item->result(i)] = op->result(i);
    }
  }
  block->push_back(op);
  VLOG(3) << "CustomEngineOpLower successfully.";
  return C_SUCCESS;
}

C_Status GraphEngineBuild(C_CustomEngineInstruction instruction) {
  VLOG(3) << "Enter GraphEngineBuild.";
  paddle::framework::CustomEngineInstruction* instruction_ =
      reinterpret_cast<paddle::framework::CustomEngineInstruction*>(
          instruction);
  pir::Operation* op = instruction_->Operation();
  const phi::KernelContext& kernel_context = instruction_->KernelContext();
  phi::KernelContext kernel_ctx = kernel_context;
  auto engine_inputs = instruction_->GetEngineInputs();
  auto engine_outputs = instruction_->GetEngineOutputs();
  auto engine_value_to_tensors = instruction_->GetEngineValueToTensors();
  auto engine_value_to_var_names = instruction_->GetEngineValueToVarNames();

  // NOTES: The memory is managed by CustomEngineInstruction, and we provide a
  // release interface here.
  custom_engine::GCUEngine* gcu_engine = new custom_engine::GCUEngine();
  auto gcu_engine_deleter = [](void* ptr) {
    custom_engine::GCUEngine* gcu_engine =
        static_cast<custom_engine::GCUEngine*>(ptr);

    if (gcu_engine != nullptr) {
      delete gcu_engine;
    } else {
      PADDLE_THROW(phi::errors::PreconditionNotMet("gcu_engine is nullptr"));
    }
  };

  std::string engine_key =
      "GCUEngine_" +
      std::to_string(reinterpret_cast<std::uintptr_t>(instruction));
  custom_engine::GCUEngineCompiler gcu_compiler(kernel_ctx,
                                                op,
                                                engine_inputs,
                                                engine_outputs,
                                                engine_value_to_tensors,
                                                engine_value_to_var_names,
                                                engine_key);
  gcu_compiler.Compile(gcu_engine);

  instruction_->SetCustomEngine(reinterpret_cast<void*>(gcu_engine));
  instruction_->SetCustomEngineDeleter(gcu_engine_deleter);
  VLOG(3) << "GraphEngineBuild successfully.";

  return C_SUCCESS;
}

C_Status GraphEngineExecute(C_CustomEngineInstruction instruction) {
  VLOG(3) << "Enter GraphEngineExecute.";
  paddle::framework::CustomEngineInstruction* instruction_ =
      reinterpret_cast<paddle::framework::CustomEngineInstruction*>(
          instruction);
  custom_engine::GCUEngine* gcu_engine =
      reinterpret_cast<custom_engine::GCUEngine*>(instruction_->CustomEngine());
  PADDLE_ENFORCE_NOT_NULL(gcu_engine, "GCUEngine is nullptr.");

  auto* dev_ctx =
      static_cast<phi::CustomContext*>(phi::DeviceContextPool::Instance().Get(
          instruction_->DeviceContext().GetPlace()));

  gcu_engine->Run(*dev_ctx);
  VLOG(3) << "GraphEngineExecute successfully.";
  return C_SUCCESS;
}

void InitPluginCustomEngine(CustomEngineParams* params) {
  memset(reinterpret_cast<void*>(params->interface),
         0,
         sizeof(C_CustomEngineInterface));

  params->interface->register_custom_engine_op = RegisterCustomEngineOp;
  params->interface->graph_engine_build = GraphEngineBuild;
  params->interface->graph_engine_execute = GraphEngineExecute;
  params->interface->custom_engine_op_lower = CustomEngineOpLower;
}
