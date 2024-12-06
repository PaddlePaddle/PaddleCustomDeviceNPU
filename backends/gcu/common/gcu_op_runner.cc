// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "common/gcu_op_runner.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/equivalence_trans/all_ops.h"
#include "backend/executor/single_op_executor.h"
#include "backend/executor/tops_compiler.h"
#include "backend/utils/gcu_op_desc.h"
#include "backend/utils/utils.h"
#include "common/gcu_funcs.h"
#include "runtime/flags.h"

namespace custom_kernel {
namespace {
const char* const kGcuDebugPrint = "GCU_DEBUG_PRINT_EAGER_TENSOR";
const char* const kRunningModeSerial = "serial";
const char* const kPlaceHolder = " ";

static std::set<std::string> kUnusedArchetype = {"ReserveSpace"};
}  // namespace

using GcuOpDesc = backend::GcuOpDesc;
using EquivalenceTransformer = backend::EquivalenceTransformer;
using TransformUtil = backend::TransformUtil;

static std::vector<std::string> ParseAttr(std::string attr_value) {
  std::vector<std::string> out_var_names;
  if (attr_value == "") return out_var_names;
  const char* divided_symbol = ";";
  size_t pos = attr_value.find(divided_symbol);
  if (pos == attr_value.npos) {
    out_var_names.emplace_back(attr_value);
  }
  while (pos != attr_value.npos) {
    std::string sub_str = attr_value.substr(0, pos);
    out_var_names.emplace_back(sub_str);
    attr_value = attr_value.substr(pos + 1, attr_value.size());
    pos = attr_value.find(divided_symbol);
  }
  if (attr_value.length() != 0) {
    out_var_names.emplace_back(attr_value);
  }
  return out_var_names;
}

void GcuOpRunner::GetInputsAndOutputs(
    const GcuExecutionContext& ctx,
    std::vector<TensorNameValuePair>& input_vars,     // NOLINT
    std::vector<TensorNameValuePair>& output_vars) {  // NOLINT
  VLOG(6) << "op " << ctx.Type() << " start to get inputs and outputs ";

  const auto& all_input_names = ctx.AllInputNames();
  const auto& all_inputs = ctx.AllInputs();

  const auto& all_output_names = ctx.AllOutputNames();
  const auto& all_outputs = ctx.AllOutputs();

  VLOG(6) << "op " << ctx.Type() << " start get inputs and outputs ";

  for (auto iter = all_input_names.begin(); iter != all_input_names.end();
       ++iter) {
    auto& names = iter->second;
    auto& tensors = all_inputs.at(iter->first);
    for (size_t i = 0; i < names.size(); ++i) {
      input_vars.push_back(TensorNameValuePair(names[i], tensors[i]));
    }
  }

  for (auto iter = all_output_names.begin(); iter != all_output_names.end();
       ++iter) {
    auto& names = iter->second;
    auto& tensors = all_outputs.at(iter->first);
    for (size_t i = 0; i < names.size(); ++i) {
      output_vars.push_back(TensorNameValuePair(names[i], tensors[i]));
    }
  }

  VLOG(6) << "op " << ctx.Type() << " get inputs and outputs finished.";
}

std::string GcuOpRunner::AttrString(const GcuExecutionContext& ctx) {
  auto attrs_map = ctx.Attrs();
  std::ostringstream os;
  os << "attrs:[ ";
  for (auto iter : attrs_map) {
    auto& value = iter.second;
    os << iter.first << ": ";
    if (value.type() == typeid(paddle::blank)) {
      os << "paddle::blank";
    } else if (value.type() == typeid(int)) {
      os << PADDLE_GET_CONST(int, attrs_map.at(iter.first));
    } else if (value.type() == typeid(float)) {
      os << PADDLE_GET_CONST(float, attrs_map.at(iter.first));
    } else if (value.type() == typeid(std::string)) {
      os << PADDLE_GET_CONST(std::string, attrs_map.at(iter.first));
    } else if (value.type() == typeid(bool)) {
      os << PADDLE_GET_CONST(bool, attrs_map.at(iter.first));
    } else if (value.type() == typeid(int64_t)) {
      os << PADDLE_GET_CONST(int64_t, attrs_map.at(iter.first));
    } else if (value.type() == typeid(std::vector<int>)) {
      os << backend::VectorToString(
          PADDLE_GET_CONST(std::vector<int>, attrs_map.at(iter.first)));
    } else if (value.type() == typeid(std::vector<float>)) {
      os << backend::VectorToString(
          PADDLE_GET_CONST(std::vector<float>, attrs_map.at(iter.first)));
    } else if (value.type() == typeid(std::vector<std::string>)) {
      os << backend::VectorToString(
          PADDLE_GET_CONST(std::vector<std::string>, attrs_map.at(iter.first)));
    } else if (value.type() == typeid(std::vector<bool>)) {
      os << backend::VectorToString(
          PADDLE_GET_CONST(std::vector<bool>, attrs_map.at(iter.first)));
    } else if (value.type() == typeid(std::vector<int64_t>)) {
      os << backend::VectorToString(
          PADDLE_GET_CONST(std::vector<int64_t>, attrs_map.at(iter.first)));
    } else if (value.type() == typeid(std::vector<double>)) {
      os << backend::VectorToString(
          PADDLE_GET_CONST(std::vector<double>, attrs_map.at(iter.first)));
    } else {
      // BlockDesc* std::vector<BlockDesc*> unsupport for string
      uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
      os << std::to_string(ms);
    }
    os << "; ";
  }
  os << " ]";
  return os.str();
}

std::string GcuOpRunner::BuildSignature(const GcuExecutionContext& ctx) {
  auto to_signature = [&](std::ostringstream& os,
                          const TensorNameMap& tensor_names,
                          const TensorValueMap& tensor_values) {
    for (auto tensor_name_iter = tensor_names.begin();
         tensor_name_iter != tensor_names.end();
         ++tensor_name_iter) {
      os << tensor_name_iter->first << "[ ";

      auto& rumtime_names = tensor_name_iter->second;
      auto& runtime_vars = tensor_values.at(tensor_name_iter->first);

      for (size_t i = 0; i < rumtime_names.size(); ++i) {
        auto* src_tensor = runtime_vars[i];
        PADDLE_ENFORCE_NOT_NULL(src_tensor);
        os << "["
           << "dims: " << src_tensor->dims().to_str()
           << ", dtype: " << src_tensor->dtype() << ", index: " << i << "]; ";
      }
      os << " ]; ";
    }
  };

  const auto& all_input_names = ctx.AllInputNames();
  const auto& all_inputs = ctx.AllInputs();

  const auto& all_output_names = ctx.AllOutputNames();
  const auto& all_outputs = ctx.AllOutputs();

  std::ostringstream os;
  os << "input: [";
  to_signature(os, all_input_names, all_inputs);
  os << "]; ";

  os << "output: [";
  to_signature(os, all_output_names, all_outputs);
  os << "]; ";

  os << AttrString(ctx) << "\n";

  return os.str();
}

void GcuOpRunner::CompileAndRun(
    const GcuExecutionContext& ctx,
    const std::vector<TensorNameValuePair>& input_vars,
    const std::vector<TensorNameValuePair>& output_vars,
    bool tensor_split) {  // NOLINT

  VLOG(3) << "op " << ctx.Type() << " start to run program ";

  std::hash<std::string> hasher;
  std::string signature = BuildSignature(ctx);

  VLOG(3) << "[JIT_KERNEL] " << ctx.Type() << " signature: " << signature;

  auto program_key = std::to_string(hasher(signature + ctx.Type()));

  std::vector<DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  for (auto name_value : input_vars) {
    input_names.push_back(name_value.first);
    inputs.push_back(name_value.second);
  }
  for (auto name_value : output_vars) {
    output_names.push_back(name_value.first);
    outputs.push_back(name_value.second);
  }

  VLOG(6) << "op " << ctx.Type() << ", input_names: " << input_names.size()
          << ", output_names: " << output_names.size()
          << ", inputs: " << inputs.size() << ", outputs: " << outputs.size();

  auto manager = backend::SingleOpGcuExecutorManager::GetInstance();
  auto gcu_exec = manager->Find(program_key);
  if (gcu_exec == nullptr) {
    CompileExecutable(
        ctx, program_key, inputs, outputs, input_names, output_names);
  }

  RunExecutableSync(ctx,
                    program_key,
                    inputs,
                    outputs,
                    input_names,
                    output_names,
                    tensor_split);

  VLOG(3) << "op " << ctx.Type() << " run program finished.";
}

GcuOpPtr GcuOpRunner::AddGteOp(const DenseTensor* tensor,
                               const std::string& tensor_name,
                               const GcuOpPtr& input) {
  auto attr_out_var_names = input->GetAttribute(backend::kAttrOpOutVarName);
  PADDLE_ENFORCE_NE(
      attr_out_var_names == builder::Attribute(""),
      true,
      phi::errors::NotFound("lack of attr [%s] for gcu tuple op, please check.",
                            backend::kAttrOpOutVarName));
  std::string out_var_names = attr_out_var_names.GetValueAsString();
  auto list_out_var_names = ParseAttr(out_var_names);
  int32_t idx = 0;
  VLOG(3) << out_var_names;
  for (const auto& name : list_out_var_names) {
    if (name != tensor_name) {
      idx++;
    } else {
      break;
    }
  }
  auto shape = phi::vectorize(tensor->dims());
  auto ptype = TransformUtil::ConvertDataType(tensor->dtype());
  builder::Type input_type(shape, ptype);
  return std::make_shared<GcuOp>(builder::GetTupleElement(*input, idx));
}

void GcuOpRunner::CompileExecutable(
    const GcuExecutionContext& ctx,
    const std::string& program_key_in,
    const std::vector<DenseTensor*>& inputs,
    const std::vector<DenseTensor*>& outputs,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {  // NOLINT
  auto op_type = ctx.Type();
  VLOG(3) << "OpType " << op_type << " start to compile. ";
  std::string program_key = program_key_in;
  std::map<std::string, GcuOpPtr> gcu_op_cache;
  std::map<std::string, DenseTensor*> tensor_cache;

  GcuBuilderPtr builder = std::make_shared<GcuBuilder>();
  PADDLE_ENFORCE_NE(
      builder,
      nullptr,
      phi::errors::Fatal("builfer is nullptr, graph:%s", program_key.c_str()));
  builder->SetShapeInference(true);

  auto func =
      EquivalenceTransformer::GetInstance().Get(op_type, backend::INSENSITIVE);

  PADDLE_ENFORCE_NE(
      func,
      nullptr,
      phi::errors::NotFound(
          "OpType %s is not register gcu op convert func, please check.",
          op_type.c_str()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& name = input_names[i];
    auto& tensor = inputs[i];
    if (gcu_op_cache.count(name) > 0) {
      PADDLE_THROW(
          phi::errors::InvalidArgument("input: %s appears twice ", name));
    }

    auto shape = phi::vectorize(tensor->dims());
    auto ptype = TransformUtil::ConvertDataType(tensor->dtype());
    builder::Type input_type(shape, ptype);
    GcuOpPtr gcu_op = nullptr;
    if (!tensor->initialized()) {
      VLOG(6) << "OpType " << op_type << " at: [" << i
              << "] create const: " << name << " dims: " << tensor->dims();
      std::vector<int64_t> const_data(tensor->numel(), 0);
      gcu_op = std::make_shared<GcuOp>(builder::Const(
          builder, static_cast<void*>(const_data.data()), input_type));
    } else {
      VLOG(6) << "OpType " << op_type << " at: [" << i
              << "] create input: " << name << " dims: " << tensor->dims();
      gcu_op = std::make_shared<GcuOp>(builder->CreateInput(input_type));
    }
    gcu_op_cache[name] = gcu_op;
    tensor_cache[name] = tensor;
  }

  for (size_t i = 0; i < output_names.size(); ++i) {
    tensor_cache[output_names[i]] = outputs[i];
  }

  const auto& all_input_names = ctx.AllInputNames();
  const auto& all_inputs = ctx.AllInputs();

  const auto& all_output_names = ctx.AllOutputNames();
  const auto& all_outputs = ctx.AllOutputs();

  std::map<std::string, std::vector<GcuOpPtr>> input_ops;
  for (const auto& e : all_input_names) {
    if (kUnusedArchetype.count(e.first) > 0) {
      continue;
    }
    if (e.second.empty()) {
      continue;
    }
    std::vector<GcuOpPtr> v;
    for (std::string n : e.second) {
      auto gcu_op = gcu_op_cache[n];
      if (gcu_op == nullptr) {
        VLOG(2) << "[WARN]Can not find transfered gcu op by"
                   "input name "
                << n;
      }
      auto gcu_shape_str =
          TransformUtil::GetShapeStr(gcu_op->GetType().GetShape());
      VLOG(6) << "Input Archetype name: " << e.first << " in name:" << n
              << " shape:" << gcu_shape_str;
      v.push_back(gcu_op);
    }
    input_ops[e.first] = v;
  }
  bool refresh_program_key = false;
  auto op_desc = std::make_shared<GcuOpDesc>(GcuOpDesc(
      ctx.Type(), ctx.AllInputNames(), ctx.AllOutputNames(), ctx.Attrs()));
  VLOG(10) << "Transfered to gcu node start, type:" << op_type;
  GcuOpPtr op = func(builder, op_desc, input_ops, kRunningModeSerial);
  VLOG(10) << "Transfered to gcu node end, type:" << op_type;

  PADDLE_ENFORCE_NE(op,
                    nullptr,
                    phi::errors::Fatal(
                        "op type:%s transfered gcu node should not be nullptr!",
                        op_type.c_str()));
  gcu_op_cache[op_type] = op;
  bool is_tuple_out = op->GetType().IsTuple();
  // check tuple condition same with pd
  if (is_tuple_out) {
    size_t gcu_output_num = op->GetType().GetTupleSize();
    size_t valid_output_counter = 0;
    for (const auto& e : op_desc->Outputs()) {
      if (kUnusedArchetype.count(e.first) > 0) {
        continue;
      }
      if (!e.second.empty()) {
        VLOG(6) << "Out Archetype name:" << e.first;
        for (const auto& p : e.second) {
          VLOG(6) << "    correspond var name:" << p;
          valid_output_counter++;
        }
      }
    }
    if (VLOG_IS_ON(10) && valid_output_counter != gcu_output_num) {
      builder->Dump();
    }
    PADDLE_ENFORCE_EQ(
        valid_output_counter,
        gcu_output_num,
        phi::errors::PreconditionNotMet(
            "op type:%s paddle valid output size is %u, but gcu is %u",
            op_type.c_str(),
            valid_output_counter,
            gcu_output_num));
  }
  if (!is_tuple_out) {
    for (const auto& e : op_desc->Outputs()) {
      if (kUnusedArchetype.count(e.first) > 0) {
        continue;
      }
      if (e.second.empty()) {
        continue;
      }
      std::string weight_name = "";
      for (std::string n : e.second) {
        VLOG(6) << "Output Archetype name: " << e.first << " out name:" << n;
        auto out_name = n;
        gcu_op_cache[out_name] = op;
        // for shape infer check
        auto gcu_shape = op->GetType().GetShape();
        auto tensor = tensor_cache[out_name];
        auto paddle_shape = phi::vectorize(tensor->dims());
        if (VLOG_IS_ON(10) && gcu_shape.size() != paddle_shape.size()) {
          builder->Dump();
        }
        // normalize scalar shape process, [] -> [1]
        if (gcu_shape.empty()) {
          gcu_shape = {1};
        }
        if (paddle_shape.empty()) {
          paddle_shape = {1};
        }
        PADDLE_ENFORCE_EQ(gcu_shape.size(),
                          paddle_shape.size(),
                          phi::errors::PreconditionNotMet(
                              "op type:%s transfered gcu node "
                              "should have same rank! "
                              "but paddle rank is %u, gcu is %u, out name "
                              "is %s, paddle shape:%s, gcu shape:%s",
                              op_type.c_str(),
                              paddle_shape.size(),
                              gcu_shape.size(),
                              out_name.c_str(),
                              TransformUtil::GetShapeStr(paddle_shape).c_str(),
                              TransformUtil::GetShapeStr(gcu_shape).c_str()));
        auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
        auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);

        VLOG(6) << "TransformUtil::IsDyn(paddle_shape) "
                << TransformUtil::IsDyn(paddle_shape)
                << " TransformUtil::IsDyn(gcu_shape): "
                << TransformUtil::IsDyn(gcu_shape)
                << " TransformUtil::GetShapeStr(paddle_shape) "
                << TransformUtil::GetShapeStr(paddle_shape)
                << " TransformUtil::GetShapeStr(gcu_shape) "
                << TransformUtil::GetShapeStr(gcu_shape) << " tensor->dims() "
                << tensor->dims();

        if (TransformUtil::IsDyn(paddle_shape) &&
            !TransformUtil::IsDyn(gcu_shape)) {
          auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
          auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);
          VLOG(6) << "out var_name:" << out_name.c_str() << " "
                  << "op_type:" << op_type.c_str() << " "
                  << "shape_pd:" << paddle_shape_str << " "
                  << "shape_gcu:" << gcu_shape_str << " "
                  << "[WARN]use gcu shape to flush paddle shape!";
          refresh_program_key = true;
          tensor->Resize(phi::make_ddim(gcu_shape));
          ctx.GetDeviceContext().Alloc(tensor, tensor->dtype());
          continue;
        }
        if (VLOG_IS_ON(10) && gcu_shape_str != paddle_shape_str) {
          builder->Dump();
        }
        PADDLE_ENFORCE_EQ(gcu_shape_str,
                          paddle_shape_str,
                          phi::errors::PreconditionNotMet(
                              "op type:%s"
                              " transfered gcu node should have same shape !"
                              " but paddle shape is %s, gcu is %s,"
                              " out name is %s",
                              op_type.c_str(),
                              paddle_shape_str.c_str(),
                              gcu_shape_str.c_str(),
                              out_name.c_str()));
      }
    }
  } else {
    std::set<std::string> names_in;
    for (const auto& e : op_desc->Inputs()) {
      if (e.second.empty()) {
        continue;
      }
      for (std::string n : e.second) {
        names_in.insert(n);
      }
    }
    for (const auto& e : op_desc->Outputs()) {
      if (kUnusedArchetype.count(e.first) > 0) {
        continue;
      }
      if (e.second.empty()) continue;
      for (const auto& out_name : e.second) {
        auto tensor = tensor_cache[out_name];
        PADDLE_ENFORCE_NE(
            tensor,
            nullptr,
            phi::errors::NotFound("op type:%s out name:%s not found var op!",
                                  op_type.c_str(),
                                  out_name.c_str()));
        GcuOpPtr gte = AddGteOp(tensor, out_name, op);
        PADDLE_ENFORCE_NE(
            gte,
            nullptr,
            phi::errors::Fatal("op type:%s transfer to gcu gte node failed!",
                               op_type.c_str()));
        gcu_op_cache[out_name] = gte;

        // for shape infer check
        auto gcu_shape = gte->GetType().GetShape();
        auto paddle_shape = phi::vectorize(tensor->dims());
        // normalize scalar shape process, [] -> [1]
        if (gcu_shape.empty()) {
          gcu_shape = {1};
        }
        if (paddle_shape.empty()) {
          paddle_shape = {1};
        }
        if (TransformUtil::IsDyn(paddle_shape) &&
            !TransformUtil::IsDyn(gcu_shape)) {
          auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
          auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);
          VLOG(6) << "out var_name:" << out_name.c_str() << " "
                  << "op_type:" << op_type.c_str() << " "
                  << "shape_pd:" << paddle_shape_str << " "
                  << "shape_gcu:" << gcu_shape_str << " "
                  << "[WARN]use gcu shape to flush paddle shape!";
          refresh_program_key = true;
          tensor->Resize(phi::make_ddim(gcu_shape));
          ctx.GetDeviceContext().Alloc(tensor, tensor->dtype());
          continue;
        }
        if (VLOG_IS_ON(10) && gcu_shape.size() != paddle_shape.size()) {
          builder->Dump();
        }
        PADDLE_ENFORCE_EQ(gcu_shape.size(),
                          paddle_shape.size(),
                          phi::errors::PreconditionNotMet(
                              "op type:%s"
                              " transfered gcu node should have same rank!"
                              "but paddle rank is %u, gcu is %u, out name is %s"
                              ", paddle shape:%s, gcu shape:%s",
                              op_type.c_str(),
                              paddle_shape.size(),
                              gcu_shape.size(),
                              out_name.c_str(),
                              TransformUtil::GetShapeStr(paddle_shape).c_str(),
                              TransformUtil::GetShapeStr(gcu_shape).c_str()));
        auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
        auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);
        if (VLOG_IS_ON(10) && gcu_shape_str != paddle_shape_str) {
          builder->Dump();
        }
        PADDLE_ENFORCE_EQ(
            gcu_shape_str,
            paddle_shape_str,
            phi::errors::PreconditionNotMet(
                "op type:%s"
                " transfered gcu node should have same shape!"
                "but origin shape is %s now is %s, out name is %s",
                op_type.c_str(),
                paddle_shape_str.c_str(),
                gcu_shape_str.c_str(),
                out_name.c_str()));
      }
    }
  }

  std::vector<GcuOp> gcu_outputs;
  std::vector<::builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;

  for (auto tensor_name : output_names) {
    PADDLE_ENFORCE_NE(
        (gcu_op_cache.count(tensor_name) == 0 ||
         tensor_cache.count(tensor_name) == 0),
        true,
        phi::errors::NotFound(
            "Output tensor %s is not found, gcu_node:%zu, var_node:%zu",
            tensor_name.c_str(),
            gcu_op_cache.count(tensor_name),
            tensor_cache.count(tensor_name)));

    auto gcu_op = gcu_op_cache[tensor_name];

    tuple_shape.push_back(gcu_op->GetType().GetShape());
    tuple_dtype.push_back(gcu_op->GetType().GetPrimitiveType());
    gcu_outputs.push_back(*gcu_op);
  }

  if (gcu_outputs.size() == 1) {
    builder->SetOutput(gcu_outputs);
  } else {
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto tuple = builder::Tuple(gcu_outputs, outputs_type);
    builder->SetOutput({tuple});
  }

  VLOG(3) << "Convert to gcu graph finished!";
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Hlir IrGraph After convert Paddle IR";
    builder->Dump();
  }

  if (refresh_program_key) {
    std::hash<std::string> hasher;
    std::string signature = BuildSignature(ctx);

    VLOG(3) << "[JIT_KERNEL] " << ctx.Type()
            << " signature(refreshed): " << signature;

    program_key = std::to_string(hasher(signature + ctx.Type()));
    auto manager = backend::SingleOpGcuExecutorManager::GetInstance();
    auto gcu_exec = manager->Find(program_key);
    if (gcu_exec != nullptr) {
      return;
    }
  }

  // compile
  std::vector<backend::GcuNode> input_nodes;
  input_nodes.reserve(inputs.size());
  std::vector<backend::GcuNode> output_nodes;
  output_nodes.reserve(outputs.size());
  for (auto tmp : inputs) {
    input_nodes.emplace_back(backend::GcuNode(*tmp));
  }
  for (auto tmp : outputs) {
    output_nodes.emplace_back(backend::GcuNode(*tmp));
  }
  auto manager = backend::SingleOpGcuExecutorManager::GetInstance();

  auto hlir_module = builder->GetModule();

  VLOG(3) << "Compiler begin to CompileHLIR for program " << program_key;
  topsExecutable_t tops_executable =
      backend::CompileTopsExecutable(hlir_module);
  VLOG(3) << "Compiler CompileHLIR end for program " << program_key;

  auto gcu_exec = std::make_shared<backend::SingleOpGcuExecutor>(
      op_type, tops_executable, input_nodes, output_nodes);
  manager->Add(program_key, gcu_exec);
}

void GcuOpRunner::RunExecutableSync(
    const GcuExecutionContext& ctx,
    const std::string& program_key,
    const std::vector<DenseTensor*>& inputs,
    const std::vector<DenseTensor*>& outputs,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    bool tensor_split) {
  VLOG(3) << "=== start RunExecutableSync ===";

  auto manager = backend::SingleOpGcuExecutorManager::GetInstance();
  auto gcu_exec = manager->Find(program_key);

  PADDLE_ENFORCE_NOT_NULL(
      gcu_exec,
      phi::errors::NotFound("Not found executor for program_key:%s",
                            program_key.c_str()));

  auto device_context =
      static_cast<const phi::CustomContext*>(&ctx.GetDeviceContext());

  static bool print_tensor = (EnvToString(kGcuDebugPrint, "") == "true");

  if (print_tensor) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::vector<float> input_vct;
      TensorToVector(*device_context, *inputs[i], *device_context, &input_vct);
      VLOG(6) << "run exec inputs " << i << " name " << input_names[i] << " "
              << backend::VectorToString(input_vct);
    }
  }

  gcu_exec->RunGcuOp(device_context, inputs, outputs, tensor_split);

  if (print_tensor) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::vector<float> output_vct;
      TensorToVector(*device_context, *inputs[i], *device_context, &output_vct);
      VLOG(6) << "run exec outputs " << i << " name " << output_names[i] << " "
              << backend::VectorToString(output_vct);
    }
  }

  VLOG(3) << "=== end RunExecutableSync ===";
}

void GcuRunner(const TensorNameMap& input_names,
               const TensorValueMap& inputs,
               const TensorNameMap& output_names,
               TensorValueMap& outputs,  // NOLINT
               const GcuAttributeMap& attrs,
               const std::string& op_type,
               const phi::DeviceContext& device_context,
               bool tensor_split) {
  GcuExecutionContext ctx(input_names,
                          inputs,
                          output_names,
                          outputs,
                          attrs,
                          op_type,
                          device_context);

  GcuOpRunner gcu_op;
  gcu_op.Compute(ctx, tensor_split);
}

}  // namespace custom_kernel
