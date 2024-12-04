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

#pragma once
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/utils/utils.h"
#include "common/gcu_funcs.h"

namespace custom_kernel {

constexpr char kEmptyVarName[] = "@EMPTY@";

using DenseTensor = phi::DenseTensor;
using DenseTensor = phi::DenseTensor;
using TensorNameMap = std::map<std::string, std::vector<std::string>>;
using TensorValueMap = std::map<std::string, std::vector<DenseTensor*>>;
using TensorNameValuePair = std::pair<std::string, DenseTensor*>;
using GcuAttributeMap = backend::GcuAttributeMap;
using GcuAttribute = backend::GcuAttribute;
using GcuOp = ::builder::Op;
using GcuOpPtr = std::shared_ptr<GcuOp>;

class GcuExecutionContext {
 public:
  GcuExecutionContext(const TensorNameMap& input_names,
                      const TensorValueMap& inputs,
                      const TensorNameMap& output_names,
                      TensorValueMap& outputs,  // NOLINT
                      const GcuAttributeMap& attrs,
                      const std::string& op_type,
                      const phi::DeviceContext& device_context)
      : input_names_(input_names),
        inputs_(inputs),
        output_names_(output_names),
        outputs_(outputs),
        attrs_(attrs),
        op_type_(op_type),
        device_context_(device_context) {}
  ~GcuExecutionContext() {}

  std::string InputName(const std::string& name) const {
    auto it = input_names_.find(name);
    PADDLE_ENFORCE_NE(
        it,
        input_names_.end(),
        phi::errors::NotFound(
            "Operator %s does not have the input %s.", op_type_, name));
    auto& ins = it->second;
    PADDLE_ENFORCE_LE(
        ins.size(),
        1UL,
        phi::errors::InvalidArgument(
            "Operator %s's input %s should contain only one variable.",
            op_type_,
            name));
    return ins.empty() ? kEmptyVarName : ins[0];
  }
  std::vector<std::string> InputNames(const std::string& name) const {
    auto it = input_names_.find(name);
    PADDLE_ENFORCE_NE(
        it,
        input_names_.end(),
        phi::errors::NotFound(
            "Operator %s does not have the input %s.", op_type_, name));
    return it->second;
  }

  std::string OutputName(const std::string& name) const {
    auto it = output_names_.find(name);
    PADDLE_ENFORCE_NE(
        it,
        output_names_.end(),
        phi::errors::NotFound(
            "Operator %s does not have the output %s.", op_type_, name));
    auto& outs = it->second;
    PADDLE_ENFORCE_LE(
        outs.size(),
        1UL,
        phi::errors::InvalidArgument(
            "Operator %s's output %s should contain only one variable.",
            op_type_,
            name));
    return outs.empty() ? kEmptyVarName : outs[0];
  }

  std::vector<std::string> OutputNames(const std::string& name) const {
    auto it = output_names_.find(name);
    PADDLE_ENFORCE_NE(
        it,
        output_names_.end(),
        phi::errors::NotFound(
            "Operator %s does not have the input %s.", op_type_, name));
    return it->second;
  }

  bool HasInput(const std::string& name) const {
    auto* var = InputTensor(name);
    return var != nullptr;
  }

  bool HasInputs(const std::string& name) const {
    const auto& ins = inputs_;
    auto it = ins.find(name);
    if (it == ins.end() || it->second.empty()) {
      return false;
    }
    for (const auto* input : it->second) {
      if (input == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutput(const std::string& name) const {
    auto* var = OutputTensor(name);
    return var != nullptr;
  }

  const DenseTensor* InputTensor(const std::string& name) const {
    auto it = inputs_.find(name);
    if (it == inputs_.end()) return nullptr;

    PADDLE_ENFORCE_LE(
        it->second.size(),
        1UL,
        phi::errors::InvalidArgument(
            "Operator %s's input %s should contain only one tensor.",
            op_type_,
            name));
    return it->second.empty() ? nullptr : it->second[0];
  }

  DenseTensor* OutputTensor(const std::string& name) const {
    auto it = outputs_.find(name);
    if (it == outputs_.end()) return nullptr;

    PADDLE_ENFORCE_LE(
        it->second.size(),
        1UL,
        phi::errors::InvalidArgument(
            "Operator %s's output %s should contain only one tensor.",
            op_type_,
            name));
    return it->second.empty() ? nullptr : it->second[0];
  }

  const TensorNameMap& AllInputNames() const { return input_names_; }

  const TensorNameMap& AllOutputNames() const { return output_names_; }

  const TensorValueMap& AllInputs() const { return inputs_; }

  const TensorValueMap& AllOutputs() const { return outputs_; }

  phi::Place GetPlace() const { return device_context_.GetPlace(); }

  const phi::DeviceContext& GetDeviceContext() const { return device_context_; }

  const std::string& Type() const { return op_type_; }

  bool HasAttr(const std::string& name) const { return attrs_.count(name); }
  const GcuAttributeMap& Attrs() const { return attrs_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return PADDLE_GET_CONST(T, GetAttr(name));
  }

  const GcuAttribute& GetAttr(const std::string& name) const {
    auto iter = attrs_.find(name);
    PADDLE_ENFORCE_NE(
        iter,
        attrs_.end(),
        phi::errors::NotFound("(%s) is not found in GcuAttributeMap and "
                              "RuntimeAttributeMap of (%s) operator.",
                              name,
                              op_type_));
    return iter->second;
  }

  const std::vector<DenseTensor*> MultiInputTensor(
      const std::string& name) const {
    auto it = inputs_.find(name);
    if (it == inputs_.end()) {
      return {};
    }
    return {it->second.begin(), it->second.end()};
  }

  std::vector<DenseTensor*> MultiOutputTensor(const std::string& name) const {
    auto it = outputs_.find(name);
    if (it == outputs_.end()) {
      return {};
    }
    return it->second;
  }

 private:
  const TensorNameMap& input_names_;
  const TensorValueMap& inputs_;
  const TensorNameMap& output_names_;
  TensorValueMap& outputs_;
  const GcuAttributeMap& attrs_;
  const std::string& op_type_;
  const phi::DeviceContext& device_context_;
};

class GcuOpRunner {
 public:
  virtual void Compute(const GcuExecutionContext& ctx, bool tensor_split) {
    std::vector<TensorNameValuePair> input_vars;
    std::vector<TensorNameValuePair> output_vars;

    GetInputsAndOutputs(ctx, input_vars, output_vars);

    CompileAndRun(ctx, input_vars, output_vars, tensor_split);
  }

  virtual ~GcuOpRunner() = default;

 private:
  void GetInputsAndOutputs(
      const GcuExecutionContext& ctx,
      std::vector<TensorNameValuePair>& input_vars,    // NOLINT
      std::vector<TensorNameValuePair>& output_vars);  // NOLINT

  std::string AttrString(const GcuExecutionContext& ctx);
  std::string BuildSignature(const GcuExecutionContext& ctx);
  void CompileAndRun(const GcuExecutionContext& ctx,
                     const std::vector<TensorNameValuePair>& input_vars,
                     const std::vector<TensorNameValuePair>& output_vars,
                     bool tensor_split);
  GcuOpPtr AddGteOp(const DenseTensor* tensor,
                    const std::string& tensor_name,
                    const GcuOpPtr& input);
  void CompileExecutable(const GcuExecutionContext& ctx,
                         const std::string& program_key_in,
                         const std::vector<DenseTensor*>& inputs,
                         const std::vector<DenseTensor*>& outputs,
                         const std::vector<std::string>& input_names,
                         const std::vector<std::string>& output_names);
  void RunExecutableSync(const GcuExecutionContext& ctx,
                         const std::string& program_key,
                         const std::vector<DenseTensor*>& inputs,
                         const std::vector<DenseTensor*>& outputs,
                         const std::vector<std::string>& input_names,
                         const std::vector<std::string>& output_names,
                         bool tensor_split);

  std::map<size_t, topsExecutable_t> executable_cache_;
};

void GcuRunner(const TensorNameMap& input_names,
               const TensorValueMap& inputs,
               const TensorNameMap& output_names,
               TensorValueMap& outputs,  // NOLINT
               const GcuAttributeMap& attrs,
               const std::string& op_type,
               const phi::DeviceContext& device_context,
               bool tensor_split = true);

}  // namespace custom_kernel
