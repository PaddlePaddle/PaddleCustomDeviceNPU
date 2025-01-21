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

#include <glog/logging.h>

#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "custom_engine/custom_engine_op.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_bool(print_ir);
namespace {
using OpListType = std::list<pir::Operation*>;

std::vector<pir::Value> AnalysisOutputs(
    const OpListType& group_ops) {  // NOLINT
  // Get output by ud chain
  std::unordered_set<pir::Operation*> op_set(group_ops.begin(),
                                             group_ops.end());

  std::vector<pir::Value> outputs;
  for (auto* op : group_ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);

      for (auto use_iter = result.use_begin(); use_iter != result.use_end();
           ++use_iter) {
        if (!op_set.count(use_iter->owner())) {
          outputs.push_back(result);
          break;
        }
      }
    }
  }

  // NOTE: If all value are not used outside, we mark last op's results
  // as outputs. But keep in mind that is risky.
  if (outputs.size() == 0) {
    for (size_t i = 0; i < group_ops.back()->num_results(); ++i) {
      outputs.push_back(group_ops.back()->result(i));
    }
  }

  return outputs;
}

std::vector<pir::Value> AnalysisInputs(const OpListType& group_ops) {  // NOLINT
  std::unordered_set<pir::Value> visited_values;
  std::vector<pir::Value> group_inputs;
  std::unordered_set<pir::Operation*> ops_set(group_ops.begin(),
                                              group_ops.end());

  // count all op's input Value
  for (auto* op : group_ops) {
    for (auto& value : op->operands_source()) {
      if (!value || !value.type() || ops_set.count(value.defining_op()))
        continue;
      if (visited_values.count(value)) continue;
      // if the input value owner op is not in OpSet, it's the group's input
      visited_values.insert(value);
      group_inputs.push_back(value);
    }
  }
  return group_inputs;
}

class ReplaceWithCustomEngineOpPattern
    : public pir::OpRewritePattern<pir::GroupOp> {
 public:
  using pir::OpRewritePattern<pir::GroupOp>::OpRewritePattern;

  bool MatchAndRewrite(
      pir::GroupOp op,
      pir::PatternRewriter& rewriter) const override {  // NOLINT
    pir::Block* block = op.block();

    if (FLAGS_print_ir) {
      std::cout
          << "ReplaceWithCustomEngineOpPattern MatchAndRewrite before IR = "
          << *(op->GetParent()) << std::endl;
    }

    OpListType group_ops = block->ops();

    const std::vector<pir::Value> inputs = AnalysisInputs(group_ops);
    const std::vector<pir::Value> outputs = op->results();

    // attrs
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::string input_name = "graph_input_" + std::to_string(i) + "_op_" +
                               std::to_string(inputs[i].defining_op()->id());
      input_names.emplace_back(input_name);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::string output_name = "graph_output_" + std::to_string(i) + "_op_" +
                                std::to_string(outputs[i].defining_op()->id());
      output_names.emplace_back(output_name);
    }

    std::vector<pir::Type> output_types;
    for (auto& value : outputs) {
      output_types.emplace_back(value.type());
    }

    auto buildin_combine_op = rewriter.Build<pir::CombineOp>(inputs);

    custom_engine::CustomEngineOp custom_engine_op =
        rewriter.Build<custom_engine::CustomEngineOp>(
            buildin_combine_op.out(), input_names, output_names, output_types);

    auto out_split_op = rewriter.Build<pir::SplitOp>(custom_engine_op.out());
    std::vector<pir::Value> new_outputs = out_split_op.outputs();

    if (FLAGS_print_ir) {
      std::cout << "custom_engine_op name: " << custom_engine_op.name()
                << std::endl;
      std::cout << "ReplaceWithCustomEngineOpPattern MatchAndRewrite mid IR = "
                << *(op->GetParent()) << std::endl;
    }

    for (auto inner_op : group_ops) {
      inner_op->MoveTo(custom_engine_op.block(),
                       custom_engine_op.block()->end());
    }
    rewriter.ReplaceOp(op, new_outputs);

    if (FLAGS_print_ir) {
      std::cout
          << "ReplaceWithCustomEngineOpPattern MatchAndRewrite after IR = "
          << *(op->GetParent()) << std::endl;
    }

    return true;
  }
};

class GcuReplaceWithCustomEngineOpPass : public pir::PatternRewritePass {
 public:
  GcuReplaceWithCustomEngineOpPass()
      : pir::PatternRewritePass("gcu_replace_with_engine_op_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(std::make_unique<ReplaceWithCustomEngineOpPattern>(context));
    return ps;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateGcuReplaceWithCustomEngineOpPass() {
  return std::make_unique<GcuReplaceWithCustomEngineOpPass>();
}

}  // namespace pir

REGISTER_IR_PASS(gcu_replace_with_engine_op_pass,
                 GcuReplaceWithCustomEngineOpPass);
