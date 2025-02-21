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

#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "runtime/flags.h"

FLAGS_DECLARE_int32(custom_engine_min_group_size);
COMMON_DECLARE_bool(print_ir);

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;
inline const char kCanRunGcuAttr[] = "__l_gcu__";

bool IsSupportedByGCU(const pir::Operation& op) {
  if (op.HasAttribute(kCanRunGcuAttr) &&
      op.attribute<pir::BoolAttribute>(kCanRunGcuAttr).data()) {
    return true;
  }
  return false;
}

class GcuSubGraphExtractPass : public pir::Pass {
 public:
  GcuSubGraphExtractPass() : pir::Pass("gcu_sub_graph_extract_pass", 2) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    PADDLE_ENFORCE_NOT_NULL(
        module_op,
        common::errors::InvalidArgument(
            "sub_graph_extract_pass should run on module op."));
    auto& block = module_op.block();

    if (FLAGS_print_ir) {
      std::cout << "GcuSubGraphExtractPass before IR = " << block << std::endl;
    }

    std::vector<GroupOpsVec> groups =
        pir::DetectSubGraphs(&block, IsSupportedByGCU);
    VLOG(3) << "GcuSubGraphExtractPass, detected " << groups.size()
            << " groups.";
    for (auto& group_ops : groups) {
      if (group_ops.size() <
          static_cast<size_t>(FLAGS_custom_engine_min_group_size)) {
        VLOG(3) << "current group_ops.size(): " << group_ops.size()
                << ", less than min_group_size:"
                << static_cast<size_t>(FLAGS_custom_engine_min_group_size)
                << ", will fallback to paddle original graph";
        continue;
      }
      VLOG(3) << "current group_ops.size(): " << group_ops.size()
              << ", greater or equal than min_group_size:"
              << static_cast<size_t>(FLAGS_custom_engine_min_group_size)
              << ", will lower to GCU graph";
      pir::ReplaceWithGroupOp(&block, group_ops);
    }
    if (FLAGS_print_ir) {
      std::cout << "GcuSubGraphExtractPass after IR = " << block << std::endl;
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateGcuSubGraphExtractPass() {
  return std::make_unique<GcuSubGraphExtractPass>();
}

}  // namespace pir

REGISTER_IR_PASS(gcu_sub_graph_extract_pass, GcuSubGraphExtractPass);
