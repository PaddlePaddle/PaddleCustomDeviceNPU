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

#include <bitset>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// inline auto kCanRunGcuAttr = paddle::dialect::kCanRunGcuAttr;
inline const char kCanRunGcuAttr[] = "__l_gcu__";

#define DEFINE_GENERAL_PATTERN(OpName, OpType)                            \
  class OpName##OpPattern : public pir::OpRewritePattern<OpType> {        \
   public:                                                                \
    using pir::OpRewritePattern<OpType>::OpRewritePattern;                \
    bool MatchAndRewrite(OpType op,                                       \
                         pir::PatternRewriter &rewriter) const override { \
      if (op->HasAttribute(kCanRunGcuAttr) &&                             \
          op->attribute<pir::BoolAttribute>(kCanRunGcuAttr).data()) {     \
        return false;                                                     \
      }                                                                   \
      op->set_attribute(kCanRunGcuAttr, rewriter.bool_attr(true));        \
      return true;                                                        \
    }                                                                     \
  };

DEFINE_GENERAL_PATTERN(Matmul, paddle::dialect::MatmulOp)
DEFINE_GENERAL_PATTERN(Add, paddle::dialect::AddOp)
DEFINE_GENERAL_PATTERN(Abs, paddle::dialect::AbsOp)
DEFINE_GENERAL_PATTERN(Full, paddle::dialect::FullOp)
DEFINE_GENERAL_PATTERN(ScaleOp, paddle::dialect::ScaleOp)

class GcuOpMarkerPass : public pir::PatternRewritePass {
 public:
  GcuOpMarkerPass() : pir::PatternRewritePass("gcu_op_marker_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

#define ADD_PATTERN(OpName) \
  ps.Add(std::make_unique<OpName##OpPattern>(context));
    ADD_PATTERN(Matmul)
    ADD_PATTERN(Add)
    ADD_PATTERN(Abs)
    ADD_PATTERN(Full)
    ADD_PATTERN(ScaleOp)
#undef ADD_PATTERN

    return ps;
  }
};
}  // namespace

namespace pir {
std::unique_ptr<Pass> CreateGcuOpMarkerPass() {
  return std::make_unique<GcuOpMarkerPass>();
}
}  // namespace pir

REGISTER_IR_PASS(gcu_op_marker_pass, GcuOpMarkerPass);
