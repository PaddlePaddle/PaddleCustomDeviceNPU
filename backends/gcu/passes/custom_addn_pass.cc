/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>

#include <any>
#include <iostream>

#include "paddle/extension.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace custom_pass {
class AddNReplacePattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "AddNReplacePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &add1 = pat.Op("pd_op.add");
    const auto &add2 = pat.Op("pd_op.add");
    pat.Tensor("add1_out") = add1(pat.Tensor("in1"), pat.Tensor("in2"));
    pat.Tensor("add2_out") = add1(pat.Tensor("add1_out"), pat.Tensor("in3"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &custom_addn = res.Op("custom_op.my_add_n",
                                     {{
                                         {"axis", res.Int64Attr(0)},
                                     }});
    res.Tensor("add2_out") =
        custom_addn(res.Tensor("in1"), res.Tensor("in2"), res.Tensor("in3"));
  }
};

class AddNReplacePass : public pir::PatternRewritePass {
 public:
  AddNReplacePass() : pir::PatternRewritePass("addn_replace_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<AddNReplacePattern>(context));
    return ps;
  }
};

}  // namespace custom_pass

REGISTER_IR_PASS(addn_replace_pass, custom_pass::AddNReplacePass);
