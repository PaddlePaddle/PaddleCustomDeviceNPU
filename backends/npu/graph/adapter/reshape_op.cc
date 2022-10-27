// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "graph/graph_executor.h"

namespace custom_graph {

class ReshapeAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");
    auto shape_tensor_vector = ctx.MultiInput("ShapeTensor");

    if (shape_tensor_vector.size() > 0) {
      graph::utils::log() << "[ERROR] reshape unsupported ShapeTensor."
                          << std::endl;
      exit(-1);
    } else {
      if (ctx.HasInput("Shape")) {
        OpCommand("Reshape").Input(x).Input(ctx.Input("Shape")).Output(out);
      } else {
        auto target_shape_vector = ctx.Attr<std::vector<int>>("shape");
        int num_negative = std::count(
            target_shape_vector.begin(), target_shape_vector.end(), -1);
        auto it_zero = std::find(
            target_shape_vector.begin(), target_shape_vector.end(), 0);
        if (it_zero != target_shape_vector.end()) {
          int x_rank = x.Shape().size();
          for (size_t i = 0; i < target_shape_vector.size(); i++) {
            if (target_shape_vector[i] == 0) {
              // PADDLE_ENFORCE_LT(
              //     i,
              //     x_rank,
              //     platform::errors::InvalidArgument(
              //         "The index of 0 in shape attribute or shape tensor",
              //         "should be less than input dim size, ",
              //         "but the index is %d and input dim size is %d",
              //         i,
              //         x_rank));
              target_shape_vector[i] = x.Shape().at(i);
            }
          }
        }
        auto it = std::find(
            target_shape_vector.begin(), target_shape_vector.end(), -1);
        if (it != target_shape_vector.end()) {
          auto ddim_out_vec = x.Shape();
          int ddim_out_product = std::accumulate(ddim_out_vec.begin(),
                                                 ddim_out_vec.end(),
                                                 1,
                                                 std::multiplies<int>());
          int reshape_out_product = std::accumulate(target_shape_vector.begin(),
                                                    target_shape_vector.end(),
                                                    -1,
                                                    std::multiplies<int>());
          int index = std::distance(target_shape_vector.begin(), it);
          target_shape_vector[index] = ddim_out_product / reshape_out_product;
        }

        OpCommand::Reshape(x, out, target_shape_vector);
      }
    }
  }
};

class ReshapeGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& d_x = ctx.Output(paddle::framework::GradVarName("X"));
    auto& d_out = ctx.Input(paddle::framework::GradVarName("Out"));
    auto in_dims = d_x.Shape();

    OpCommand::Reshape(d_out, d_x, in_dims);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(reshape2, custom_graph::ReshapeAdapter);
REG_OP_ADAPTER(reshape2_grad, custom_graph::ReshapeGradAdapter);
