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

#include <vector>

#include "custom_engine/ir_translator/translator_registry.h"

namespace custom_engine {

static GcuOpPtr TranslateMatmul(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  PADDLE_ENFORCE_EQ(
      gcu_op_inputs.size(),
      2,
      common::errors::PreconditionNotMet(
          "Intput op num check failed, get num:%zu", gcu_op_inputs.size()));
  auto X = *(gcu_op_inputs[0][0]);
  auto Y = *(gcu_op_inputs[1][0]);

  auto x_shape = X.GetType().GetShape();
  auto y_shape = Y.GetType().GetShape();

  bool trans_x =
      op->attribute("transpose_x").dyn_cast<pir::BoolAttribute>().data();
  bool trans_y =
      op->attribute("transpose_y").dyn_cast<pir::BoolAttribute>().data();

  int64_t x_rank = x_shape.size();
  int64_t y_rank = y_shape.size();
  int64_t max_rank = std::max(x_rank, y_rank);
  int64_t rank_diff = std::abs(x_rank - y_rank);
  auto ptype = X.GetType().GetPrimitiveType();
  int64_t batch_dim;

  if (x_rank > y_rank) {
    if (trans_x || y_rank == 1) {
      std::vector<int64_t> broadcast_dims;
      std::vector<int64_t> bc_shape;
      if (y_rank == 1) {
        for (int64_t i = 0; i < rank_diff - 1; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        bc_shape.emplace_back(y_shape[0]);
        bc_shape.emplace_back(1);
        broadcast_dims.emplace_back(rank_diff - 1);
      } else {
        for (int64_t i = 0; i < rank_diff; i++) {
          bc_shape.emplace_back(x_shape[i]);
        }
        for (int64_t i = 0; i < y_rank; i++) {
          bc_shape.emplace_back(y_shape[i]);
        }
        int iter = 0;
        for (int64_t i = 0; i < x_rank; ++i) {
          if (i < rank_diff) {
            ++iter;
          } else {
            broadcast_dims.emplace_back(i);
          }
        }
      }
      builder::Type type(bc_shape, ptype);
      Y = builder::BroadcastInDim(Y, broadcast_dims, type);
    }
    if (y_rank == 1) {
      batch_dim = rank_diff - 1;
    } else {
      batch_dim = rank_diff;
    }

  } else if (x_rank < y_rank) {
    std::vector<int64_t> broadcast_dims;
    std::vector<int64_t> bc_shape;
    if (x_rank == 1) {
      for (int64_t i = 0; i < rank_diff - 1; i++) {
        bc_shape.emplace_back(y_shape[i]);
      }
      bc_shape.emplace_back(1);
      bc_shape.emplace_back(x_shape[0]);
      broadcast_dims.emplace_back(rank_diff);
    } else {
      for (int64_t i = 0; i < rank_diff; i++) {
        bc_shape.emplace_back(y_shape[i]);
      }
      for (int64_t i = 0; i < x_rank; i++) {
        bc_shape.emplace_back(x_shape[i]);
      }
      int iter = 0;
      for (int64_t i = 0; i < y_rank; ++i) {
        if (i < rank_diff) {
          ++iter;
        } else {
          broadcast_dims.emplace_back(i);
        }
      }
    }
    builder::Type type(bc_shape, ptype);
    X = builder::BroadcastInDim(X, broadcast_dims, type);
    if (x_rank == 1) {
      batch_dim = rank_diff - 1;
    } else {
      batch_dim = rank_diff;
    }

  } else {
    batch_dim = max_rank - 2;
    if (x_rank == y_rank && x_rank > 3) {
      auto x_brd_shape = x_shape;
      auto y_brd_shape = y_shape;
      std::vector<int64_t> x_brd_dims, y_brd_dims;
      for (int64_t i = 0; i < x_rank - 2; ++i) {
        x_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
        y_brd_shape[i] = x_shape[i] > y_shape[i] ? x_shape[i] : y_shape[i];
      }
      x_brd_dims.resize(x_rank);
      y_brd_dims.resize(y_rank);
      std::iota(x_brd_dims.begin(), x_brd_dims.end(), 0);
      std::iota(y_brd_dims.begin(), y_brd_dims.end(), 0);
      if (x_brd_shape != x_shape) {
        X = builder::BroadcastInDim(
            X, x_brd_dims, builder::Type(x_brd_shape, ptype));
      }
      if (y_brd_shape != y_shape) {
        Y = builder::BroadcastInDim(
            Y, y_brd_dims, builder::Type(y_brd_shape, ptype));
      }
    }
  }

  builder::DotDimensionNumbers dims_attr;
  std::vector<int64_t> lhs_batching_dimensions = {};
  std::vector<int64_t> rhs_batching_dimensions = {};
  std::vector<int64_t> lhs_contracting_dimensions = {};
  std::vector<int64_t> rhs_contracting_dimensions = {};
  if (x_rank == 1 && y_rank == 1) {
    lhs_contracting_dimensions.emplace_back(0);
    rhs_contracting_dimensions.emplace_back(0);
  } else if (x_rank <= y_rank || trans_x || y_rank == 1) {
    for (int64_t i = 0; i < max_rank - 1; ++i) {
      if (i < batch_dim) {
        lhs_batching_dimensions.emplace_back(i);
        rhs_batching_dimensions.emplace_back(i);
      } else {
        if (trans_x && x_rank != 1) {
          lhs_contracting_dimensions.emplace_back(i);
        } else {
          lhs_contracting_dimensions.emplace_back(i + 1);
        }
        if (trans_y && y_rank != 1) {
          rhs_contracting_dimensions.emplace_back(i + 1);
        } else {
          rhs_contracting_dimensions.emplace_back(i);
        }
      }
    }
  } else {
    lhs_contracting_dimensions.emplace_back(x_rank - 1);
    if (y_rank != 1) {
      if (trans_y) {
        rhs_contracting_dimensions.emplace_back(y_rank - 1);
      } else {
        rhs_contracting_dimensions.emplace_back(y_rank - 2);
      }
    } else {
      rhs_contracting_dimensions.emplace_back(0);
    }
  }

  dims_attr.set_lhs_batching_dimensions(lhs_batching_dimensions);
  dims_attr.set_rhs_batching_dimensions(rhs_batching_dimensions);
  dims_attr.set_lhs_contracting_dimensions(lhs_contracting_dimensions);
  dims_attr.set_rhs_contracting_dimensions(rhs_contracting_dimensions);
  std::vector<const char *> precision_config = {};
  auto dot = builder::DotGeneral(X, Y, dims_attr, precision_config);
  dot.SetAttribute("op_type", builder::Attribute("DotInference"));
  if (x_rank == 1 && y_rank == 1) {
    auto type = dot.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    new_shape.push_back(1);
    builder::Type output_type(new_shape, type);
    dot = builder::Reshape(dot, output_type);
  } else if (y_rank == 1) {
    auto shape = dot.GetType().GetShape();
    auto type = dot.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < shape.size() - 1; i++) {
      new_shape.push_back(shape[i]);
    }
    builder::Type output_type(new_shape, type);
    dot = builder::Reshape(dot, output_type);
  } else if (x_rank == 1) {
    auto shape = dot.GetType().GetShape();
    auto type = dot.GetType().GetPrimitiveType();
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < shape.size(); i++) {
      if (i != shape.size() - 2) {
        new_shape.push_back(shape[i]);
      }
    }
    builder::Type output_type(new_shape, type);
    dot = builder::Reshape(dot, output_type);
  }

  auto result = std::make_shared<GcuOp>(dot);

  return result;
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_matmul, custom_engine::TranslateMatmul)
