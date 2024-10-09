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

#include "kernels/einsum_kernel.h"

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/string_helper.h"
namespace custom_kernel {

inline static void ValidationCheck(const std::string& equation) {
  const std::string arrow = "->";
  auto n_part = split_string(equation, arrow).size();
  PADDLE_ENFORCE_EQ(n_part,
                    2,
                    phi::errors::InvalidArgument(
                        "Required at least one `->` in equation of EinsumOp."));
  size_t pos;
  auto trimed_equ = equation;
  if ((pos = trimed_equ.find("->", 0)) != std::string::npos) {
    trimed_equ.replace(pos, 2, "");
  }
  auto is_valid_char = [](char c) {
    if (c >= 'a' && c <= 'z') return true;
    if (c == ',') return true;
    return false;
  };
  for (auto c : trimed_equ) {
    if (!is_valid_char(c))
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Found invalid char in equation. Einsum only accept `a`-`z` and `...`"
          "but get:`%c`",
          c));
  }
}

inline std::string label_to_string(const std::vector<char>& all_labels,
                                   const LabelMap& label2type) {
  std::string str;
  for (int a : all_labels) {
    std::stringstream ss;
    ss << label2type[a];
    str += ss.str();
  }
  return str;
}

// inline static void ReplaceEllipsis(std::string& s) {  // NOLINT
//   size_t pos;
//   if ((pos = s.find("...", 0)) != std::string::npos) {
//     s.replace(pos, 3, ".");
//   }
//   // remove all the space in the expression
//   while ((pos = s.find(" ", 0)) != std::string::npos) {
//     s.replace(pos, 1, "");
//   }
// }

template <typename CharIterable1, typename CharIterable2>
inline std::vector<char> union_labels(const CharIterable1& a,
                                      const CharIterable2& b) {
  LabelMap counter(0);
  std::vector<char> res;
  auto f = [&](char c) {
    if (counter[static_cast<int>(c)] == 0) {
      res.push_back(c);
    }
    counter[static_cast<int>(c)] += 1;
  };
  std::for_each(a.begin(), a.end(), f);
  std::for_each(b.begin(), b.end(), f);
  return res;
}

template <typename CharIterable>
inline std::vector<char> unique_labels(const CharIterable& a) {
  return union_labels(a, CharIterable());
}

// Apply transforms to all_labels and get another all_labels
inline std::vector<char> TransformLabelsOrder(
    const std::vector<char>& all_labels,
    const LabelMap& type,
    std::vector<LabelType> new_order) {
  std::vector<char> ret;
  for (auto cnt_type : new_order) {
    std::vector<char> tmp;
    for (int c : all_labels) {
      if (type[c] == cnt_type) tmp.push_back(c);
    }
    ret.insert(ret.end(), tmp.begin(), tmp.end());
  }
  return ret;
}

inline static void GlobalInfo(const std::vector<std::string>& op_labels,
                              const std::string& right,
                              LabelMap* label2type,
                              std::vector<char>* sorted_labels) {
  std::vector<char> all;
  LabelMap counter(0);
  for (auto& ch : right) {  // char
    int c = ch;
    (*label2type)[c] = LabelType::BO;
  }

  for (auto& op : op_labels) {
    for (auto& ch : unique_labels(op)) {  // char
      int c = ch;
      if (!counter.exist(c)) {
        all.push_back(ch);
      }
      counter[c] += 1;
      if ((*label2type)[c] != LabelType::BO && counter[c] == 2)
        (*label2type)[c] = LabelType::Contraction;
      else if (counter[c] == 2)
        (*label2type)[c] = LabelType::Batch;
    }
  }

  // BO is represent Free, so we need find the AO.
  for (int c : op_labels[0]) {
    if ((*label2type)[c] == LabelType::BO) (*label2type)[c] = LabelType::AO;
  }

  if (sorted_labels->size()) {
    std::set<char> exist(all.begin(), all.end());
    all.clear();
    std::for_each(
        sorted_labels->begin(), sorted_labels->end(), [&exist, &all](char c) {
          if (exist.count(c)) all.push_back(c);
        });
  }

  *sorted_labels = TransformLabelsOrder(all,
                                        *label2type,
                                        {LabelType::Batch,
                                         LabelType::AO,
                                         LabelType::BO,
                                         LabelType::Contraction,
                                         LabelType::Reduction});

  VLOG(5) << "GlobalInfo: sorted_labels after: "
          << join_strings(*sorted_labels, ",");
}

inline static void InferLabelShape(
    const std::vector<std::string>& op_labels,
    const std::vector<phi::DDim>& inputs,
    LabelMap* labelshape,
    std::vector<std::vector<int>>* broadcast_shapes) {
  VLOG(5) << "Start InferLabelShape";
  for (size_t i = 0; i < op_labels.size(); ++i) {
    auto& op_str = op_labels[i];
    auto& op_dim = inputs[i];
    int dim_ptr = 0;
    for (auto& c : op_str) {
      if (!labelshape->exist(c) || abs((*labelshape)[c]) == 1) {
        (*labelshape)[c] = static_cast<int>(op_dim[dim_ptr]);
      } else if (abs(op_dim[dim_ptr]) != 1) {
        PADDLE_ENFORCE_EQ(
            (*labelshape)[c],
            op_dim[dim_ptr],
            phi::errors::InvalidArgument(
                "Same label have different shapes for label: `%c`", c));
      }
      dim_ptr++;
    }
  }
  for (size_t i = 0; i < op_labels.size(); ++i) {
    for (auto& c : op_labels[i]) {
      (*broadcast_shapes)[i].push_back((*labelshape)[c]);
    }
  }
  for (size_t i = 0; i < op_labels.size(); ++i) {
    VLOG(5) << "InferLabelShape: After broadcast shape is:"
            << join_strings((*broadcast_shapes)[i], ",");
  }
}

template <class CharIterable>
inline static void InferLabelPerm(const CharIterable& op,
                                  LabelMap* label2perm) {
  int cur = 0;
  for (int c : op) {
    if (!label2perm->exist(
            c))  // can appear repeatedly. we just record the first position.
      (*label2perm)[c] = cur;
    cur += 1;
  }
}

inline static void InferOutputDims(const std::string& right,
                                   const LabelMap& labelshape,
                                   std::vector<int>* output_dims) {
  for (int c : right) {
    output_dims->push_back(labelshape[c]);
  }
}

inline static void ParseEinsumEquation(
    const std::string& equation,
    const std::vector<phi::DDim>& inputs,
    LabelMap* labelshape,
    LabelMap* labeltype,
    std::vector<char>* all_labels,
    std::vector<LabelMap>* label2perms,
    std::vector<std::vector<int>>* broadcast_shapes,
    std::vector<int>* output_dims,
    std::string* right,
    std::vector<std::string>* input_strs) {
  VLOG(5) << "Start ParseEinsumEquation " << equation;
  auto results = split_string(equation, "->");
  auto left = results[0];
  *right = results[1];
  auto op_labels = split_string(left, ",");
  // split_string("i,") -> ["i", ""], we push back a "".
  // split_string("->") -> [], we push back a "".
  if (op_labels.empty()) op_labels.emplace_back("");
  GlobalInfo(op_labels, *right, labeltype, all_labels);
  InferLabelShape(op_labels, inputs, labelshape, broadcast_shapes);
  VLOG(5) << "Einsum Infershape: right:" << *right;
  VLOG(5) << "Einsum Infershape: left :" << join_strings(op_labels, '\n');
  InferOutputDims(*right, *labelshape, output_dims);
  for (size_t i = 0; i < inputs.size(); ++i) {
    InferLabelPerm(op_labels[i], &((*label2perms)[i]));
    (*input_strs).push_back(std::move(op_labels[i]));
  }
}

// inline static std::vector<int> perm_moveto(int n, int from, int to) {
//   // a permution means moving `from` to `to`.
//   /*
//   f => t   permtation
//   --------------------
//            0 1 2 3 4 5
//   5 => 2 : 0 2 5 2 3 4
//   2 => 5 : 0 1 3 4 5 2
//   we can conclude the following rules.
//   */
//   if (from < 0) from = n + from;
//   if (to < 0) to = n + to;
//   std::vector<int> res(n);
//   for (int i = 0; i < n; ++i) {
//     res[i] = i;
//   }
//   res[to] = from;
//   auto offset = from > to ? -1 : 1;
//   auto start = from > to ? to + 1 : from;
//   auto end = from > to ? from : to - 1;
//   for (int i = start; i <= end; ++i) {
//     res[i] += offset;
//   }
//   return res;
// }

template <typename T>
std::vector<T> GetLabelIndexByType(const std::vector<char>& all_labels,
                                   const LabelMap& type,
                                   const LabelMap& perm,
                                   LabelType filter) {
  std::vector<T> res;
  for (T c : all_labels) {
    if ((filter == LabelType::ALL_TYPE || type[c] == filter) && perm[c] != -1) {
      res.push_back(perm[c]);
    }
  }
  return res;
}

template <typename T>
std::vector<T> GetShapeByType(const std::vector<char>& all_labels,
                              const LabelMap& type,
                              const LabelMap& perm,
                              const LabelMap& label2shape,
                              std::set<LabelType> filter) {
  std::vector<T> res;
  for (T c : all_labels) {
    if ((filter.count(LabelType::ALL_TYPE) ||
         filter.count(LabelType(type[c]))) &&
        perm[c] != -1) {
      res.push_back(label2shape[c]);
    }
  }
  return res;
}

inline static std::vector<int> perm_moveto(int n, int from, int to) {
  // a permutation means moving `from` to `to`.
  /*
  f => t   permutation
  --------------------
           0 1 2 3 4 5
  5 => 2 : 0 2 5 2 3 4
  2 => 5 : 0 1 3 4 5 2
  we can conclude the following rules.
  */
  if (from < 0) from = n + from;
  if (to < 0) to = n + to;
  std::vector<int> res(n);
  for (int i = 0; i < n; ++i) {
    res[i] = i;
  }
  res[to] = from;
  auto offset = from > to ? -1 : 1;
  auto start = from > to ? to + 1 : from;
  auto end = from > to ? from : to - 1;
  for (int i = start; i <= end; ++i) {
    res[i] += offset;
  }
  return res;
}

template <typename T, typename Context>
phi::DenseTensor Undiagonal(const Context& dev_ctx,
                            const phi::DenseTensor& tensor,
                            size_t insert_pos,
                            size_t axis) {
  // tensor with shape (3, 4, 5, 2, 1), insert_pos = 5, axis = 2.
  // output is (3, 4, 5, 2, 1, 5)
  VLOG(5) << "Start undiagonal with args: insert_pos = " << insert_pos
          << ", axis = " << axis;
  std::vector<int> shape(tensor.dims().size() + 1);
  int point = 0;  // point to the tensor.dims()
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == insert_pos)
      shape[i] = tensor.dims()[axis];
    else
      shape[i] = tensor.dims()[point++];
  }
  auto zeros = custom_kernel::Full<T, Context>(dev_ctx, shape, 0);
  auto diags = custom_kernel::Transpose<T, Context>(
      dev_ctx, tensor, perm_moveto(tensor.dims().size(), axis, -1));
  return custom_kernel::FillDiagonalTensor<T, Context>(
      dev_ctx, zeros, diags, 0, insert_pos, axis + (insert_pos <= axis));
}

template <typename T, typename Context>
phi::DenseTensor PerformUndiagonal(const Context& dev_ctx,
                                   const phi::DenseTensor& tensor,
                                   const std::string& equ) {
  //  if the equ is 'iijjkij', then the tensor must be 'ijk', so we have enough
  //  information to do un-diagonal with equ.
  auto res = tensor;
  LabelMap label2perm(-1);
  InferLabelPerm(equ, &label2perm);
  // Un-Diagonal
  int tot = equ.size();
  int cur = tot - 1;
  for (auto it = equ.rbegin(); it != equ.rend(); ++it) {
    char c = *it;
    if (cur != label2perm[c]) {
      // do diagonal, followed by movedim().
      auto insert_pos = cur - tot + res.dims().size() + 1;
      res = Undiagonal<T, Context>(dev_ctx, res, insert_pos, label2perm[c]);
    }
    --cur;
  }
  return res;
}

template <typename T, typename Context>
phi::DenseTensor PerformDiagonalAndReduction(
    const Context& dev_ctx,
    const phi::DenseTensor& tensor,
    const std::string& equ,
    const LabelMap& label2perm,
    const std::vector<char>& all_labels,
    const std::vector<int>& broadcast_shape,
    const LabelMap& label2type) {
  auto res = tensor;
  int tot = equ.size();
  // tiling tensor for broadcast
  std::vector<int> repeat_times;
  auto tensor_origin_shape = common::vectorize(tensor.dims());
  for (size_t i = 0; i < tensor_origin_shape.size(); ++i) {
    VLOG(4) << "broadcast shape is " << broadcast_shape[i]
            << ", tensor shape is " << tensor_origin_shape[i];
    repeat_times.push_back(broadcast_shape[i] / tensor_origin_shape[i]);
  }
  phi::DenseTensor after_tile;
  bool is_all_ones = std::all_of(
      repeat_times.begin(), repeat_times.end(), [](int x) { return x == 1; });
  if (!is_all_ones) {
    custom_kernel::TileKernel<T, Context>(
        dev_ctx, res, repeat_times, &after_tile);
    res = after_tile;
  }
  // Diagonal
  int cur = tot - 1;
  for (auto it = equ.rbegin(); it != equ.rend(); ++it) {
    char c = *it;
    if (cur != label2perm[c]) {
      // do diagonal, followed by movedim().
      VLOG(5) << "Do diagonal with shape="
              << join_strings(common::vectorize<int>(res.dims()), ',')
              << ", axis1=" << cur << ", axis2=" << label2perm[c];
      res = custom_kernel::Diagonal<T, Context>(
          dev_ctx, res, 0, cur, label2perm[c]);
      res = custom_kernel::Transpose<T, Context>(
          dev_ctx, res, perm_moveto(res.dims().size(), -1, label2perm[c]));
    }
    --cur;
  }
  // reduction
  auto indices = GetLabelIndexByType<int64_t>(
      all_labels, label2type, label2perm, LabelType::Reduction);
  VLOG(5) << "call PerformDiagonalAndReduction: with axis: "
          << join_strings(indices, ",");
  if (indices.empty()) return res;
  return custom_kernel::Sum<T, Context>(
      dev_ctx, res, phi::IntArray(indices), res.dtype(), true);
}

inline bool is_no_need_transpose(const std::vector<int>& axis) {
  for (size_t i = 0; i < axis.size(); ++i) {
    if (i != static_cast<size_t>(axis[i])) return false;
  }
  return true;
}

template <typename T, typename Context>
phi::DenseTensor PerformTranspose(const Context& dev_ctx,
                                  const phi::DenseTensor& tensor,
                                  const LabelMap& label2perm,
                                  const std::vector<char>& all_labels,
                                  const LabelMap& label2type) {
  auto axis = GetLabelIndexByType<int>(
      all_labels, label2type, label2perm, LabelType::ALL_TYPE);
  VLOG(5) << "PerformTranspose: " << join_strings(axis, ",");
  if (is_no_need_transpose(axis)) {
    return tensor;
  }
  auto ret = custom_kernel::Transpose<T, Context>(dev_ctx, tensor, axis);
  VLOG(5) << "PerformTranspose: do_transpose()";
  return ret;
}

template <typename T, typename Context>
phi::DenseTensor PerformContraction(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& operands,
    const std::vector<std::string>& input_strs,
    const std::vector<LabelMap>& label2perm,
    const std::vector<char>& all_labels,
    const LabelMap& label2type,
    const LabelMap& label2shape,
    const std::vector<std::vector<int>>& broadcast_shapes,
    std::vector<phi::DenseTensor*> cache,
    bool use_cache) {
  auto all_valid = LabelMap(1);
  auto recover_dim = GetShapeByType<int>(
      all_labels, label2type, all_valid, label2shape, {LabelType::Batch});
  auto preprocess = [&](const phi::DenseTensor& t,
                        const LabelMap& perm,
                        const std::vector<int>& broadcast,
                        int operand_idx) -> phi::DenseTensor {
    // reshape
    auto frees = GetShapeByType<int>(all_labels,
                                     label2type,
                                     perm,
                                     label2shape,
                                     {LabelType::AO, LabelType::BO});
    auto conts = GetShapeByType<int>(
        all_labels, label2type, perm, label2shape, {LabelType::Contraction});
    std::vector<char> reordered_all_labels = all_labels;
    if (operand_idx == 1) {
      reordered_all_labels = TransformLabelsOrder(all_labels,
                                                  label2type,
                                                  {LabelType::Batch,
                                                   LabelType::Contraction,
                                                   LabelType::AO,
                                                   LabelType::BO,
                                                   LabelType::Reduction});
    }
    // reduction   trans_t.ShareBufferWith(*(cache[operand_idx]));
    phi::DenseTensor trans_t;
    if (use_cache && cache[operand_idx] != nullptr &&
        cache[operand_idx]->initialized()) {
      // trans_t.ShareBufferWith(*(cache[operand_idx]));
      TensorCopy<Context>(dev_ctx, *(cache[operand_idx]), true, &trans_t);
      VLOG(5) << "Cache Used!";
    } else {
      auto reduct_t =
          PerformDiagonalAndReduction<T, Context>(dev_ctx,
                                                  t,
                                                  input_strs[operand_idx],
                                                  perm,
                                                  all_labels,
                                                  broadcast_shapes[operand_idx],
                                                  label2type);
      trans_t = PerformTranspose<T, Context>(
          dev_ctx, reduct_t, perm, reordered_all_labels, label2type);
      if (cache[operand_idx] != nullptr)
        // cache[operand_idx]->ShareBufferWith(trans_t);
        TensorCopy<Context>(dev_ctx, trans_t, true, cache[operand_idx]);
    }
    auto mul_dims = GetShapeByType<int>(
        all_labels, label2type, perm, label2shape, {LabelType::Batch});
    recover_dim.insert(recover_dim.end(), frees.begin(), frees.end());
    if (operand_idx == 0) {
      mul_dims.push_back(std::accumulate(
          frees.begin(), frees.end(), 1, std::multiplies<int>()));
      mul_dims.push_back(std::accumulate(
          conts.begin(), conts.end(), 1, std::multiplies<int>()));
    } else {
      mul_dims.push_back(std::accumulate(
          conts.begin(), conts.end(), 1, std::multiplies<int>()));
      mul_dims.push_back(std::accumulate(
          frees.begin(), frees.end(), 1, std::multiplies<int>()));
    }
    VLOG(5) << "PerformContraction: mul_dims: " << join_strings(mul_dims, ",");
    trans_t.Resize(common::make_ddim(mul_dims));
    return trans_t;
  };

  // Reduction, Reshape and Matmul
  phi::DenseTensor after_contraction;
  if (operands.size() == 2) {
    auto trans_a =
        preprocess(*(operands[0]), label2perm[0], broadcast_shapes[0], 0);
    auto trans_b =
        preprocess(*(operands[1]), label2perm[1], broadcast_shapes[1], 1);
    after_contraction = custom_kernel::Matmul<T, Context>(
        dev_ctx, trans_a, trans_b, false, false);
  } else if (operands.size() == 1) {
    after_contraction =
        preprocess(*(operands[0]), label2perm[0], broadcast_shapes[0], 0);
  }
  if (recover_dim.empty()) recover_dim.push_back(1);
  VLOG(5) << "PerformContraction: recover_dim: "
          << join_strings(recover_dim, ",");
  after_contraction.Resize(common::make_ddim(recover_dim));
  return after_contraction;
}

template <typename T, typename Context>
phi::DenseTensor TransposeToOutput(const Context& dev_ctx,
                                   const phi::DenseTensor& to_trans,
                                   const std::vector<char>& right,
                                   const std::vector<char>& all_labels) {
  std::vector<int> axis;
  for (char c : right) {
    auto it = std::find(all_labels.begin(), all_labels.end(), c);
    PADDLE_ENFORCE_NE(it,
                      all_labels.end(),
                      phi::errors::InvalidArgument("Must in all_labels."));
    axis.push_back(it - all_labels.begin());
  }
  if (is_no_need_transpose(axis)) {
    return to_trans;
  }
  VLOG(5) << "call TransposeToOutput: with axis: " << join_strings(axis, ",")
          << "  to trans dims is: " << to_trans.dims();
  auto output = custom_kernel::Transpose<T, Context>(dev_ctx, to_trans, axis);
  VLOG(5) << "After Transpose.";
  return output;
}

// template <typename T, typename Context>
// phi::DenseTensor Undiagonal(const Context& dev_ctx,
//                             const phi::DenseTensor& tensor,
//                             size_t insert_pos,
//                             size_t axis) {
//   // tensor with shape (3, 4, 5, 2, 1), insert_pos = 5, axis = 2.
//   // output is (3, 4, 5, 2, 1, 5)
//   VLOG(5) << "Start undiagonal with args: insert_pos = " << insert_pos
//           << ", axis = " << axis;
//   std::vector<int> shape(tensor.dims().size() + 1);
//   int point = 0;  // point to the tensor.dims()
//   for (size_t i = 0; i < shape.size(); ++i) {
//     if (i == insert_pos)
//       shape[i] = tensor.dims()[axis];
//     else
//       shape[i] = tensor.dims()[point++];
//   }
//   auto zeros = custom_kernel::Full<T, Context>(dev_ctx, shape, 0);
//   auto diags = custom_kernel::Transpose<T, Context>(
//       dev_ctx, tensor, perm_moveto(tensor.dims().size(), axis, -1));
//   return custom_kernel::FillDiagonalTensor<T, Context>(
//       dev_ctx, zeros, diags, 0, insert_pos, axis + (insert_pos <= axis));
// }

// template <typename T, typename Context>
// phi::DenseTensor PerformUndiagonal(const Context& dev_ctx,
//                                    const phi::DenseTensor& tensor,
//                                    int n_broadcast,
//                                    const std::string& equ) {
//   //  if the equ is 'iijjkij', then the tensor must be 'ijk', so we have
//   nenough
//   //  information to do un-diagonal with equ.
//   auto res = tensor;
//   LabelMap label2perm(-1);
//   InferLabelPerm(equ, n_broadcast, &label2perm);
//   // Un-Diagonal
//   int tot =
//       equ.size() + n_broadcast + (equ.find(".") != std::string::npos ? -1 :
//       0);
//   int cur = tot - 1;
//   for (auto it = equ.rbegin(); it != equ.rend(); ++it) {
//     char c = *it;
//     if (c == '.') {
//       cur -= n_broadcast;
//     } else {
//       if (cur != label2perm[c]) {
//         // do diagonal, followed by movedim().
//         auto insert_pos = cur - tot + res.dims().size() + 1;
//         res = Undiagonal<T, Context>(dev_ctx, res, insert_pos,
//         label2perm[c]);
//       }
//       --cur;
//     }
//   }
//   return res;
// }

template <typename T, typename Context>
void EinsumKernelImpl(const Context& dev_ctx,
                      const std::vector<char>& forward_all_labels,
                      const std::vector<const phi::DenseTensor*>& inputs,
                      const std::string& equation,
                      phi::DenseTensor* out,
                      std::vector<phi::DenseTensor*> cache,
                      bool is_forward = true) {
  VLOG(5) << "Start EinsumKernelImpl with inputs(" << inputs.size() << "): ";
  for (auto& i : inputs) {
    VLOG(5) << "      inputs [ " << i << " ].shape=" << i->dims();
  }
  ValidationCheck(equation);
  // collect the following informations to prepare einsum.
  LabelMap labelshape(0);
  LabelMap labeltype(LabelType::Reduction);
  std::vector<LabelMap> label2perms(inputs.size(), LabelMap(-1));
  std::vector<char> all_labels;  // order: ABO, AO, BO, AB, Reduce
  std::vector<std::vector<int>> broadcast_shapes(2);
  std::vector<int> output_dims;

  std::vector<phi::DDim> input_dims;
  for (auto& i : inputs) {
    input_dims.push_back(i->dims());
  }
  std::vector<std::string> input_strs;
  std::string right;
  if (!is_forward) {
    all_labels = forward_all_labels;
  }
  ParseEinsumEquation(equation,
                      input_dims,
                      &labelshape,
                      &labeltype,
                      &all_labels,
                      &label2perms,
                      &broadcast_shapes,
                      &output_dims,
                      &right,
                      &input_strs);
  if (inputs.size() > 2) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "EinsumOp kernel only support len(operands) between (0, 2]. Use "
        "opt_einsum first to convert multi-variable to binary-variable."));
  }
  auto after_contraction = PerformContraction<T, Context>(dev_ctx,
                                                          inputs,
                                                          input_strs,
                                                          label2perms,
                                                          all_labels,
                                                          labeltype,
                                                          labelshape,
                                                          broadcast_shapes,
                                                          cache,
                                                          !is_forward);
  *out = TransposeToOutput<T, Context>(
      dev_ctx, after_contraction, unique_labels(right), all_labels);
  *out = PerformUndiagonal<T, Context>(dev_ctx, *out, right);
  out->Resize(common::make_ddim(output_dims));
}

template <typename T, typename Context>
void EinsumKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& inputs,
                  const std::string& equation,
                  phi::DenseTensor* out,
                  std::vector<phi::DenseTensor*> cache,
                  std::vector<phi::DenseTensor*> xshape UNUSED) {
  std::vector<char> tmp;
  // for the sake of compatibility, we may load and run v2.3 EinsumOp. Output
  // may have nullptr and the cache.size() is not equal to inputs.size(). refer
  // to BuildPhiKernelContext for details.
  int diff = inputs.size() - cache.size();
  for (int i = 0; i < diff; ++i) {
    cache.push_back(nullptr);
  }
  EinsumKernelImpl<T, Context>(
      dev_ctx, tmp, inputs, equation, out, cache, /*forward=*/true);
}

template <typename T, typename Context>
void EinsumInferKernel(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& inputs,
                       const std::string& equation,
                       phi::DenseTensor* out) {
  std::vector<char> place_holder;
  std::vector<phi::DenseTensor*> cache_tensor(
      inputs.size());  // set empty; TA, TB, TdC
  for (size_t i = 0; i < inputs.size(); ++i) {
    cache_tensor[i] = nullptr;
  }
  EinsumKernelImpl<T, Context>(
      dev_ctx, place_holder, inputs, equation, out, cache_tensor, true);
}

template <typename T, typename Context>
phi::DenseTensor PerformTileAndReduction(
    const Context& dev_ctx,
    const LabelMap& label2type,
    const LabelMap& label2shape,
    const std::vector<int>& broadcast_shape,
    const std::vector<int> x_shape,
    std::string equ,        // value pass
    phi::DenseTensor& t) {  // NOLINT
  auto tmp_label = equ;
  auto tmp_union = unique_labels(tmp_label);
  auto op_label = std::string(tmp_union.begin(), tmp_union.end());
  VLOG(5) << "Start PerformTileAndReduction equation " << equ
          << " with operand shape: "
          << join_strings(common::vectorize<int>(t.dims()), ",");
  phi::DenseTensor ret;
  std::vector<int> repeat_times;
  std::vector<int> resize_dims;
  std::vector<int> recover_shape;
  for (int c : op_label) {
    if (label2type[c] == LabelType::Reduction) {
      repeat_times.push_back(label2shape[c]);
      resize_dims.push_back(1);
      recover_shape.push_back(label2shape[c]);
    } else {
      resize_dims.push_back(label2shape[c]);
      repeat_times.push_back(1);
      recover_shape.push_back(label2shape[c]);
    }
  }
  t.Resize(common::make_ddim(resize_dims));
  phi::DenseTensor after_tile;
  if (std::all_of(repeat_times.begin(), repeat_times.end(), [](int x) {
        return x == 1;
      })) {
    after_tile = t;
  } else {
    VLOG(4) << "do TileKernel with repeat_times="
            << join_strings(repeat_times, ",");
    custom_kernel::TileKernel<T, Context>(
        dev_ctx, t, repeat_times, &after_tile);
  }
  ret = after_tile;
  VLOG(5) << "PermformTileAndReduction: recover shape: "
          << join_strings(recover_shape, ",");
  ret.Resize(common::make_ddim(recover_shape));

  // undiagonalize by einsum equation. only contain undiagonal operations.
  phi::DenseTensor undiagonal_out;
  if (op_label != equ) {
    VLOG(5) << "Undiagonal by einsum with args: " << op_label + "->" + equ;
    custom_kernel::EinsumInferKernel<T, Context>(
        dev_ctx, {&ret}, op_label + "->" + equ, &undiagonal_out);
  } else {
    undiagonal_out = ret;
  }

  // call TileGradKernel to reverse broadcast operation.
  VLOG(5) << "After diagonalize, we have tensor with shape: "
          << join_strings(common::vectorize<int>(undiagonal_out.dims()), ',');
  repeat_times.clear();
  for (size_t i = 0; i < x_shape.size(); ++i) {
    VLOG(4) << "broadcast shape is " << broadcast_shape[i] << ", x_shape is "
            << x_shape[i];
    repeat_times.push_back(broadcast_shape[i] / x_shape[i]);
  }
  bool is_all_ones = std::all_of(
      repeat_times.begin(), repeat_times.end(), [](int x) { return x == 1; });
  if (is_all_ones) {
    VLOG(4) << "don't need broadcast recover, we just return undiagonal_out.";
    return undiagonal_out;
  }
  phi::DenseTensor tmp_x;
  phi::DenseTensor broadcast_out;
  tmp_x.Resize(common::make_ddim(x_shape));
  broadcast_out.Resize(common::make_ddim(x_shape));
  custom_kernel::TileGradKernel<T, Context>(
      dev_ctx, tmp_x, undiagonal_out, repeat_times, &broadcast_out);
  VLOG(5) << "After broadcast recover, we have tensor with shape: "
          << join_strings(common::vectorize<int>(broadcast_out.dims()), ',');
  return broadcast_out;
}

template <typename T, typename Context>
void EinsumGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& x,
                      const std::vector<const phi::DenseTensor*>& inner_cache,
                      const phi::DenseTensor& out_grad,
                      const std::string& equation,
                      std::vector<phi::DenseTensor*> x_grad) {
  VLOG(5) << "Start EinsumGradKernel:";
  LabelMap labelshape(0);
  LabelMap labeltype(LabelType::Reduction);
  std::vector<LabelMap> label2perms(x.size(), LabelMap(-1));
  std::vector<char> all_labels;  // order: ABO, AO, BO, AB, Reduce
  std::vector<std::vector<int>> broadcast_shapes(2);
  std::vector<int> output_dims;

  std::vector<phi::DDim> input_dims;
  for (auto& i : x) {
    input_dims.push_back(i->dims());
  }
  std::vector<std::string> input_strs;
  std::string right;
  ParseEinsumEquation(equation,
                      input_dims,
                      &labelshape,
                      &labeltype,
                      &all_labels,
                      &label2perms,
                      &broadcast_shapes,
                      &output_dims,
                      &right,
                      &input_strs);

  VLOG(4) << "After grad parse einsum equation.";

  auto gather_labels_except_reduction = [&labeltype](std::string all) {
    std::string res("");
    for (auto c : all)
      if (labeltype[static_cast<int>(c)] != LabelType::Reduction) res += c;
    auto tmp_unique = unique_labels(res);
    return std::string(tmp_unique.begin(), tmp_unique.end());
  };
  if (x.size() == 1) {  // Unary
    auto splits = split_string(equation, "->");
    auto left = splits[0];
    right = splits[1];
    auto new_equation = right + "->" + gather_labels_except_reduction(left);
    auto new_operands = std::vector<const phi::DenseTensor*>();
    new_operands.push_back(&out_grad);
    phi::DenseTensor before_tile;
    VLOG(5) << "new_equation is " << new_equation;
    custom_kernel::EinsumInferKernel<T, Context>(
        dev_ctx, new_operands, new_equation, &before_tile);
    *(x_grad[0]) = custom_kernel::PerformTileAndReduction<T, Context>(
        dev_ctx,
        labeltype,
        labelshape,
        broadcast_shapes[0],
        common::vectorize<int>(x[0]->dims()),
        left,
        before_tile);
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
    *(x_grad[0]) = custom_kernel::Conj<T, Context>(dev_ctx, *x_grad[0]);
#endif
  } else {
    auto splits = split_string(equation, "->");
    auto left = splits[0];
    auto ops = split_string(left, ",");
    right = splits[1];
    auto equation_for_A =
        ops[1] + "," + right + "->" + gather_labels_except_reduction(ops[0]);
    auto equation_for_B =
        right + "," + ops[0] + "->" + gather_labels_except_reduction(ops[1]);
    auto operands_for_A = std::vector<const phi::DenseTensor*>();
    auto operands_for_B = std::vector<const phi::DenseTensor*>();
    phi::DenseTensor dA, dB;
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
    auto out_grad_conj = custom_kernel::Conj<T, Context>(dev_ctx, out_grad);
#else
    auto out_grad_conj = out_grad;
#endif
    // dA = einsum(B, dC)
    operands_for_A.push_back(x[1]);
    operands_for_A.push_back(&out_grad_conj);
    // dB = einsum(dC, A)
    operands_for_B.push_back(&out_grad_conj);
    operands_for_B.push_back(x[0]);

    std::vector<phi::DenseTensor> cache(3);  // set empty; TA, TB, TdC
    if (inner_cache.size() >
        0) {  // for compatibility,  we can load and run v2.3 EinsumOp.
      // cache[0].ShareBufferWith(*(inner_cache[0]));
      // cache[1].ShareBufferWith(*(inner_cache[1]));
      TensorCopy<Context>(dev_ctx, *(inner_cache[0]), true, &cache[0]);
      TensorCopy<Context>(dev_ctx, *(inner_cache[1]), true, &cache[1]);
    }
    custom_kernel::EinsumKernelImpl<T, Context>(dev_ctx,
                                                all_labels,
                                                operands_for_A,
                                                equation_for_A,
                                                &dA,
                                                {&cache[1], &cache[2]},
                                                false);

    custom_kernel::EinsumKernelImpl<T, Context>(dev_ctx,
                                                all_labels,
                                                operands_for_B,
                                                equation_for_B,
                                                &dB,
                                                {&cache[2], &cache[0]},
                                                false);

    // release the cache tensor dTC to save memory right now. they are useless
    // now.
    cache.clear();
    if (x_grad[0]) {
      *(x_grad[0]) = custom_kernel::PerformTileAndReduction<T, Context>(
          dev_ctx,
          labeltype,
          labelshape,
          broadcast_shapes[0],
          common::vectorize<int>(x[0]->dims()),
          ops[0],
          dA);
      VLOG(4) << "After call dA";
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
      *(x_grad[0]) = custom_kernel::Conj<T, Context>(dev_ctx, *x_grad[0]);
#endif
    }
    if (x_grad[1]) {
      *(x_grad[1]) = custom_kernel::PerformTileAndReduction<T, Context>(
          dev_ctx,
          labeltype,
          labelshape,
          broadcast_shapes[1],
          common::vectorize<int>(x[1]->dims()),
          ops[1],
          dB);
#ifndef PADDLE_WITH_XPU  // xpu is not support conj now, we just disable it.
      *(x_grad[1]) = custom_kernel::Conj<T, Context>(dev_ctx, *x_grad[1]);
#endif
      VLOG(4) << "After call dA";
    }
  }
}

}  // namespace custom_kernel

// This operator is also implemented by the CPU in the npu_kernel.
PD_REGISTER_PLUGIN_KERNEL(einsum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EinsumKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(einsum_infer,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EinsumInferKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(einsum_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EinsumGradKernel,
                          double,
                          float,
                          phi::dtype::float16) {}
