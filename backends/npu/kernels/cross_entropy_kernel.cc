/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/string_helper.h"
namespace custom_kernel {

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out);

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out);

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out);

template <typename T>
T TolerableValue(const T& x) {
  const T kApproInf = 1e20;
  if (x == INFINITY) return kApproInf;
  if (x == -INFINITY) return -kApproInf;
  return x;
}

static inline int SizeToAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, typename U>
void CrossEntropyCpuImpl(const T* logits,
                         const U* labels,
                         bool soft_label,
                         size_t batch_size,
                         size_t num_classes,
                         int ignore_index,
                         int axis_dim,
                         T* loss_out) {
  auto num_remain = num_classes / axis_dim;
  if (soft_label) {
    for (auto i = 0; i < batch_size; ++i) {
      for (auto k = 0; k < num_remain; ++k) {
        loss_out[i * num_remain + k] = 0;
        for (auto j = 0; j < axis_dim; ++j) {
          auto idx = i * num_classes + j * num_remain + k;
          loss_out[i * num_remain + k] -= static_cast<T>(
              static_cast<float>(labels[idx]) *
              TolerableValue<float>(std::log(static_cast<float>(logits[idx]))));
        }
      }
    }
  } else {
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_remain; j++) {
        int lbl = static_cast<int>(labels[i * num_remain + j]);
        if (lbl != ignore_index) {
          PD_CHECK(lbl >= 0,
                   "label value should >= 0 when label "
                   "value(%f) not equal to ignore_index(%f)",
                   lbl,
                   ignore_index);
          PD_CHECK(lbl < axis_dim,
                   "label value should less than the shape of axis dimension "
                   "when label value(%f) not equal to ignore_index(%f), But "
                   "received label value as %ld and shape of axis dimension "
                   "is %d",
                   lbl,
                   ignore_index,
                   lbl,
                   axis_dim);
        }
        int index = i * num_classes + lbl * num_remain + j;
        int loss_idx = i * num_remain + j;
        loss_out[loss_idx] = static_cast<T>(
            lbl == ignore_index ? 0
                                : -TolerableValue<float>(std::log(
                                      static_cast<float>(logits[index]))));
      }
    }
  }
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss) {
  auto logits_dims = logits.dims();
  const int rank = logits_dims.size();
  const int use_axis = axis < 0 ? axis + rank : axis;
  const int axis_dim = logits_dims[use_axis];
  const int n = SizeToAxis(use_axis, logits_dims);
  const int d = SizeFromAxis(use_axis, logits_dims);
  auto loss_dims = loss->dims();
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(loss);
  dev_ctx.template Alloc<T>(softmax);
  if (use_softmax) {
    if (logits.dtype() == phi::DataType::FLOAT64) {
      phi::DenseTensor cast_logits, cast_softmax;
      cast_logits.Resize(logits.dims());
      dev_ctx.template Alloc<float>(&cast_logits);
      cast_softmax.Resize(softmax->dims());
      dev_ctx.template Alloc<float>(&cast_softmax);
      const auto& runner_cast_logits = NpuOpRunner(
          "Cast", {logits}, {cast_logits}, {{"dst_type", ACL_FLOAT}});
      runner_cast_logits.Run(stream);
      const auto& runner_softmax =
          NpuOpRunner("SoftmaxV2",
                      {cast_logits},
                      {cast_softmax},
                      {{"axes", std::vector<int32_t>({axis})}});
      runner_softmax.Run(stream);
      const auto& runner_cast_softmax = NpuOpRunner(
          "Cast", {cast_softmax}, {*softmax}, {{"dst_type", ACL_DOUBLE}});
      runner_cast_softmax.Run(stream);
    } else {
      const auto& runner_softmax =
          NpuOpRunner("SoftmaxV2",
                      {logits},
                      {*softmax},
                      {{"axes", std::vector<int32_t>({axis})}});
      runner_softmax.Run(stream);
    }

  } else {
    // cause of input is softmax, copy to output softmax, directly
    phi::Copy<Context>(dev_ctx, logits, dev_ctx.GetPlace(), false, softmax);
  }

  phi::DenseTensorMeta labels_zero_meta = {phi::DataType::INT32, labels.dims()};
  phi::DenseTensor labels_zero_tensor;
  labels_zero_tensor.set_meta(labels_zero_meta);
  dev_ctx.template Alloc<T>(&labels_zero_tensor);
  EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, labels_zero_tensor);
  labels_zero_tensor.Resize(labels.dims());

  phi::DenseTensorMeta labels_bool_meta = {phi::DataType::BOOL, labels.dims()};
  phi::DenseTensor labels_bool;
  labels_bool.set_meta(labels_bool_meta);
  custom_kernel::GreaterEqualKernel<T, Context>(
      dev_ctx, labels, labels_zero_tensor, &labels_bool);

  phi::DenseTensor labels_result;
  labels_result.Resize(phi::make_ddim({}));
  std::vector<int64_t> emptyVector;
  custom_kernel::AllKernel<bool, Context>(
      dev_ctx, labels_bool, emptyVector, false, &labels_result);

  std::vector<bool> labels_condition_vec;
  TensorToVector(dev_ctx, *&labels_result, dev_ctx, &labels_condition_vec);
  bool condition = labels_condition_vec[0];

  // Use NPU IR
  if (!soft_label && labels.dims()[use_axis] == 1 && condition && use_softmax) {
    PADDLE_ENFORCE_EQ(soft_label,
                      false,
                      phi::errors::Unimplemented(
                          "soft_label=True is not supported in "
                          "the npu kernel of softmax_with_cross_entropy."));

    PADDLE_ENFORCE_EQ(
        labels.dims()[use_axis],
        1,
        phi::errors::Unimplemented(
            "The size of labels of axis should be 1"
            "but got size of labels is %d and phi::funcs::SizeToAxis is %d.",
            labels.dims()[use_axis]));

    PADDLE_ENFORCE_EQ(use_softmax,
                      true,
                      phi::errors::Unimplemented(
                          "use_softmax=false is not supported in "
                          "the npu kernel of softmax_with_cross_entropy."));

    phi::DenseTensor logits_2d(logits), labels_1d(labels), loss_1d(*loss),
        transformed_logits;
    auto dims = labels.dims();
    // transpose
    std::vector<int> perm;
    std::vector<int> perm_shape;
    for (int i = 0; i < rank; i++) {
      if (i != use_axis) {
        perm.push_back(i);
      }
    }
    perm.push_back(use_axis);
    for (int i = 0; i < rank; i++) {
      perm_shape.push_back(logits_dims[perm[i]]);
    }
    transformed_logits.Resize(phi::make_ddim(perm_shape));
    dev_ctx.template Alloc<T>(&transformed_logits);
    custom_kernel::TransposeKernel<T, Context>(
        dev_ctx, logits_2d, perm, &transformed_logits);

    transformed_logits.Resize({logits.numel() / axis_dim, axis_dim});
    labels_1d.Resize({labels.numel()});
    loss_1d.Resize({labels.numel()});

    // SparseSoftmaxCrossEntropyWithLogits
    phi::DenseTensor backprop_2d;
    phi::DenseTensorMeta meta = {logits.dtype(),
                                 {logits.numel() / axis_dim, axis_dim}};
    backprop_2d.set_meta(meta);
    dev_ctx.template Alloc<T>(&backprop_2d);
    const auto& runner_s = NpuOpRunner("SparseSoftmaxCrossEntropyWithLogits",
                                       {transformed_logits, labels_1d},
                                       {loss_1d, backprop_2d},
                                       {});
    runner_s.Run(stream);

    // deal with ignore_index
    phi::DenseTensor ignore_index_tensor;
    ignore_index_tensor.Resize(labels_1d.dims());
    FillNpuTensorWithConstant<int>(&ignore_index_tensor, dev_ctx, ignore_index);
    ignore_index_tensor.Resize(labels_1d.dims());
    phi::DenseTensorMeta out_meta = {phi::DataType::BOOL, labels_1d.dims()};
    phi::DenseTensor mid_out;
    mid_out.set_meta(out_meta);
    custom_kernel::EqualKernel<T, Context>(
        dev_ctx, labels_1d, ignore_index_tensor, &mid_out);
    int zero_ = 0;
    phi::DenseTensorMeta zero_meta = {phi::DataType::INT32, loss_1d.dims()};
    phi::DenseTensor zero_tensor;
    zero_tensor.set_meta(zero_meta);
    dev_ctx.template Alloc<T>(&zero_tensor);
    EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, zero_tensor);
    zero_tensor.Resize(loss_1d.dims());

    custom_kernel::WhereKernel<T, Context>(
        dev_ctx, mid_out, zero_tensor, loss_1d, &loss_1d);
    loss_1d.Resize(dims);
  } else if (labels.dims()[use_axis] != 1 && use_softmax) {
    phi::DenseTensor loss_1d(*loss), transformed_logits, transformed_lables;
    auto dims = labels.dims();
    dims[use_axis] = 1;

    // transpose
    std::vector<int> perm;
    std::vector<int> perm_shape;
    for (int i = 0; i < rank; i++) {
      if (i != use_axis) {
        perm.push_back(i);
      }
    }
    perm.push_back(use_axis);
    for (int i = 0; i < rank; i++) {
      perm_shape.push_back(logits_dims[perm[i]]);
    }
    transformed_logits.Resize(phi::make_ddim(perm_shape));
    transformed_lables.Resize(phi::make_ddim(perm_shape));
    dev_ctx.template Alloc<T>(&transformed_logits);
    dev_ctx.template Alloc<T>(&transformed_lables);
    custom_kernel::TransposeKernel<T, Context>(
        dev_ctx, logits, perm, &transformed_logits);
    custom_kernel::TransposeKernel<T, Context>(
        dev_ctx, labels, perm, &transformed_lables);

    transformed_logits.Resize({logits.numel() / axis_dim, axis_dim});
    transformed_lables.Resize({labels.numel() / axis_dim, axis_dim});
    loss_1d.Resize({logits.numel() / axis_dim});

    // SoftmaxCrossEntropyWithLogits
    phi::DenseTensor backprop_2d;
    phi::DenseTensorMeta meta = {logits.dtype(),
                                 {logits.numel() / axis_dim, axis_dim}};
    backprop_2d.set_meta(meta);
    dev_ctx.template Alloc<T>(&backprop_2d);
    const auto& runner_s = NpuOpRunner("SoftmaxCrossEntropyWithLogits",
                                       {transformed_logits, transformed_lables},
                                       {loss_1d, backprop_2d},
                                       {});
    runner_s.Run(stream);
    loss_1d.Resize(dims);
  } else {
    if (labels.numel() != n) {
      VLOG(4) << "The size of labels should be equal to phi::funcs::SizeToAxis "
                 "of logits, but got size of labels is "
              << labels.numel() << " and phi::funcs::SizeToAxis is " << n;
    }
    if (soft_label) {
      VLOG(4) << "soft_label = True is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    if (ignore_index != -100) {
      VLOG(4) << "ignore_index = -1 is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    VLOG(4) << "CrossEntropyWithSoftmaxGradKernel of npu is implemented using "
               "the CPU";

    std::vector<T> softmax_data_vec;
    TensorToVector(dev_ctx, *softmax, dev_ctx, &softmax_data_vec);

    phi::DenseTensor cpu_loss_out_tensor;
    cpu_loss_out_tensor.Resize({n, d / axis_dim});
    auto cpu_loss_out_data =
        dev_ctx.template HostAlloc<T>(&cpu_loss_out_tensor);
    if (soft_label) {
      std::vector<T> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, T>(softmax_data_vec.data(),
                                labels_data_vec.data(),
                                soft_label,
                                n,
                                d,
                                ignore_index,
                                axis_dim,
                                cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT32) {
      std::vector<int32_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int32_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT64) {
      std::vector<int64_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int64_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT16) {
      std::vector<int16_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int16_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::INT8) {
      std::vector<int8_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, int8_t>(softmax_data_vec.data(),
                                     labels_data_vec.data(),
                                     soft_label,
                                     n,
                                     d,
                                     ignore_index,
                                     axis_dim,
                                     cpu_loss_out_data);
    } else if (labels.dtype() == phi::DataType::UINT8) {
      std::vector<uint8_t> labels_data_vec;
      TensorToVector(dev_ctx, *&labels, dev_ctx, &labels_data_vec);
      CrossEntropyCpuImpl<T, uint8_t>(softmax_data_vec.data(),
                                      labels_data_vec.data(),
                                      soft_label,
                                      n,
                                      d,
                                      ignore_index,
                                      axis_dim,
                                      cpu_loss_out_data);
    } else {
      PD_CHECK(false, "The dtype of label must be int.");
    }
    TensorCopy(dev_ctx, cpu_loss_out_tensor, true, loss);
    loss->Resize(loss_dims);
  }
}

template <typename T, typename Context, typename LabelT>
void CrossEntropyWithSoftmaxGradCPUKernel(const Context& dev_ctx,
                                          const phi::DenseTensor& label,
                                          const phi::DenseTensor& softmax,
                                          const phi::DenseTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis,
                                          phi::DenseTensor* logits_grad) {
  auto logits_grad_dims = logits_grad->dims();
  dev_ctx.template Alloc<T>(logits_grad);
  phi::DenseTensor cpu_logits_grad_tensor;
  cpu_logits_grad_tensor.Resize(logits_grad_dims);
  auto cpu_logits_grad_data =
      dev_ctx.template HostAlloc<T>(&cpu_logits_grad_tensor);

  if (logits_grad != &softmax || !use_softmax) {
    std::vector<T> softmax_data_vec;
    TensorToVector(dev_ctx, *&softmax, dev_ctx, &softmax_data_vec);
    memcpy(cpu_logits_grad_data,
           softmax_data_vec.data(),
           softmax.numel() * sizeof(T));
  }

  const int rank = logits_grad_dims.size();
  const int axis_v = axis < 0 ? axis + rank : axis;
  int axis_dim = logits_grad_dims[axis_v];
  PD_CHECK(axis_dim > 0,
           "The axis dimention should be larger than 0, but received "
           "axis dimention is %d.",
           axis_dim);

  const int n = SizeToAxis(axis_v, logits_grad_dims);
  PD_CHECK(n > 0,
           "The size of axis should be larger than 0, but received "
           "SizeToAxis of logit_grad is %d.",
           n);

  const int d = SizeFromAxis(axis_v, logits_grad_dims);
  int remain = d / axis_dim;

  std::vector<T> out_grad_vec;
  std::vector<LabelT> label_data_vec;
  TensorToVector(dev_ctx, *&loss_grad, dev_ctx, &out_grad_vec);
  TensorToVector(dev_ctx, *&label, dev_ctx, &label_data_vec);

  if (!use_softmax) {
    // use_softmax step1
    if (soft_label) {
      for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < axis_dim; ++j) {
          for (auto k = 0; k < remain; ++k) {
            auto index = i * d + j * remain + k;
            auto l_index = i * remain + k;
            cpu_logits_grad_data[index] =
                -static_cast<T>(label_data_vec[index]) /
                cpu_logits_grad_data[index] * out_grad_vec[l_index];
          }
        }
      }
    } else {
      // use_softmax step2
      for (int i = 0; i < n; ++i) {         // for each sample_1_dim
        for (int j = 0; j < remain; j++) {  // for each sample_other_dims
          int idx = i * remain + j;
          auto lbl = static_cast<int64_t>(label_data_vec[idx]);
          if (lbl == ignore_index) {
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              cpu_logits_grad_data[i * d + k * remain + j] = 0;
            }
          } else {
            // only for this sample's label_idx, the label is 1, others is 0,
            // so, only compute this label_idx's class
            cpu_logits_grad_data[i * d + lbl * remain + j] =
                (static_cast<T>(-1) /
                 cpu_logits_grad_data[i * d + lbl * remain + j]) *
                out_grad_vec[idx];
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              if (static_cast<LabelT>(k) !=
                  label_data_vec[idx]) {  // label_data_vec[idx]: this
                                          // sample's label
                cpu_logits_grad_data[i * d + k * remain + j] = 0;
              }
            }
          }
        }
      }
    }
    TensorCopy(dev_ctx, cpu_logits_grad_tensor, true, logits_grad);
    return;
  }
  // for use_softmax=False, continue
  if (soft_label) {
    // when soft_label = True, ignore_index is not supported
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < axis_dim; ++j) {
        for (auto k = 0; k < remain; ++k) {
          auto index = i * d + j * remain + k;
          auto l_index = i * remain + k;
          cpu_logits_grad_data[index] =
              out_grad_vec[l_index] * (cpu_logits_grad_data[index] -
                                       static_cast<T>(label_data_vec[index]));
        }
      }
    }
  } else {
    for (auto i = 0; i < n; ++i) {
      for (auto j = 0; j < axis_dim; ++j) {
        for (auto k = 0; k < remain; ++k) {
          auto index = i * d + j * remain + k;
          auto l_index = i * remain + k;

          cpu_logits_grad_data[index] =
              out_grad_vec[l_index] * cpu_logits_grad_data[index];
        }
      }
    }
    for (int i = 0; i < n; ++i) {         // for each sample_1_dim
      for (int j = 0; j < remain; j++) {  // for each sample_other_dims
        int idx = i * remain + j;
        auto lbl = static_cast<int64_t>(label_data_vec[idx]);
        if (lbl == ignore_index) {
          for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
            cpu_logits_grad_data[i * d + k * remain + j] = 0;
          }
        } else {
          cpu_logits_grad_data[i * d + lbl * remain + j] -= out_grad_vec[idx];
        }
      }
    }
  }
  TensorCopy(dev_ctx, cpu_logits_grad_tensor, true, logits_grad);
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const phi::DenseTensor& labels,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {
  auto logits_grad_dims = logits_grad->dims();
  const int rank = logits_grad_dims.size();
  const int use_axis = axis < 0 ? axis + rank : axis;
  const int axis_dim = logits_grad_dims[use_axis];
  const int n = SizeToAxis(use_axis, logits_grad_dims);
  // Use NPU IR
  if (!soft_label && labels.numel() == n && ignore_index == -100) {
    int cls_num = softmax.dims()[softmax.dims().size() - 1];
    auto stream = dev_ctx.stream();
    // cast label from int64/int32 to int32 for OneHotD
    phi::DenseTensor casted_labels;
    if (labels.dtype() != phi::DataType::INT32) {
      phi::DenseTensorMeta casted_labels_meta = {phi::DataType::INT32,
                                                 labels.dims()};
      casted_labels.set_meta(casted_labels_meta);
      dev_ctx.template Alloc<int32_t>(&casted_labels);
      auto dst_dtype = ConvertToNpuDtype(phi::DataType::INT32);
      const auto& runner_cast_label =
          NpuOpRunner("Cast",
                      {labels},
                      {casted_labels},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_label.Run(stream);
    } else {
      casted_labels = labels;
    }

    // on and off
    phi::DenseTensor on_tensor, off_tensor;
    phi::DenseTensorMeta on_off_meta = {phi::DataType::INT32, {1}};
    on_tensor.set_meta(on_off_meta);
    off_tensor.set_meta(on_off_meta);
    dev_ctx.template Alloc<int32_t>(&on_tensor);
    dev_ctx.template Alloc<int32_t>(&off_tensor);
    FillNpuTensorWithConstant<int32_t>(
        &on_tensor, dev_ctx, static_cast<int>(1));
    FillNpuTensorWithConstant<int32_t>(
        &off_tensor, dev_ctx, static_cast<int>(0));

    // one_hot
    phi::DenseTensor tmp_onehot;
    phi::DenseTensorMeta tmp_onehot_meta = {on_tensor.dtype(), softmax.dims()};
    tmp_onehot.set_meta(tmp_onehot_meta);
    dev_ctx.template Alloc<int32_t>(&tmp_onehot);

    NpuOpRunner runner_onehot;
    runner_onehot.SetType("OneHot")
        .AddInput(casted_labels)
        .AddInput(dev_ctx,
                  std::vector<int32_t>(1, static_cast<int32_t>(cls_num)))
        .AddInput(on_tensor)
        .AddInput(off_tensor)
        .AddOutput(tmp_onehot)
        .AddAttr("axis", -1);
    runner_onehot.Run(stream);

    // cast one_hot from int32 to T
    phi::DenseTensor casted_onehot;
    if (softmax.dtype() != phi::DataType::INT32) {
      phi::DenseTensorMeta onehot_meta = {softmax.dtype(), tmp_onehot.dims()};
      casted_onehot.set_meta(onehot_meta);
      dev_ctx.template Alloc<T>(&casted_onehot);
      auto dst_dtype = ConvertToNpuDtype(softmax.dtype());
      const auto& runner_cast_onehot =
          NpuOpRunner("Cast",
                      {tmp_onehot},
                      {casted_onehot},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_onehot.Run(stream);
    } else {
      casted_onehot = tmp_onehot;
    }

    // sub
    phi::DenseTensor tmp_sub;
    phi::DenseTensorMeta tmp_sub_meta = {softmax.dtype(), softmax.dims()};
    tmp_sub.set_meta(tmp_sub_meta);
    dev_ctx.template Alloc<T>(&tmp_sub);

    const auto& runner_sub =
        NpuOpRunner("Sub", {softmax, casted_onehot}, {tmp_sub}, {});
    runner_sub.Run(stream);

    // mul
    dev_ctx.template Alloc<T>(logits_grad);
    const auto& runner_mul =
        NpuOpRunner("Mul", {loss_grad, tmp_sub}, {*logits_grad}, {});
    runner_mul.Run(stream);
  } else {  // Use CPU impl
    if (labels.numel() != n) {
      VLOG(4) << "The size of labels should be equal to phi::funcs::SizeToAxis "
                 "of logits, but got size of labels is "
              << labels.numel() << " and phi::funcs::SizeToAxis is " << n;
    }
    if (soft_label) {
      VLOG(4) << "soft_label = True is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    if (ignore_index != -1) {
      VLOG(4) << "ignore_index = -1 is not supported in the npu kernel of "
                 "softmax_with_cross_entropy.";
    }
    VLOG(4) << "CrossEntropyWithSoftmaxGradKernel of npu is implemented using "
               "the CPU";

    if (soft_label) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, T>(dev_ctx,
                                                          labels,
                                                          softmax,
                                                          loss_grad,
                                                          soft_label,
                                                          use_softmax,
                                                          numeric_stable_mode,
                                                          ignore_index,
                                                          axis,
                                                          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT32) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int32_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT64) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int64_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT16) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int16_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::INT8) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, int8_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else if (labels.dtype() == phi::DataType::UINT8) {
      CrossEntropyWithSoftmaxGradCPUKernel<T, Context, uint8_t>(
          dev_ctx,
          labels,
          softmax,
          loss_grad,
          soft_label,
          use_softmax,
          numeric_stable_mode,
          ignore_index,
          axis,
          logits_grad);
    } else {
      PD_CHECK(false, "The dtype of labels must be int.");
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {}
