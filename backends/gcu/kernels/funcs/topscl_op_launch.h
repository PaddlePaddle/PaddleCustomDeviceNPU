// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs/topscl_op_utils.h"

namespace custom_kernel {

namespace {  // NOLINT

template <typename T>
struct topscl_variable {
  explicit topscl_variable(const T& var) { value = var; }

  T value;
};

template <>
struct topscl_variable<phi::DenseTensor> {
  explicit topscl_variable(const phi::DenseTensor& tensor) {
    value = CreateTopsclTensor(tensor);
  }

  topscl::Tensor value;
};

template <>
struct topscl_variable<paddle::optional<phi::DenseTensor>> {
  explicit topscl_variable(
      const paddle::optional<phi::DenseTensor>& opt_tensor) {
    value = OptionalTensorToTopsclTensor(opt_tensor);
  }

  topscl::Tensor value;
};

template <>
struct topscl_variable<std::vector<phi::DenseTensor>> {
  explicit topscl_variable(const std::vector<phi::DenseTensor>& tensor_list) {
    for (int64_t i = 0; i < tensor_list.size(); ++i) {
      value.emplace_back(CreateTopsclTensor(tensor_list[i]));
    }
  }

  std::vector<topscl::Tensor> value;
};

template <>
struct topscl_variable<std::vector<phi::DenseTensor*>> {
  explicit topscl_variable(const std::vector<phi::DenseTensor*>& tensor_list) {
    for (int64_t i = 0; i < tensor_list.size(); ++i) {
      value_item.emplace_back(CreateTopsclTensor(*(tensor_list[i])));
    }
    for (int64_t i = 0; i < tensor_list.size(); ++i) {
      value.emplace_back(&(value_item[i]));
    }
  }

  std::vector<topscl::Tensor> value_item;
  std::vector<topscl::Tensor*> value;
};

template <>
struct topscl_variable<std::vector<phi::DenseTensor*>*> {
  explicit topscl_variable(std::vector<phi::DenseTensor*>* tensor_list) {
    for (int64_t i = 0; i < tensor_list->size(); ++i) {
      value_item.emplace_back(CreateTopsclTensor(*(tensor_list->at(i))));
    }
    for (int64_t i = 0; i < tensor_list->size(); ++i) {
      value_ptr.emplace_back(&(value_item[i]));
    }
    value = &value_ptr;
  }

  std::vector<topscl::Tensor> value_item;
  std::vector<topscl::Tensor*> value_ptr;
  std::vector<topscl::Tensor*>* value;
};

template <>
struct topscl_variable<phi::Scalar> {
  explicit topscl_variable(const phi::Scalar& scalar) {
    value = ScalarToTopsclScalar(scalar);
  }

  topscl::Scalar value;
};

template <>
struct topscl_variable<paddle::optional<phi::Scalar>> {
  explicit topscl_variable(const paddle::optional<phi::Scalar>& opt_scalar) {
    value = OptionalScalarToTopsclScalar(opt_scalar);
  }

  topscl::Scalar value;
};

template <>
struct topscl_variable<phi::DataType> {
  explicit topscl_variable(const phi::DataType& data_type) {
    value = DataTypeToTopsclDataType(data_type);
  }

  topscl::DType value;
};

template <>
struct topscl_variable<common::DataLayout> {
  explicit topscl_variable(const common::DataLayout& layout) {
    value = DataLayoutToTopsclMemoryFormat(layout);
  }

  topscl::MemoryFormat value;
};

template <>
struct topscl_variable<phi::IntArray> {
  explicit topscl_variable(const phi::IntArray& array) {
    value = IntArrayToVector64(array);
  }

  std::vector<int64_t> value;
};

#define DEFINE_LAUNCH_TOPSCL_OP(op_type)                                \
  template <typename... Args>                                           \
  void launch_pd_##op_type(const phi::CustomContext& dev_ctx,           \
                           const std::string& abstract_info,            \
                           phi::DenseTensor& out,                       \
                           const Args&... args) {                       \
    auto stream = static_cast<void*>(dev_ctx.stream());                 \
    auto xout = topscl_variable<phi::DenseTensor>(out);                 \
    {                                                                   \
      GCU_AOT_KERNEL_TRACE(abstract_info);                              \
      topscl_api::pd_##op_type(                                         \
          &(xout.value), topscl_variable<Args>(args).value..., stream); \
      GcuOpMaybeStreamSync(dev_ctx);                                    \
    }                                                                   \
  }

#define DEFINE_LAUNCH_TOPSCL_OP_WITH_VECTOR_OUTS(op_type)              \
  template <typename... Args>                                          \
  void launch_pd_##op_type(const phi::CustomContext& dev_ctx,          \
                           const std::string& abstract_info,           \
                           std::vector<phi::DenseTensor*>* out,        \
                           const Args&... args) {                      \
    auto stream = static_cast<void*>(dev_ctx.stream());                \
    auto xout = topscl_variable<std::vector<phi::DenseTensor*>*>(out); \
    {                                                                  \
      GCU_AOT_KERNEL_TRACE(abstract_info);                             \
      topscl_api::pd_##op_type(                                        \
          xout.value, topscl_variable<Args>(args).value..., stream);   \
      GcuOpMaybeStreamSync(dev_ctx);                                   \
    }                                                                  \
  }

#define LAUNCH_TOPSCLOP(topsclop, dev_ctx, topsclop_args...)            \
  {                                                                     \
    auto op_info = [&]() -> std::string {                               \
      return custom_kernel::GetOpInfo(                                  \
          #topsclop,                                                    \
          ##topsclop_args,                                              \
          static_cast<topsStream_t>(dev_ctx.stream()));                 \
    };                                                                  \
    VLOG(6) << "[AOT_KERNEL] Start to launch topscl op, " << op_info(); \
    std::string abstract_info =                                         \
        custom_kernel::GetAbstractInfo(#topsclop, ##topsclop_args);     \
    custom_kernel::launch_pd_##topsclop(                                \
        dev_ctx, abstract_info, ##topsclop_args);                       \
    VLOG(6) << "Launch topscl op successfully, details:" << op_info();  \
  }

#define LAUNCH_TOPSCLOP_WITH_RAW_TOPSCL_DEF(                                \
    topsclop, dev_ctx, abstract_info, ...)                                  \
  {                                                                         \
    VLOG(6) << "[AOT_KERNEL] Start to launch topscl op, " << abstract_info; \
    GCU_AOT_KERNEL_TRACE(abstract_info);                                    \
    topscl_api::pd_##topsclop(__VA_ARGS__,                                  \
                              static_cast<topsStream_t>(dev_ctx.stream())); \
    VLOG(6) << "Launch topscl op successfully, details:" << abstract_info;  \
    GcuOpMaybeStreamSync(dev_ctx);                                          \
  }

}  // namespace

// reduce op
DEFINE_LAUNCH_TOPSCL_OP(max)
DEFINE_LAUNCH_TOPSCL_OP(take_along_axis)
DEFINE_LAUNCH_TOPSCL_OP(strided_slice)
DEFINE_LAUNCH_TOPSCL_OP(eye)
DEFINE_LAUNCH_TOPSCL_OP(diag)
DEFINE_LAUNCH_TOPSCL_OP(diagonal)
DEFINE_LAUNCH_TOPSCL_OP(increment)
DEFINE_LAUNCH_TOPSCL_OP(nms)
DEFINE_LAUNCH_TOPSCL_OP_WITH_VECTOR_OUTS(unstack)

}  // namespace custom_kernel
