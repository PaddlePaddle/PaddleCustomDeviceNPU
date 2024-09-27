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
#include "kernels/funcs/topsaten_op_utils.h"
#include "topsaten/topsaten.h"

namespace custom_kernel {

namespace {  // NOLINT

template <typename T>
struct topsaten_variable {
  explicit topsaten_variable(const T& var) { value = var; }

  T value;
};

template <>
struct topsaten_variable<phi::DenseTensor> {
  explicit topsaten_variable(const phi::DenseTensor& tensor) {
    value = CreateTopsatenTensor(tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<paddle::optional<phi::DenseTensor>> {
  explicit topsaten_variable(
      const paddle::optional<phi::DenseTensor>& opt_tensor) {
    value = OptionalTensorToTopsatenTensor(opt_tensor);
  }

  topsatenTensor value;
};

template <>
struct topsaten_variable<std::vector<phi::DenseTensor>> {
  explicit topsaten_variable(const std::vector<phi::DenseTensor>& tensor_list) {
    for (int64_t i = 0; i < tensor_list.size(); ++i) {
      value.emplace_back(CreateTopsatenTensor(tensor_list[i]));
    }
  }

  std::vector<topsatenTensor> value;
};

template <>
struct topsaten_variable<phi::Scalar> {
  explicit topsaten_variable(const phi::Scalar& scalar) {
    value = ScalarToTopsatenScalar(scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<paddle::optional<phi::Scalar>> {
  explicit topsaten_variable(const paddle::optional<phi::Scalar>& opt_scalar) {
    value = OptionalScalarToTopsatenScalar(opt_scalar);
  }

  topsatenScalar_t value;
};

template <>
struct topsaten_variable<phi::IntArray> {
  explicit topsaten_variable(const phi::IntArray& array) {
    value = IntArrayToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<paddle::optional<phi::IntArray>> {
  explicit topsaten_variable(const paddle::optional<phi::IntArray>& array) {
    value = OptionalIntArrayToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<std::vector<int64_t>> {
  explicit topsaten_variable(const std::vector<int64_t>& array) {
    value = IntArrayToTopsatenSize(array);
  }

  topsatenSize_t value;
};

template <>
struct topsaten_variable<phi::DataType> {
  explicit topsaten_variable(const phi::DataType& data_type) {
    value = DataTypeToTopsatenDataType(data_type);
  }

  topsatenDataType_t value;
};

template <>
struct topsaten_variable<std::pair<uint64_t, uint64_t>> {
  explicit topsaten_variable(const std::pair<uint64_t, uint64_t>& gen_data) {
    value.seed = gen_data.first;
    value.offset = gen_data.second;
  }

  topsatenGenerator_t value;
};

#define ATEN_OP_CALL_MAYBE_SYNC_WITH_INFO(func, ctx, abstract_info)           \
  do {                                                                        \
    VLOG(6) << "[AOT_KERNEL] Call tops aten op: "                             \
            << GetOpNameFromCallStatement(std::string(#func));                \
    std::string key = (abstract_info);                                        \
    if (key == "DEFAULT") {                                                   \
      key = GetOpNameFromCallStatement(std::string(#func));                   \
    }                                                                         \
    GCU_AOT_KERNEL_TRACE(key);                                                \
    auto status = (func);                                                     \
    PADDLE_ENFORCE_EQ(                                                        \
        status,                                                               \
        TOPSATEN_STATUS_SUCCESS,                                              \
        phi::errors::Fatal("Failed to call aten op, get error: %d", status)); \
    GcuOpMaybeStreamSync(ctx);                                                \
  } while (false)

#define ATEN_OP_CALL_MAYBE_SYNC(func, ctx) \
  ATEN_OP_CALL_MAYBE_SYNC_WITH_INFO(func, ctx, "DEFAULT")

#define DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(namespace, topsatenop) \
  template <typename... Args>                                          \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,          \
                           const std::string& abstract_info,           \
                           phi::DenseTensor& out,                      \
                           const Args&... args) {                      \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());         \
    auto xout = topsaten_variable<phi::DenseTensor>(out);              \
    topsatenStatus_t status;                                           \
    {                                                                  \
      GCU_AOT_KERNEL_TRACE(abstract_info);                             \
      status = namespace ::topsatenop(                                 \
          xout.value, topsaten_variable<Args>(args).value..., stream); \
      GcuOpMaybeStreamSync(dev_ctx);                                   \
    }                                                                  \
    return status;                                                     \
  }

#define DEFINE_LAUNCH_TOPSATENOP(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topsaten, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_VLLM(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topsvllm, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_TNNC(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topstnnc, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_PADDLE(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topspaddle, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_XLA(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_WITH_NAMESPACE(topsxla, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_OUT2_WITH_NAMESPACE(namespace, topsatenop)   \
  template <typename... Args>                                                 \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,                 \
                           const std::string& abstract_info,                  \
                           phi::DenseTensor& out1,                            \
                           phi::DenseTensor& out2,                            \
                           const Args&... args) {                             \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());                \
    auto xout1 = topsaten_variable<phi::DenseTensor>(out1);                   \
    auto xout2 = topsaten_variable<phi::DenseTensor>(out2);                   \
    topsatenStatus_t status;                                                  \
    {                                                                         \
      GCU_AOT_KERNEL_TRACE(abstract_info);                                    \
      status = namespace ::topsatenop(xout1.value,                            \
                                      xout2.value,                            \
                                      topsaten_variable<Args>(args).value..., \
                                      stream);                                \
      GcuOpMaybeStreamSync(dev_ctx);                                          \
    }                                                                         \
    return status;                                                            \
  }

#define DEFINE_LAUNCH_TOPSATENOP_OUT3_WITH_NAMESPACE(namespace, topsatenop)   \
  template <typename... Args>                                                 \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,                 \
                           const std::string& abstract_info,                  \
                           phi::DenseTensor& out1,                            \
                           phi::DenseTensor& out2,                            \
                           phi::DenseTensor& out3,                            \
                           const Args&... args) {                             \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());                \
    auto xout1 = topsaten_variable<phi::DenseTensor>(out1);                   \
    auto xout2 = topsaten_variable<phi::DenseTensor>(out2);                   \
    auto xout3 = topsaten_variable<phi::DenseTensor>(out3);                   \
    topsatenStatus_t status;                                                  \
    {                                                                         \
      GCU_AOT_KERNEL_TRACE(abstract_info);                                    \
      status = namespace ::topsatenop(xout1.value,                            \
                                      xout2.value,                            \
                                      xout3.value,                            \
                                      topsaten_variable<Args>(args).value..., \
                                      stream);                                \
      GcuOpMaybeStreamSync(dev_ctx);                                          \
    }                                                                         \
    return status;                                                            \
  }

#define DEFINE_LAUNCH_TOPSATENOP_VOUT_WITH_NAMESPACE(namespace, topsatenop) \
  template <typename... Args>                                               \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,               \
                           const std::string& abstract_info,                \
                           std::vector<phi::DenseTensor>& out,              \
                           const Args&... args) {                           \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());              \
    auto xout = topsaten_variable<std::vector<phi::DenseTensor>>(out);      \
    topsatenStatus_t status;                                                \
    {                                                                       \
      GCU_AOT_KERNEL_TRACE(abstract_info);                                  \
      status = namespace ::topsatenop(                                      \
          xout.value, topsaten_variable<Args>(args).value..., stream);      \
      GcuOpMaybeStreamSync(dev_ctx);                                        \
    }                                                                       \
    return status;                                                          \
  }

#define DEFINE_LAUNCH_TOPSATENOP_VOUT(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_VOUT_WITH_NAMESPACE(topsaten, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_OUT2_WITH_NAMESPACE(topsaten, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_OUT3(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_OUT3_WITH_NAMESPACE(topsaten, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_OUT2_VLLM(topsatenop) \
  DEFINE_LAUNCH_TOPSATENOP_OUT2_WITH_NAMESPACE(topsvllm, topsatenop)

#define DEFINE_LAUNCH_TOPSATENOP_NATIVE_BATCH_NORM(topsatenop)   \
  template <typename... Args>                                    \
  auto launch_##topsatenop(const phi::CustomContext& dev_ctx,    \
                           const std::string& abstract_info,     \
                           phi::DenseTensor& out,                \
                           phi::DenseTensor& save_mean,          \
                           phi::DenseTensor& save_var,           \
                           const phi::DenseTensor& x,            \
                           const phi::DenseTensor& scale,        \
                           const phi::DenseTensor& bias,         \
                           phi::DenseTensor& run_mean,           \
                           phi::DenseTensor& run_var,            \
                           bool training,                        \
                           double momentum,                      \
                           double eps) {                         \
    auto stream = static_cast<topsStream_t>(dev_ctx.stream());   \
    auto xout1 = topsaten_variable<phi::DenseTensor>(out);       \
    auto xout2 = topsaten_variable<phi::DenseTensor>(save_mean); \
    auto xout3 = topsaten_variable<phi::DenseTensor>(save_var);  \
    auto xout4 = topsaten_variable<phi::DenseTensor>(run_mean);  \
    auto xout5 = topsaten_variable<phi::DenseTensor>(run_var);   \
    topsatenStatus_t status;                                     \
    {                                                            \
      GCU_AOT_KERNEL_TRACE(abstract_info);                       \
      status = topsaten::topsatenop(                             \
          xout1.value,                                           \
          xout2.value,                                           \
          xout3.value,                                           \
          topsaten_variable<phi::DenseTensor>(x).value,          \
          topsaten_variable<phi::DenseTensor>(scale).value,      \
          topsaten_variable<phi::DenseTensor>(bias).value,       \
          xout4.value,                                           \
          xout5.value,                                           \
          training,                                              \
          momentum,                                              \
          eps,                                                   \
          stream);                                               \
      GcuOpMaybeStreamSync(dev_ctx);                             \
    }                                                            \
    return status;                                               \
  }

#define LAUNCH_TOPSATENOP(topsatenop, dev_ctx, topsatenop_args...)         \
  {                                                                        \
    auto op_info = [&]() -> std::string {                                  \
      return custom_kernel::GetOpInfo(                                     \
          #topsatenop,                                                     \
          ##topsatenop_args,                                               \
          static_cast<topsStream_t>(dev_ctx.stream()));                    \
    };                                                                     \
    VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info(); \
    std::string abstract_info =                                            \
        custom_kernel::GetAbstractInfo(#topsatenop, ##topsatenop_args);    \
    auto status = custom_kernel::launch_##topsatenop(                      \
        dev_ctx, abstract_info, ##topsatenop_args);                        \
    PADDLE_ENFORCE_EQ(                                                     \
        status,                                                            \
        TOPSATEN_STATUS_SUCCESS,                                           \
        phi::errors::Fatal(                                                \
            "Failed to call aten op, get error: %d, details: %s",          \
            status,                                                        \
            op_info().c_str()));                                           \
    VLOG(6) << "Launch tops aten op successfully, details:" << op_info();  \
  }

#define LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(                                   \
    topsatenop, dev_ctx, abstract_info, ...)                                   \
  {                                                                            \
    VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << abstract_info; \
    GCU_AOT_KERNEL_TRACE(abstract_info);                                       \
    auto status = topsaten::topsatenop(                                        \
        __VA_ARGS__, static_cast<topsStream_t>(dev_ctx.stream()));             \
    PADDLE_ENFORCE_EQ(                                                         \
        status,                                                                \
        TOPSATEN_STATUS_SUCCESS,                                               \
        phi::errors::Fatal("Failed to call aten op, get error: %d.", status)); \
    VLOG(6) << "Launch tops aten op successfully, details:" << abstract_info;  \
    GcuOpMaybeStreamSync(dev_ctx);                                             \
  }
}  // namespace

// binary op
DEFINE_LAUNCH_TOPSATENOP(topsatenAdd)
DEFINE_LAUNCH_TOPSATENOP(topsatenSub)
DEFINE_LAUNCH_TOPSATENOP(topsatenMul)
DEFINE_LAUNCH_TOPSATENOP(topsatenDiv)
DEFINE_LAUNCH_TOPSATENOP(topsatenMatmul)
DEFINE_LAUNCH_TOPSATENOP(topsatenRemainder)
DEFINE_LAUNCH_TOPSATENOP(topsatenAtan2)

// binary compare op
DEFINE_LAUNCH_TOPSATENOP(topsatenGt)
DEFINE_LAUNCH_TOPSATENOP(topsatenGe)
DEFINE_LAUNCH_TOPSATENOP(topsatenNe)
DEFINE_LAUNCH_TOPSATENOP(topsatenEq)
DEFINE_LAUNCH_TOPSATENOP(topsatenLe)
DEFINE_LAUNCH_TOPSATENOP(topsatenLt)

// logical op
DEFINE_LAUNCH_TOPSATENOP(topsatenLogicalAnd)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogicalNot)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogicalOr)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogicalXor)

// bitwise op
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseAnd)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseNot)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseOr)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseXor)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseLeftShift)
DEFINE_LAUNCH_TOPSATENOP(topsatenBitwiseRightShift)
DEFINE_LAUNCH_TOPSATENOP(topsatenLShift)
DEFINE_LAUNCH_TOPSATENOP(topsatenRShift)

// unary op
DEFINE_LAUNCH_TOPSATENOP(topsatenCos)
DEFINE_LAUNCH_TOPSATENOP(topsatenRsqrt)
DEFINE_LAUNCH_TOPSATENOP(topsatenSin)
DEFINE_LAUNCH_TOPSATENOP(topsatenLog)
DEFINE_LAUNCH_TOPSATENOP(topsatenLog2)
DEFINE_LAUNCH_TOPSATENOP(topsatenLog10)
DEFINE_LAUNCH_TOPSATENOP(topsatenLog1p)
DEFINE_LAUNCH_TOPSATENOP(topsatenSquare)
DEFINE_LAUNCH_TOPSATENOP(topsatenSqrt)
DEFINE_LAUNCH_TOPSATENOP(topsatenTanh)
DEFINE_LAUNCH_TOPSATENOP(topsatenPow)
DEFINE_LAUNCH_TOPSATENOP(topsatenCopy)
DEFINE_LAUNCH_TOPSATENOP(topsatenExpand)
DEFINE_LAUNCH_TOPSATENOP(topsatenTrunc)
DEFINE_LAUNCH_TOPSATENOP(topsatenSign)

// activation op
DEFINE_LAUNCH_TOPSATENOP(topsatenAbs)
DEFINE_LAUNCH_TOPSATENOP(topsatenAtan)
DEFINE_LAUNCH_TOPSATENOP(topsatenExp)
DEFINE_LAUNCH_TOPSATENOP(topsatenFloor)
DEFINE_LAUNCH_TOPSATENOP(topsatenCeil)
DEFINE_LAUNCH_TOPSATENOP(topsatenGelu)
DEFINE_LAUNCH_TOPSATENOP(topsatenLeakyRelu)
DEFINE_LAUNCH_TOPSATENOP(topsatenRelu)
DEFINE_LAUNCH_TOPSATENOP(topsatenRelu6)
DEFINE_LAUNCH_TOPSATENOP(topsatenSilu)
// DEFINE_LAUNCH_TOPSATENOP(topsatenSwish) same with Silu
DEFINE_LAUNCH_TOPSATENOP(topsatenSigmoid)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogSigmoid)
DEFINE_LAUNCH_TOPSATENOP(topsatenHardswish)
DEFINE_LAUNCH_TOPSATENOP(topsatenHardsigmoid)
DEFINE_LAUNCH_TOPSATENOP(topsatenReciprocal)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogit)
DEFINE_LAUNCH_TOPSATENOP(topsatenCelu)
DEFINE_LAUNCH_TOPSATENOP(topsatenHardshrink)
DEFINE_LAUNCH_TOPSATENOP(topsatenSoftshrink)
DEFINE_LAUNCH_TOPSATENOP(topsatenSoftplus)
DEFINE_LAUNCH_TOPSATENOP(topsatenAcos)
DEFINE_LAUNCH_TOPSATENOP(topsatenAcosh)
DEFINE_LAUNCH_TOPSATENOP(topsatenAsin)
DEFINE_LAUNCH_TOPSATENOP(topsatenAsinh)
DEFINE_LAUNCH_TOPSATENOP(topsatenAtanh)
DEFINE_LAUNCH_TOPSATENOP(topsatenCosh)
DEFINE_LAUNCH_TOPSATENOP(topsatenElu)
DEFINE_LAUNCH_TOPSATENOP(topsatenRound)
DEFINE_LAUNCH_TOPSATENOP(topsatenSinh)
DEFINE_LAUNCH_TOPSATENOP(topsatenTan)
DEFINE_LAUNCH_TOPSATENOP(topsatenErf)
DEFINE_LAUNCH_TOPSATENOP(topsatenExpm1)
DEFINE_LAUNCH_TOPSATENOP(topsatenHardtanh)

// reduce op
DEFINE_LAUNCH_TOPSATENOP(topsatenAny)
DEFINE_LAUNCH_TOPSATENOP(topsatenMean)
DEFINE_LAUNCH_TOPSATENOP(topsatenSum)
DEFINE_LAUNCH_TOPSATENOP(topsatenMax)
DEFINE_LAUNCH_TOPSATENOP(topsatenMin)
DEFINE_LAUNCH_TOPSATENOP(topsatenAmax)
DEFINE_LAUNCH_TOPSATENOP(topsatenAmin)
DEFINE_LAUNCH_TOPSATENOP(topsatenProd)
DEFINE_LAUNCH_TOPSATENOP(topsatenCumsum)
DEFINE_LAUNCH_TOPSATENOP(topsatenCumprod)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogsumexp)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogcumsumexp)
DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenCummax)
DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenCummin)

// DEFINE_LAUNCH_TOPSATENOP(topsatenExpand)
DEFINE_LAUNCH_TOPSATENOP(topsatenFull)
DEFINE_LAUNCH_TOPSATENOP(topsatenTril)
DEFINE_LAUNCH_TOPSATENOP(topsatenTriu)
DEFINE_LAUNCH_TOPSATENOP(topsatenWhere)
DEFINE_LAUNCH_TOPSATENOP(topsatenEmbedding)
DEFINE_LAUNCH_TOPSATENOP(topsatenSoftmaxForward)
DEFINE_LAUNCH_TOPSATENOP(topsatenLogSoftmaxForward)
// DEFINE_LAUNCH_TOPSATENOP(topsatenTo)
// DEFINE_LAUNCH_TOPSATENOP(topsatenStack)
DEFINE_LAUNCH_TOPSATENOP(topsatenIsInf)
DEFINE_LAUNCH_TOPSATENOP(topsatenIsNan)
DEFINE_LAUNCH_TOPSATENOP(topsatenIsfinite)
DEFINE_LAUNCH_TOPSATENOP(topsatenIsclose)
DEFINE_LAUNCH_TOPSATENOP(topsatenClip)
DEFINE_LAUNCH_TOPSATENOP(topsatenFlip)
DEFINE_LAUNCH_TOPSATENOP(topsatenOneHot)

DEFINE_LAUNCH_TOPSATENOP(topsatenArgSort)
DEFINE_LAUNCH_TOPSATENOP(topsatenArgmax)
DEFINE_LAUNCH_TOPSATENOP(topsatenArgmin)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndex)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndexAdd)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndexSelect)
DEFINE_LAUNCH_TOPSATENOP(topsatenMaskedSelect)
DEFINE_LAUNCH_TOPSATENOP(topsatenScatter)
DEFINE_LAUNCH_TOPSATENOP(topsatenGather)
DEFINE_LAUNCH_TOPSATENOP(topsatenIndexPut)
DEFINE_LAUNCH_TOPSATENOP(topsatenMinimum)
DEFINE_LAUNCH_TOPSATENOP(topsatenMaximum)
DEFINE_LAUNCH_TOPSATENOP(topsatenFmax)
DEFINE_LAUNCH_TOPSATENOP(topsatenFmin)
DEFINE_LAUNCH_TOPSATENOP(topsatenCopysign)
DEFINE_LAUNCH_TOPSATENOP(topsatenCross)
DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenSort)

DEFINE_LAUNCH_TOPSATENOP(topsatenOnes)
DEFINE_LAUNCH_TOPSATENOP(topsatenZeros)
DEFINE_LAUNCH_TOPSATENOP(topsatenArange)
DEFINE_LAUNCH_TOPSATENOP(topsatenTile)
DEFINE_LAUNCH_TOPSATENOP(topsatenUpsampleNearest2d)
DEFINE_LAUNCH_TOPSATENOP(topsatenUpsampleBilinear2d)
DEFINE_LAUNCH_TOPSATENOP(topsatenGridSampler)
DEFINE_LAUNCH_TOPSATENOP(topsatenLinear)
DEFINE_LAUNCH_TOPSATENOP(topsatenLerp)
DEFINE_LAUNCH_TOPSATENOP(topsatenRoll)
DEFINE_LAUNCH_TOPSATENOP(topsatenMeshgrid)

// distributions
DEFINE_LAUNCH_TOPSATENOP(topsatenRngUniform)
DEFINE_LAUNCH_TOPSATENOP(topsatenNormal)
DEFINE_LAUNCH_TOPSATENOP(topsatenMultinomial)

DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenTopk)
DEFINE_LAUNCH_TOPSATENOP(topsatenDropout)
DEFINE_LAUNCH_TOPSATENOP_OUT2(topsatenNativeDropout)

// cnn
DEFINE_LAUNCH_TOPSATENOP(topsatenConv2d)
DEFINE_LAUNCH_TOPSATENOP(topsatenConvTranspose2d)
DEFINE_LAUNCH_TOPSATENOP(topsatenConvDepthwise2d)
DEFINE_LAUNCH_TOPSATENOP(topsatenConvolution)
DEFINE_LAUNCH_TOPSATENOP(topsatenConvBiasActivation)
DEFINE_LAUNCH_TOPSATENOP_PADDLE(topsatenConvScaledBiasActivation)

// loss
DEFINE_LAUNCH_TOPSATENOP(topsatenHuberLoss)

// pool
DEFINE_LAUNCH_TOPSATENOP(topsatenAvgPool2d)
DEFINE_LAUNCH_TOPSATENOP(topsatenMaxPool2d)

// norm
DEFINE_LAUNCH_TOPSATENOP(topsatenLayerNorm)
DEFINE_LAUNCH_TOPSATENOP(topsatenInstanceNorm)
DEFINE_LAUNCH_TOPSATENOP_OUT3(topsatenNativeLayerNorm)
DEFINE_LAUNCH_TOPSATENOP_NATIVE_BATCH_NORM(topsatenNativeBatchNorm)

// linalg
DEFINE_LAUNCH_TOPSATENOP(topsatenCholesky)

// lstm
DEFINE_LAUNCH_TOPSATENOP(topsatenRnnTanhCell)
DEFINE_LAUNCH_TOPSATENOP(topsatenRnnReluCell)
DEFINE_LAUNCH_TOPSATENOP_VOUT(topsatenLstmCell)

// domain
DEFINE_LAUNCH_TOPSATENOP_XLA(topsxlaGather)
DEFINE_LAUNCH_TOPSATENOP_XLA(topsxlaScatter)
DEFINE_LAUNCH_TOPSATENOP_PADDLE(topsatenConvTransposeActivation)

// vllm
DEFINE_LAUNCH_TOPSATENOP_OUT2_VLLM(topsvllmRotaryEmbedding)
DEFINE_LAUNCH_TOPSATENOP_OUT2_VLLM(topsvllmFusedAddRmsNorm)
DEFINE_LAUNCH_TOPSATENOP_VLLM(topsvllmMemoryEfficientAttention)
DEFINE_LAUNCH_TOPSATENOP_VLLM(topsvllmRmsNorm)
DEFINE_LAUNCH_TOPSATENOP_VLLM(topsvllmSiluAndMul)

}  // namespace custom_kernel
