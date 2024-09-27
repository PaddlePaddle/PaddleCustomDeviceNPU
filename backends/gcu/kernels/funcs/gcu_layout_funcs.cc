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

#include "kernels/funcs/gcu_layout_funcs.h"

#include "common/gcu_env_list.h"
#include "kernels/funcs/common_ops.h"

namespace custom_kernel {
namespace {
inline bool IsValidPermutation(const std::vector<int64_t>& permutation) {
  auto size = permutation.size();
  std::vector<bool> flags(size, false);
  for (size_t i = 0; i < size; ++i) {
    auto k = permutation[i];
    if (k >= 0 && k < size && !flags[k])
      flags[k] = true;
    else
      return false;
  }
  return true;
}

inline std::vector<int64_t> ReorderVector(
    const std::vector<int64_t>& src, const std::vector<int64_t>& permutation) {
  PADDLE_ENFORCE(
      permutation.size() == src.size() && IsValidPermutation(permutation),
      phi::errors::InvalidArgument("Invalid permutation."));

  std::vector<int64_t> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = src[permutation[i]];
  }
  return dst;
}
}  // namespace

bool EnableTransposeOptimize() {
  static const char* enable_opt_env = std::getenv(env::kEnableTransOpt);
  static bool enable_opt =
      (enable_opt_env != nullptr && std::string(enable_opt_env) == "true");
  // just for log
  static bool enable_trans_opt = ((VLOG(0) << "Enable transpose optimize:"
                                           << (enable_opt ? "true" : "false")),
                                  (enable_opt));
  if (enable_trans_opt) {
    return true;
  }
  return false;
}

void SetLayout(phi::DenseTensor& tensor,  // NOLINT
               const common::DataLayout& layout) {
  auto meta = tensor.meta();
  meta.layout = layout;
  tensor.set_meta(meta);
}

void Transpose(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& axis,
               phi::DenseTensor* out) {
  auto x_perm = x;
  PermutedShapeAndStrides(x_perm, axis);
  LAUNCH_TOPSATENOP(topsatenCopy, dev_ctx, *out, x_perm, false);
}

phi::DenseTensor Transpose(const phi::CustomContext& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int64_t>& axis) {
  // infer dst shape
  std::vector<int64_t> src_dims = phi::vectorize(x.dims());
  std::vector<int64_t> dst_dims = ReorderVector(src_dims, axis);

  phi::DenseTensor dst_tensor;
  phi::DenseTensorMeta meta(x.dtype(), phi::make_ddim(dst_dims));
  dst_tensor.set_meta(meta);
  dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
  Transpose(dev_ctx, x, axis, &dst_tensor);
  return dst_tensor;
}

//
// ----------------------------------------------------------------------------
//    format     shape  strides  contiguous  layout           notes
// ----------------------------------------------------------------------------
//     NCHW      NCHW    NCHW       Y         kNCHW          normal
// PdOriginNHWC  NHWC    NHWC       Y         kNCHW          normal
// PdCustomNHWC  NCHW    NCHW       Y         kNHWC   Transfer between kernels
//    AtenNHWC   NCHW  NHWC_perm    N         kNHWC      Aten kernel in/out
// ----------------------------------------------------------------------------
//
// Notes:
// 1. NCHW and PdOriginNHWC are normal expressions;
// 2. Use NCHW or PdCustomNHWC to transfer tensors between kernels;
// 3. AtenNHWC is only used for input and output of aten kernel, and is
//    expressed as PdCustomNHWC before being passed to paddle kernel;
// 4. The shape of AtenNHWC is expressed as NCHW, and strides are calculated and
//    permuted from the real NHWC shape;
// 5. Only the layout of PdCustomNHWC and AtenNHWC is expressed as kNHWC, Note
//    that the layout of PdOriginNHWC uses the default value kNCHW.
//
bool DataPdCustomNHWC(const phi::DenseTensor& tensor) {
  return (EnableTransposeOptimize() &&
          tensor.layout() == common::DataLayout::kNHWC);
}

bool DataPdCustomNHWC(const std::vector<phi::DenseTensor>& tensors) {
  return (EnableTransposeOptimize() &&
          std::any_of(tensors.begin(),
                      tensors.end(),
                      [](const phi::DenseTensor& tensor) {
                        return tensor.layout() == common::DataLayout::kNHWC;
                      }));
}

// ////////////////  Permuted funcs ////////////////
void PermutedShapeWithcontiguousStrides(phi::DenseTensor& tensor,  // NOLINT
                                        const std::vector<int64_t>& permutation,
                                        const common::DataLayout& layout) {
  auto meta = tensor.meta();
  std::vector<int64_t> dst_dims(meta.dims.size());
  for (size_t i = 0; i < meta.dims.size(); ++i) {
    dst_dims[i] = meta.dims[permutation[i]];
  }
  meta.dims = common::make_ddim(dst_dims);
  meta.strides = meta.calc_strides(meta.dims);
  meta.layout = layout;
  tensor.set_meta(meta);
}

void RecoverPdCustomNHWCMeta(phi::DenseTensor& tensor) {  // NOLINT
  PermutedShapeWithcontiguousStrides(
      tensor, layout_trans::kNCHW_to_NHWC, common::DataLayout::kNCHW);
}

void PermutedStridesWithoutShape(phi::DenseTensor& tensor,  // NOLINT
                                 const std::vector<int64_t>& shape_perm,
                                 const std::vector<int64_t>& strides_perm,
                                 const common::DataLayout& layout) {
  auto meta = tensor.meta();
  std::vector<int64_t> dst_dims(meta.dims.size());
  for (size_t i = 0; i < meta.dims.size(); ++i) {
    dst_dims[i] = meta.dims[shape_perm[i]];
  }
  meta.strides = meta.calc_strides(common::make_ddim(dst_dims));

  std::vector<int64_t> dst_strides(meta.strides.size());
  for (size_t i = 0; i < meta.strides.size(); ++i) {
    dst_strides[i] = meta.strides[strides_perm[i]];
  }
  meta.strides = common::make_ddim(dst_strides);
  meta.layout = layout;
  tensor.set_meta(meta);
}

void PermutedShapeAndStrides(phi::DenseTensor& tensor,  // NOLINT
                             const std::vector<int64_t>& permutation,
                             const common::DataLayout& layout) {
  auto meta = tensor.meta();
  meta.strides = meta.calc_strides(meta.dims);

  std::vector<int64_t> dst_dims(meta.dims.size());
  std::vector<int64_t> dst_strides(meta.strides.size());
  for (size_t i = 0; i < meta.dims.size(); ++i) {
    dst_dims[i] = meta.dims[permutation[i]];
    dst_strides[i] = meta.strides[permutation[i]];
  }
  meta.dims = common::make_ddim(dst_dims);
  meta.strides = common::make_ddim(dst_strides);
  meta.layout = layout;
  tensor.set_meta(meta);
}

// ////////////////  Transpose funcs ////////////////
phi::DenseTensor NCHWTransToPdOriginNHWC(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x) {
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNCHW,
      phi::errors::InvalidArgument("Layout of x should be origin NCHW."));
  auto out = custom_kernel::Transpose(dev_ctx, x, layout_trans::kNCHW_to_NHWC);
  return out;  // shape is NHWC, strides is NHWC, contiguous
}

phi::DenseTensor NCHWTransToPdCustomNHWC(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x) {
  auto out = NCHWTransToPdOriginNHWC(dev_ctx, x);
  auto meta = x.meta();
  meta.layout = common::DataLayout::kNHWC;
  out.set_meta(meta);  // shape is NCHW, strides is NCHW, contiguous
  return out;
}

phi::DenseTensor NCHWTransToAtenNHWC(const phi::CustomContext& dev_ctx,
                                     const phi::DenseTensor& x) {
  auto out = NCHWTransToPdCustomNHWC(dev_ctx, x);
  PdCustomNHWCRepresentAsAtenNHWC(out);
  return out;
}

phi::DenseTensor PdCustomNHWCTransToNCHW(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x) {
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNHWC,
      phi::errors::InvalidArgument("Layout of x should be PdCustomNHWC."));
  phi::DenseTensor tensor = x;  // shape is NCHW, strides is NCHW, contiguous
  RecoverPdCustomNHWCMeta(tensor);
  tensor =
      custom_kernel::Transpose(dev_ctx, tensor, layout_trans::kNHWC_to_NCHW);
  return tensor;  // shape is NCHW, strides is NCHW, contiguous
}

phi::DenseTensor PdOriginNHWCTransToNCHW(const phi::CustomContext& dev_ctx,
                                         const phi::DenseTensor& x) {
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNCHW,
      phi::errors::InvalidArgument("Layout of x should be origin NCHW."));
  auto out = custom_kernel::Transpose(dev_ctx, x, layout_trans::kNHWC_to_NCHW);
  return out;  // shape is NCHW, strides is NCHW, contiguous
}

// ////////////////  Represent funcs ////////////////
phi::DenseTensor NoNeedTransNCHWRepresentAsOriginNHWC(
    const phi::DenseTensor& x) {
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNCHW,
      phi::errors::InvalidArgument("Layout of x should be origin NHWC."));
  phi::DenseTensor tensor = x;
  RecoverPdCustomNHWCMeta(tensor);
  return tensor;
}

void PdCustomNHWCRepresentAsAtenNHWC(phi::DenseTensor& x,  // NOLINT
                                     bool weight_or_output) {
  if (!weight_or_output) {
    PADDLE_ENFORCE_EQ(
        x.layout(),
        common::DataLayout::kNHWC,
        phi::errors::InvalidArgument("Layout of x should be PdCustomNHWC."));
  }
  // input x shape is NCHW, strides is NCHW, contiguous
  // output x shape is NCHW, strides is NHWC_perm, NOT contiguous
  PermutedStridesWithoutShape(x,
                              layout_trans::kNCHW_to_NHWC,
                              layout_trans::kNHWC_to_NCHW,
                              common::DataLayout::kNHWC);
}

void AtenNHWCRepresentAsPdCustomNHWC(phi::DenseTensor& x,  // NOLINT
                                     bool raw_output) {
  if (!raw_output) {
    PADDLE_ENFORCE_EQ(
        x.layout(),
        common::DataLayout::kNHWC,
        phi::errors::InvalidArgument("Layout of x should be AtenNHWC."));
  }
  // input x shape is NCHW, strides is NHWC_perm, NOT contiguous
  // output x shape is NCHW, strides is NCHW, contiguous
  x.Resize(x.dims());  // calc contiguous strides
  auto meta = x.meta();
  meta.layout = common::DataLayout::kNHWC;
  x.set_meta(meta);
}

void OriginNHWCRepresentAsAtenNHWC(phi::DenseTensor& x) {  // NOLINT
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNCHW,
      phi::errors::InvalidArgument("Layout of x should be origin NHWC."));
  // input x shape is NHWC, strides is NHWC, contiguous
  // output x shape is NCHW, strides is NHWC_perm, NOT contiguous
  PermutedShapeAndStrides(
      x, layout_trans::kNHWC_to_NCHW, common::DataLayout::kNHWC);
}

void AtenNHWCRepresentAsOriginNHWC(phi::DenseTensor& x) {  // NOLINT
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNHWC,
      phi::errors::InvalidArgument("Layout of x should be AtenNHWC."));
  // input x shape is NCHW, strides is NHWC_perm, NOT contiguous
  // output x shape is NHWC, strides is NHWC, contiguous
  PermutedShapeAndStrides(
      x, layout_trans::kNCHW_to_NHWC, common::DataLayout::kNCHW);
}

void PdCustomNHWCRepresentAsOriginNHWC(phi::DenseTensor& x,  // NOLINT
                                       bool raw_output) {
  if (!raw_output) {
    PADDLE_ENFORCE_EQ(
        x.layout(),
        common::DataLayout::kNHWC,
        phi::errors::InvalidArgument("Layout of x should be PdCustomNHWC."));
  }
  // input x shape is NCHW, strides is NCHW, contiguous
  // output x shape is NHWC, strides is NHWC, contiguous
  RecoverPdCustomNHWCMeta(x);
}

void OriginNHWCRepresentAsPdCustomNHWC(phi::DenseTensor& x) {  // NOLINT
  PADDLE_ENFORCE_EQ(
      x.layout(),
      common::DataLayout::kNCHW,
      phi::errors::InvalidArgument("Layout of x should be origin NHWC."));
  // input x shape is NHWC, strides is NHWC, contiguous
  // output x shape is NCHW, strides is NCHW, contiguous
  PermutedShapeWithcontiguousStrides(
      x, layout_trans::kNHWC_to_NCHW, common::DataLayout::kNHWC);
}

void RepresentPdCustomNHWC(phi::DenseTensor& x) {  // NOLINT
  x.Resize(x.dims());                              // calc contiguous strides
  auto meta = x.meta();
  meta.layout = common::DataLayout::kNHWC;
  x.set_meta(meta);
}

}  // namespace custom_kernel
