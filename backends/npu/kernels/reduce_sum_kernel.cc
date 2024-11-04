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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void AclopSetTensorValueNPUImplKernel(const Context& dev_ctx,
                                      const phi::DenseTensor& x,
                                      const phi::DenseTensor& value,
                                      const phi::IntArray& starts,
                                      const phi::IntArray& ends,
                                      const phi::IntArray& steps,
                                      const std::vector<int64_t>& axes,
                                      const std::vector<int64_t>& decrease_axes,
                                      const std::vector<int64_t>& none_axes,
                                      phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  auto in_dims = x.dims();
  // Unsqueeze and Pad x in the last dim from 1 to 8/16.
  // StridedSliceAssign op only support the last dim of
  // the input greater than 8 for 4 bytes input and 16
  // for 8 bytes input.
  std::vector<int32_t> in_dims_arr = phi::vectorize<int32_t>(x.dims());
  std::vector<int32_t> pad_in_dims_arr = phi::vectorize<int32_t>(x.dims());
  in_dims_arr.push_back(1);
#if (CANN_VERSION_CODE >= 700000)
  if (x.dtype() == phi::DataType::FLOAT16) {
    pad_in_dims_arr.push_back(16);
  } else {
    pad_in_dims_arr.push_back(8);
  }
#else
  pad_in_dims_arr.push_back(8);
#endif

  phi::DenseTensor x_tmp(x);
  x_tmp.Resize(phi::make_ddim(in_dims_arr));

  phi::DenseTensor pad_last_dim_x, pad_last_dim_out;
  pad_last_dim_x.Resize(phi::make_ddim(pad_in_dims_arr));
  dev_ctx.template Alloc<T>(&pad_last_dim_x);
  pad_last_dim_out.Resize(phi::make_ddim(pad_in_dims_arr));
  dev_ctx.template Alloc<T>(&pad_last_dim_out);
  // Broadcast x to pad_last_dims_x
  NpuOpRunner runner_brd;
  runner_brd.SetType("BroadcastTo")
      .AddInput(x_tmp)
      .AddInput(dev_ctx, std::move(pad_in_dims_arr))
      .AddOutput(pad_last_dim_x)
      .Run(stream);

  auto pad_in_dims = pad_last_dim_x.dims();

  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  custom_kernel::CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);
  auto slice_dims = custom_kernel::GetSliceDims(
      in_dims, axes, starts_local, ends_local, &steps_local);
  auto decrease_slice_dims =
      custom_kernel::GetDecreasedDims(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;
  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;

    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    for (int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      if (decrease_axes_cur < decrease_axes.size() &&
          decrease_axes[decrease_axes_cur] == i) {
        decrease_axes_cur++;
      } else {
        slice_dims_with_none.push_back(slice_dims[i]);
      }
    }
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }

    slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
  }

  TensorCopy(dev_ctx, pad_last_dim_x, false, &pad_last_dim_out);

  auto starts_indices = std::vector<int64_t>(in_dims.size(), 0);
  auto ends_indices = std::vector<int64_t>(in_dims.size(), 0);
  auto strides_indices = std::vector<int64_t>(in_dims.size(), 0);
  std::vector<int> flip_axis;

  for (int i = 0; i < in_dims.size(); ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = slice_dims[i];
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = starts_local[i];
    ends_indices[axis_index] = ends_local[i];
    strides_indices[axis_index] = steps_local[i];
  }

  // Because StridedSliceAssign does not support the case
  // of stride < 0 temporarily, the coordinates of
  // starts_indices, ends_indices and strides_indices
  // need to be converted.
  bool need_flip = false;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    if (strides_indices[i] < 0) {
      if (!need_flip) {
        need_flip = true;
      }
      flip_axis.push_back(i);
      strides_indices[i] = strides_indices[i] * (-1);
      ends_indices[i] = starts_indices[i] + 1;
      starts_indices[i] =
          starts_indices[i] - (slice_dims[i] - 1) * strides_indices[i];
    }
  }
  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value.dims() ||
      slice_dims_for_assign.size() == 0) {
    value_temp = value;
  } else {
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(value)
        .AddInput(dev_ctx, phi::vectorize(slice_dims_for_assign))
        .AddOutput(value_temp)
        .Run(stream);
  }

  phi::DenseTensor reverse_value;
  if (need_flip) {
    reverse_value.Resize(value_temp.dims());
    dev_ctx.template Alloc<T>(&reverse_value);
    NpuOpRunner reverse_runner;
    reverse_runner.SetType("ReverseV2")
        .AddInput(value_temp)
        .AddInput(dev_ctx, std::move(flip_axis))
        .AddOutput(reverse_value);
    reverse_runner.Run(stream);
  } else {
    reverse_value = value_temp;
  }

  // Add last dim for index
  starts_indices.push_back(0);
#if (CANN_VERSION_CODE >= 700000)
  if (x.dtype() == phi::DataType::FLOAT16) {
    ends_indices.push_back(16);
  } else {
    ends_indices.push_back(8);
  }
#else
  ends_indices.push_back(8);
#endif
  strides_indices.push_back(1);

  // Broadcast value;
  phi::DenseTensor value_brd;
  auto slice_dims_brd = phi::vectorize<int32_t>(slice_dims_for_assign);
  slice_dims_brd.push_back(1);
  reverse_value.Resize(phi::make_ddim(slice_dims_brd));
#if (CANN_VERSION_CODE >= 700000)
  if (x.dtype() == phi::DataType::FLOAT16) {
    slice_dims_brd[slice_dims_brd.size() - 1] = 16;
  } else {
    slice_dims_brd[slice_dims_brd.size() - 1] = 8;
  }
#else
  slice_dims_brd[slice_dims_brd.size() - 1] = 8;
#endif
  value_brd.Resize(phi::make_ddim(slice_dims_brd));
  dev_ctx.template Alloc<T>(&value_brd);
  NpuOpRunner runner_brd1;
  runner_brd1.SetType("BroadcastTo")
      .AddInput(reverse_value)
      .AddInput(dev_ctx, std::move(slice_dims_brd))
      .AddOutput(value_brd)
      .Run(stream);

  NpuOpRunner strideslicerunner;
  strideslicerunner.SetType("StridedSliceAssign")
      .AddInput(pad_last_dim_x)
      .AddInput(dev_ctx, std::move(starts_indices))
      .AddInput(dev_ctx, std::move(ends_indices))
      .AddInput(dev_ctx, std::move(strides_indices))
      .AddInput(value_brd)
      .AddAttr("begin_mask", 0)
      .AddAttr("end_mask", 0)
      .AddAttr("ellipsis_mask", 0)
      .AddAttr("new_axis_mask", 0)
      .AddAttr("shrink_axis_mask", 0)
      .AddOutput(pad_last_dim_out)
      .Run(stream);

  int32_t axis = static_cast<int32_t>(pad_last_dim_x.dims().size() - 1);
  auto out_dims_arr = phi::vectorize(in_dims);
  out_dims_arr.push_back(1);
  out->Resize(phi::make_ddim(out_dims_arr));
  dev_ctx.template Alloc<T>(out);
  NpuOpRunner gather_runner;
  // StridedSliceAssign op's output shares memory with in-place output.
  // So, here use pad_last_dim_x as the input.
  gather_runner.SetType("GatherV2")
      .AddInput(pad_last_dim_x)
      .AddInput(dev_ctx, std::vector<int32_t>({0}))
      .AddInput(dev_ctx, std::vector<int32_t>({axis}))
      .AddOutput(*out)
      .Run(stream);

  out->Resize(in_dims);
}

template <typename T, typename Context>
void SetTensorValueNPUImplKernel(const Context& dev_ctx,
                                 const phi::DenseTensor& x,
                                 const phi::DenseTensor& value,
                                 const phi::IntArray& starts,
                                 const phi::IntArray& ends,
                                 const phi::IntArray& steps,
                                 const std::vector<int64_t>& axes,
                                 const std::vector<int64_t>& decrease_axes,
                                 const std::vector<int64_t>& none_axes,
                                 phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnStridedSliceAssignV2,
                   (custom_kernel::AclopSetTensorValueNPUImplKernel<T, Context>(
                       dev_ctx,
                       x,
                       value,
                       starts,
                       ends,
                       steps,
                       axes,
                       decrease_axes,
                       none_axes,
                       out)));

  auto stream = dev_ctx.stream();
  auto in_dims = x.dims();
  // Unsqueeze and Pad x in the last dim from 1 to 8/16.
  // StridedSliceAssign op only support the last dim of
  // the input greater than 8 for 4 bytes input and 16
  // for 8 bytes input.
  std::vector<int64_t> in_dims_arr = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> pad_in_dims_arr = phi::vectorize<int64_t>(x.dims());

  in_dims_arr.push_back(1);
#if (CANN_VERSION_CODE >= 700000)
  if (x.dtype() == phi::DataType::FLOAT16) {
    pad_in_dims_arr.push_back(16);
  } else {
    pad_in_dims_arr.push_back(8);
  }
#else
  pad_in_dims_arr.push_back(8);
#endif

  phi::DenseTensor x_tmp(x);
  x_tmp.Resize(phi::make_ddim(in_dims_arr));
  phi::DenseTensor pad_last_dim_x;
  pad_last_dim_x.Resize(phi::make_ddim(pad_in_dims_arr));
  dev_ctx.template Alloc<T>(&pad_last_dim_x);
  EXEC_NPU_CMD(aclnnExpand, dev_ctx, x_tmp, pad_in_dims_arr, pad_last_dim_x);

  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  custom_kernel::CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);
  auto slice_dims = custom_kernel::GetSliceDims(
      in_dims, axes, starts_local, ends_local, &steps_local);
  auto decrease_slice_dims =
      custom_kernel::GetDecreasedDims(slice_dims, decrease_axes);
  auto slice_dims_for_assign = decrease_slice_dims;

  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;

    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    for (int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      if (decrease_axes_cur < decrease_axes.size() &&
          decrease_axes[decrease_axes_cur] == i) {
        decrease_axes_cur++;
      } else {
        slice_dims_with_none.push_back(slice_dims[i]);
      }
    }
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }

    slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
  }

  auto starts_indices = std::vector<int64_t>(in_dims.size(), 0);
  auto ends_indices = std::vector<int64_t>(in_dims.size(), 0);
  auto strides_indices = std::vector<int64_t>(in_dims.size(), 0);
  std::vector<int64_t> flip_axis;

  for (int i = 0; i < in_dims.size(); ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = slice_dims[i];
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = starts_local[i];
    ends_indices[axis_index] = ends_local[i];
    strides_indices[axis_index] = steps_local[i];
  }

  // Because StridedSliceAssign does not support the case
  // of stride < 0 temporarily, the coordinates of
  // starts_indices, ends_indices and strides_indices
  // need to be converted.
  bool need_flip = false;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    if (strides_indices[i] < 0) {
      if (!need_flip) {
        need_flip = true;
      }
      flip_axis.push_back(i);
      strides_indices[i] = strides_indices[i] * (-1);
      ends_indices[i] = starts_indices[i] + 1;
      starts_indices[i] =
          starts_indices[i] - (slice_dims[i] - 1) * strides_indices[i];
    }
  }

  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value.dims() ||
      slice_dims_for_assign.size() == 0) {
    value_temp = value;
  } else {
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);

    std::vector<int64_t> slice_dims_int64 =
        phi::vectorize(slice_dims_for_assign);
    EXEC_NPU_CMD(aclnnExpand, dev_ctx, value, slice_dims_int64, value_temp);
  }

  phi::DenseTensor reverse_value;
  if (need_flip) {
    reverse_value.Resize(value_temp.dims());
    dev_ctx.template Alloc<T>(&reverse_value);
    EXEC_NPU_CMD(aclnnFlip, dev_ctx, value_temp, flip_axis, reverse_value);
  } else {
    reverse_value = value_temp;
  }

  // Add last dim for index
  starts_indices.push_back(0);
#if (CANN_VERSION_CODE >= 700000)
  if (x.dtype() == phi::DataType::FLOAT16) {
    ends_indices.push_back(16);
  } else {
    ends_indices.push_back(8);
  }
#else
  ends_indices.push_back(8);
#endif
  strides_indices.push_back(1);

  // Broadcast value;
  phi::DenseTensor value_brd;

  auto slice_dims_brd = phi::vectorize<int64_t>({});
  for (int64_t i = 0; i < starts_indices.size(); ++i) {
    float end_v = static_cast<float>(ends_indices[i]);
    float starts_v = static_cast<float>(starts_indices[i]);
    float strides_v = static_cast<float>(strides_indices[i]);
    int dim_v = std::ceil((end_v - starts_v) / strides_v);

    if (i == 0) {
      slice_dims_brd[i] = dim_v;
    } else {
      slice_dims_brd.push_back(dim_v);
    }
  }

  slice_dims_brd[slice_dims_brd.size() - 1] = 1;
  reverse_value.Resize(phi::make_ddim(slice_dims_brd));

#if (CANN_VERSION_CODE >= 700000)
  if (x.dtype() == phi::DataType::FLOAT16) {
    slice_dims_brd[slice_dims_brd.size() - 1] = 16;
  } else {
    slice_dims_brd[slice_dims_brd.size() - 1] = 8;
  }
#else
  slice_dims_brd[slice_dims_brd.size() - 1] = 8;
#endif

  value_brd.Resize(phi::make_ddim(slice_dims_brd));
  dev_ctx.template Alloc<T>(&value_brd);
  EXEC_NPU_CMD(aclnnExpand, dev_ctx, reverse_value, slice_dims_brd, value_brd);

  std::vector<int64_t> axes_optional(pad_last_dim_x.dims().size());
  for (int64_t i = 0; i < pad_last_dim_x.dims().size(); ++i) {
    axes_optional[i] = i;
  }

  EXEC_NPU_CMD(aclnnStridedSliceAssignV2,
               dev_ctx,
               pad_last_dim_x,
               value_brd,
               starts_indices,
               ends_indices,
               strides_indices,
               axes_optional);

  auto out_dims_arr = phi::vectorize(in_dims);
  out_dims_arr.push_back(1);
  out->Resize(phi::make_ddim(out_dims_arr));
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor zero;
  phi::DenseTensorMeta meta = {phi::DataType::INT64, {1}};
  zero.set_meta(meta);
  dev_ctx.template Alloc<int64_t>(&zero);
  EXEC_NPU_CMD(aclnnInplaceZero, dev_ctx, zero);
  int64_t dim_last = static_cast<int64_t>(pad_last_dim_x.dims().size() - 1);
  EXEC_NPU_CMD(aclnnGatherV2, dev_ctx, pad_last_dim_x, dim_last, zero, *out);
  out->Resize(in_dims);
}

template <typename T, typename Context>
void SetTensorValueNPUKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& value,
                             const phi::IntArray& starts,
                             const phi::IntArray& ends,
                             const phi::IntArray& steps,
                             const std::vector<int64_t>& axes,
                             const std::vector<int64_t>& decrease_axes,
                             const std::vector<int64_t>& none_axes,
                             phi::DenseTensor* out) {
  phi::DenseTensor tmp_x, tmp_value, tmp_out;
  phi::DenseTensorMeta tmp_x_meta, tmp_value_meta, tmp_out_meta;
  // StridedSliceAssign only support fp32 and int32.
  // We directly transform fp64/int64 to fp32/int32 before impl function.
  auto stream = dev_ctx.stream();
  if (x.dtype() == phi::DataType::FLOAT64) {
    tmp_x_meta = {phi::DataType::FLOAT32, x.dims()};
    tmp_value_meta = {phi::DataType::FLOAT32, value.dims()};
    tmp_out_meta = {phi::DataType::FLOAT32, out->dims()};
    tmp_x.set_meta(tmp_x_meta);
    tmp_value.set_meta(tmp_value_meta);
    tmp_out.set_meta(tmp_out_meta);
    dev_ctx.template Alloc<float>(&tmp_x);
    dev_ctx.template Alloc<float>(&tmp_value);
    dev_ctx.template Alloc<float>(&tmp_out);
    const auto& runner1 =
        NpuOpRunner("Cast", {x}, {tmp_x}, {{"dst_type", ACL_FLOAT}});
    runner1.Run(stream);
    const auto& runner2 =
        NpuOpRunner("Cast", {value}, {tmp_value}, {{"dst_type", ACL_FLOAT}});
    runner2.Run(stream);
    SetTensorValueNPUImplKernel<float, Context>(dev_ctx,
                                                tmp_x,
                                                tmp_value,
                                                starts,
                                                ends,
                                                steps,
                                                axes,
                                                decrease_axes,
                                                none_axes,
                                                &tmp_out);
    out->Resize(tmp_out.dims());
    dev_ctx.template Alloc<T>(out);
    const auto& runner3 =
        NpuOpRunner("Cast", {tmp_out}, {*out}, {{"dst_type", ACL_DOUBLE}});
    runner3.Run(stream);
  } else if (x.dtype() == phi::DataType::INT64) {
    tmp_x_meta = {phi::DataType::INT32, x.dims()};
    tmp_value_meta = {phi::DataType::INT32, value.dims()};
    tmp_out_meta = {phi::DataType::INT32, out->dims()};
    tmp_x.set_meta(tmp_x_meta);
    tmp_value.set_meta(tmp_value_meta);
    tmp_out.set_meta(tmp_out_meta);
    dev_ctx.template Alloc<int32_t>(&tmp_x);
    dev_ctx.template Alloc<int32_t>(&tmp_value);
    dev_ctx.template Alloc<int32_t>(&tmp_out);
    const auto& runner1 =
        NpuOpRunner("Cast", {x}, {tmp_x}, {{"dst_type", ACL_INT32}});
    runner1.Run(stream);
    const auto& runner2 =
        NpuOpRunner("Cast", {value}, {tmp_value}, {{"dst_type", ACL_INT32}});
    runner2.Run(stream);
    SetTensorValueNPUImplKernel<int32_t, Context>(dev_ctx,
                                                  tmp_x,
                                                  tmp_value,
                                                  starts,
                                                  ends,
                                                  steps,
                                                  axes,
                                                  decrease_axes,
                                                  none_axes,
                                                  &tmp_out);
    out->Resize(tmp_out.dims());
    dev_ctx.template Alloc<T>(out);
    const auto& runner3 =
        NpuOpRunner("Cast", {tmp_out}, {*out}, {{"dst_type", ACL_INT64}});
    runner3.Run(stream);
  } else {
    SetTensorValueNPUImplKernel<T, Context>(dev_ctx,
                                            x,
                                            value,
                                            starts,
                                            ends,
                                            steps,
                                            axes,
                                            decrease_axes,
                                            none_axes,
                                            out);
  }
}

template <typename T, typename Context>
void SetValueNPUKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::IntArray& starts,
                       const phi::IntArray& ends,
                       const phi::IntArray& steps,
                       const std::vector<int64_t>& axes,
                       const std::vector<int64_t>& decrease_axes,
                       const std::vector<int64_t>& none_axes,
                       const std::vector<int64_t>& shape,
                       const std::vector<phi::Scalar>& values,
                       phi::DenseTensor* out) {
  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  phi::DenseTensor value_tensor;
  value_tensor.Resize(phi::make_ddim(shape));
  custom_kernel::TensorFromVector(
      dev_ctx, assgin_values, dev_ctx, &value_tensor);
  value_tensor.Resize(phi::make_ddim(shape));

  custom_kernel::SetTensorValueNPUKernel<T, Context>(dev_ctx,
                                                     x,
                                                     value_tensor,
                                                     starts,
                                                     ends,
                                                     steps,
                                                     axes,
                                                     decrease_axes,
                                                     none_axes,
                                                     out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(set_value,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SetValueNPUKernel,
                          float,
#if (CANN_VERSION_CODE >= 700000)
                          phi::dtype::float16,
#else
#endif
                          double,
                          int,
                          int64_t) {
}

PD_REGISTER_PLUGIN_KERNEL(set_value_with_tensor,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SetTensorValueNPUKernel,
                          float,
#if (CANN_VERSION_CODE >= 700000)
                          phi::dtype::float16,
#else
#endif
                          double,
                          int,
                          int64_t) {
}
