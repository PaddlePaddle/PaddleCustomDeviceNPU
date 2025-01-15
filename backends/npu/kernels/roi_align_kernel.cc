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

namespace custom_kernel {

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void AclopRoiAlignKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& boxes,
                         const paddle::optional<phi::DenseTensor>& boxes_num,
                         int pooled_height,
                         int pooled_width,
                         float spatial_scale,
                         int sampling_ratio,
                         bool aligned,
                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto roi_end_mode = 0;
  PADDLE_ENFORCE_EQ(
      aligned,
      false,
      phi::errors::InvalidArgument(
          "ROIAlignNPU only support Aligned attribute equaled to False"));

  NPUAttributeMap attr_boxes = {{"spatial_scale", spatial_scale},
                                {"pooled_height", pooled_height},
                                {"pooled_width", pooled_width},
                                {"sample_num", sampling_ratio},
                                {"roi_end_mode", roi_end_mode}};

  int dtype = static_cast<int>(ConvertToNpuDtype(phi::DataType::FLOAT32));
  NPUAttributeMap attr_cast = {{"dst_type", dtype}};

  auto stream = dev_ctx.stream();
  int boxes_batch_size;
  std::vector<float> roi_batch_id_data((boxes.dims()[0]));
  int batch_size = x.dims()[0];
  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The batch size of rois and the batch size of images "
            " must be the same. But received the batch size of rois is %d, "
            "and the batch size of images is %d",
            boxes_batch_size,
            batch_size));

    std::vector<int> boxes_num_data;
    TensorToVector(dev_ctx, *boxes_num, dev_ctx, &boxes_num_data);
    int start = 0;
    // transfrom boxes_num to roi_batch_id_data
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (int i = start; i < start + boxes_num_data[n]; ++i) {
        roi_batch_id_data[i] = static_cast<float>(n);
      }
      start += boxes_num_data[n];
    }
  } else {
    auto lod = boxes.lod();
    PADDLE_ENFORCE_EQ(
        lod.empty(),
        false,
        phi::errors::InvalidArgument("Input(ROIs) Tensor of ROIAlignOp "
                                     "does not contain LoD information."));
    auto boxes_lod = lod.back();
    boxes_batch_size = boxes_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The boxes_batch_size and imgs "
            "batch_size must be the same. But received boxes_batch_size = %d, "
            "batch_size = %d",
            boxes_batch_size,
            batch_size));
    int boxes_num_with_lod = boxes_lod[boxes_batch_size];
    PADDLE_ENFORCE_EQ(
        boxes.dims()[0],
        boxes_num_with_lod,
        phi::errors::InvalidArgument(
            "The actual number of rois and the number of rois "
            "provided from Input(RoIsLoD) in RoIAlign must be the same."
            " But received actual number of rois is %d, and the number "
            "of rois from RoIsLoD is %d",
            boxes.dims()[0],
            boxes_num_with_lod));
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
  }
  phi::DenseTensor boxes_num_fp;
  TensorFromVector<float>(dev_ctx, roi_batch_id_data, dev_ctx, &boxes_num_fp);
  boxes_num_fp.Resize({boxes.dims()[0], 1});
  phi::DenseTensor boxes_fp(boxes);
  if (x.dtype() != phi::DataType::FLOAT32) {
    // cast boxes dtype to float32
    const auto& runner_c = NpuOpRunner("Cast", {boxes}, {boxes_fp}, attr_cast);
    runner_c.Run(stream);
  }

  // concate to make (N, 5)
  std::vector<phi::DenseTensor> x_list;
  x_list.push_back(boxes_num_fp);
  x_list.push_back(boxes_fp);

  auto axis = 1;
  // output of concate
  phi::DenseTensor boxes_N5;
  phi::DenseTensorMeta boxes_N5_meta = {phi::DataType::FLOAT32,
                                        phi::make_ddim({boxes.dims()[0], 5})};
  boxes_N5.set_meta(boxes_N5_meta);
  dev_ctx.template Alloc<float>(&boxes_N5);

  // attribute of concate
  auto EleNum = 2;
  NPUAttributeMap attr_concat = {{"N", EleNum}, {"concat_dim", axis}};

  NpuOpRunner runner0;
  runner0.SetType("ConcatD")
      .AddInputs(x_list)
      .AddOutput(boxes_N5)
      .AddInputNames({"x0", "x1"})
      .AddAttrs(attr_concat);
  runner0.Run(stream);

  if (x.dtype() == phi::DataType::FLOAT32) {
    const auto& runner =
        NpuOpRunner("ROIAlign", {x, boxes_N5}, {*out}, attr_boxes);
    runner.Run(stream);
  } else {
    // cast x to float32
    phi::DenseTensor x_fp;
    phi::DenseTensorMeta x_meta = {phi::DataType::FLOAT32, x.dims()};
    x_fp.set_meta(x_meta);
    dev_ctx.template Alloc<float>(&x_fp);
    const auto& runner_c1 = NpuOpRunner("Cast", {x}, {x_fp}, attr_cast);
    runner_c1.Run(stream);

    // cast out
    phi::DenseTensor out_fp;
    phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT32, out->dims()};
    out_fp.set_meta(out_meta);
    dev_ctx.template Alloc<float>(&out_fp);
    const auto& runner_c2 = NpuOpRunner("Cast", {*out}, {out_fp}, attr_cast);
    runner_c2.Run(stream);

    const auto& runner =
        NpuOpRunner("ROIAlign", {x_fp, boxes_N5}, {out_fp}, attr_boxes);
    runner.Run(stream);

    // cast output tp given dtype
    int src_dtype = static_cast<int>(ConvertToNpuDtype(out->dtype()));
    const auto& runner_c3 =
        NpuOpRunner("Cast", {out_fp}, {*out}, {{"dst_type", src_dtype}});
    runner_c3.Run(stream);
  }
}

template <typename T, typename Context>
void RoiAlignKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& boxes,
                    const paddle::optional<phi::DenseTensor>& boxes_num,
                    int pooled_height,
                    int pooled_width,
                    float spatial_scale,
                    int sampling_ratio,
                    bool aligned,
                    phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnRoiAlign,
      (custom_kernel::AclopRoiAlignKernel<T, Context>(dev_ctx,
                                                      x,
                                                      boxes,
                                                      boxes_num,
                                                      pooled_height,
                                                      pooled_width,
                                                      spatial_scale,
                                                      sampling_ratio,
                                                      aligned,
                                                      out)));

  PADDLE_ENFORCE_EQ(
      aligned,
      false,
      phi::errors::InvalidArgument(
          "ROIAlignNPU only support Aligned attribute equaled to False"));

  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  int boxes_batch_size;
  std::vector<float> roi_batch_id_data((boxes.dims()[0]));
  int batch_size = x.dims()[0];
  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The batch size of rois and the batch size of images "
            " must be the same. But received the batch size of rois is %d, "
            "and the batch size of images is %d",
            boxes_batch_size,
            batch_size));

    std::vector<int> boxes_num_data;
    TensorToVector(dev_ctx, *boxes_num, dev_ctx, &boxes_num_data);
    int start = 0;
    // transfrom boxes_num to roi_batch_id_data
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (int i = start; i < start + boxes_num_data[n]; ++i) {
        roi_batch_id_data[i] = static_cast<float>(n);
      }
      start += boxes_num_data[n];
    }
  } else {
    auto lod = boxes.lod();
    PADDLE_ENFORCE_EQ(
        lod.empty(),
        false,
        phi::errors::InvalidArgument("Input(ROIs) Tensor of ROIAlignOp "
                                     "does not contain LoD information."));
    auto boxes_lod = lod.back();
    boxes_batch_size = boxes_lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        boxes_batch_size,
        batch_size,
        phi::errors::InvalidArgument(
            "The boxes_batch_size and imgs "
            "batch_size must be the same. But received boxes_batch_size = %d, "
            "batch_size = %d",
            boxes_batch_size,
            batch_size));
    int boxes_num_with_lod = boxes_lod[boxes_batch_size];
    PADDLE_ENFORCE_EQ(
        boxes.dims()[0],
        boxes_num_with_lod,
        phi::errors::InvalidArgument(
            "The actual number of rois and the number of rois "
            "provided from Input(RoIsLoD) in RoIAlign must be the same."
            " But received actual number of rois is %d, and the number "
            "of rois from RoIsLoD is %d",
            boxes.dims()[0],
            boxes_num_with_lod));
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        roi_batch_id_data[i] = n;
      }
    }
  }

  phi::DenseTensor boxes_fp(boxes);
  if (x.dtype() != phi::DataType::FLOAT32) {
    // cast boxes dtype to float32
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, boxes, phi::DataType::FLOAT32, &boxes_fp);
  }

  phi::DenseTensor boxes_num_fp;
  TensorFromVector<float>(dev_ctx, roi_batch_id_data, dev_ctx, &boxes_num_fp);

  phi::DenseTensor boxes_num_int;
  phi::DenseTensorMeta boxes_num_int_meta = {phi::DataType::INT32,
                                             boxes_num_fp.dims()};
  boxes_num_int.set_meta(boxes_num_int_meta);
  custom_kernel::CastKernel<T, Context>(
      dev_ctx, boxes_num_fp, phi::DataType::INT32, &boxes_num_int);

  if (sampling_ratio < 0) {
    sampling_ratio = 0;
  }
  char* roi_end_mode_str = "avg";

  if (x.dtype() == phi::DataType::FLOAT32) {
    EXEC_NPU_CMD(aclnnRoiAlign,
                 dev_ctx,
                 x,
                 boxes_fp,
                 boxes_num_int,
                 roi_end_mode_str,
                 pooled_height,
                 pooled_width,
                 sampling_ratio,
                 spatial_scale,
                 *out);
  } else {
    // cast x to float32
    phi::DenseTensor x_fp;
    phi::DenseTensorMeta x_meta = {phi::DataType::FLOAT32, x.dims()};
    x_fp.set_meta(x_meta);
    custom_kernel::CastKernel<T, Context>(
        dev_ctx, x, phi::DataType::FLOAT32, &x_fp);

    // cast out
    phi::DenseTensor out_fp;
    phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT32, out->dims()};
    out_fp.set_meta(out_meta);
    dev_ctx.template Alloc<float>(&out_fp);
    EXEC_NPU_CMD(aclnnRoiAlign,
                 dev_ctx,
                 x_fp,
                 boxes_fp,
                 boxes_num_int,
                 roi_end_mode_str,
                 pooled_height,
                 pooled_width,
                 sampling_ratio,
                 spatial_scale,
                 out_fp);

    custom_kernel::CastKernel<T, Context>(dev_ctx, out_fp, out->dtype(), out);
  }
}

template <typename T, typename Context>
void RoiAlignGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& boxes,
                        const paddle::optional<phi::DenseTensor>& boxes_num,
                        const phi::DenseTensor& out_grad,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        phi::DenseTensor* dx) {
  auto in_dims = x.dims();
  int rois_num = boxes.dims()[0];
  auto stream = dev_ctx.stream();

  if (!dx) {
    return;
  }
  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_EQ(
      aligned,
      false,
      phi::errors::InvalidArgument(
          "ROIAlignGradNPU only support Aligned attribute equaled to False"));

  //   PADDLE_ENFORCE_EQ(
  //       boxes.dtype(),
  //       phi::DataType::FLOAT32,
  //       phi::errors::InvalidArgument(
  //           "ROIAlignGradNPU only support ROIs type equaled to FP32."));

  int dtype = static_cast<int>(ConvertToNpuDtype(phi::DataType::FLOAT32));
  NPUAttributeMap attr_cast = {{"dst_type", dtype}};

  int boxes_batch_size;
  int batch_size = x.dims()[0];
  std::vector<float> box_batch_id_data((boxes.dims()[0]));

  if (boxes_num) {
    boxes_batch_size = boxes_num->numel();
    std::vector<int> boxes_num_data;
    TensorToVector(dev_ctx, *boxes_num, dev_ctx, &boxes_num_data);
    int start = 0;
    // transfrom boxes_num to box_batch_id_data
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (int i = start; i < start + boxes_num_data[n]; ++i) {
        box_batch_id_data[i] = static_cast<float>(n);
      }
      start += boxes_num_data[n];
    }
  } else {
    auto boxes_lod = boxes.lod().back();
    boxes_batch_size = boxes_lod.size() - 1;
    for (int n = 0; n < boxes_batch_size; ++n) {
      for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
        box_batch_id_data[i] = n;
      }
    }
  }
  phi::DenseTensor boxes_num_fp;
  TensorFromVector<float>(dev_ctx, box_batch_id_data, dev_ctx, &boxes_num_fp);
  boxes_num_fp.Resize({boxes.dims()[0], 1});
  phi::DenseTensor boxes_fp(boxes);
  if (x.dtype() != phi::DataType::FLOAT32) {
    // cast boxes dtype to float32
    const auto& runner_c = NpuOpRunner("Cast", {boxes}, {boxes_fp}, attr_cast);
    runner_c.Run(stream);
  }

  // Cast boxes_num to fp32 tensor
  phi::DenseTensor boxes_N5;
  phi::DenseTensorMeta boxes_N5_meta = {phi::DataType::FLOAT32,
                                        phi::make_ddim({rois_num, 5})};
  boxes_N5.set_meta(boxes_N5_meta);
  dev_ctx.template Alloc<float>(&boxes_N5);

  std::vector<phi::DenseTensor> x_list;
  x_list.push_back(boxes_num_fp);
  x_list.push_back(boxes_fp);
  const auto& runner_concat = NpuOpRunner(
      "ConcatD", {x_list}, {boxes_N5}, {{"N", 2}, {"concat_dim", 1}});
  runner_concat.Run(stream);

  int roi_end_mode = 0;
  if (x.dtype() == phi::DataType::FLOAT32) {
    const auto& runner_roi_align_grad =
        NpuOpRunner("ROIAlignGrad",
                    {out_grad, boxes_N5},
                    {*dx},
                    {{"xdiff_shape", phi::vectorize<int>(in_dims)},
                     {"pooled_width", pooled_width},
                     {"pooled_height", pooled_height},
                     {"spatial_scale", spatial_scale},
                     {"sample_num", sampling_ratio},
                     {"roi_end_mode", roi_end_mode}});
    runner_roi_align_grad.Run(stream);
  } else {
    // cast out_grad
    phi::DenseTensor out_grad_fp;
    phi::DenseTensorMeta out_grad_meta = {phi::DataType::FLOAT32,
                                          out_grad.dims()};
    out_grad_fp.set_meta(out_grad_meta);
    dev_ctx.template Alloc<float>(&out_grad_fp);
    const auto& runner_c1 =
        NpuOpRunner("Cast", {out_grad}, {out_grad_fp}, attr_cast);
    runner_c1.Run(stream);

    // cast output
    phi::DenseTensor out_fp;
    phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT32, dx->dims()};
    out_fp.set_meta(out_meta);
    dev_ctx.template Alloc<float>(&out_fp);
    const auto& runner_c2 = NpuOpRunner("Cast", {*dx}, {out_fp}, attr_cast);
    runner_c2.Run(stream);

    //  Call ascend RoiAlignGrad function
    int roi_end_mode = 0;
    const auto& runner_roi_align_grad =
        NpuOpRunner("ROIAlignGrad",
                    {out_grad_fp, boxes_N5},
                    {out_fp},
                    {{"xdiff_shape", phi::vectorize<int>(in_dims)},
                     {"pooled_width", pooled_width},
                     {"pooled_height", pooled_height},
                     {"spatial_scale", spatial_scale},
                     {"sample_num", sampling_ratio},
                     {"roi_end_mode", roi_end_mode}});
    runner_roi_align_grad.Run(stream);

    // cast output to given dtype
    int src_dtype = static_cast<int>(ConvertToNpuDtype(dx->dtype()));
    const auto& runner_c3 =
        NpuOpRunner("Cast", {out_fp}, {*dx}, {{"dst_type", src_dtype}});
    runner_c3.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    roi_align, npu, ALL_LAYOUT, custom_kernel::RoiAlignKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(roi_align_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignGradKernel,
                          float,
                          double) {}
