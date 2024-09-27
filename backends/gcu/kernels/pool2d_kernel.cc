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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const phi::DDim data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = phi::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(data_dims.size() * 2,
                      paddings->size(),
                      phi::errors::InvalidArgument(
                          "Paddings size %d should be the same or twice as the "
                          "pooling size %d.",
                          paddings->size(),
                          data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T, typename Context>
void Pool2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& in_x,
                  const phi::IntArray& kernel_size,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("pool2d");
  dev_ctx.template Alloc<T>(out);
  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;
  const bool channel_last = data_format == "NHWC";
  auto in_x_dims = in_x.dims();
  auto out_dims = out->dims();
  phi::DDim data_dims;
  phi::DDim out_data_dims;

  phi::DenseTensor in_x_tensor(in_x), out_tensor(*out);
  std::vector<int> ksize_vec(4, 1);
  std::vector<int> strides_vec(4, 1);

  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
    ksize_vec[1] = ksize[0];
    ksize_vec[2] = ksize[1];
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    phi::DenseTensorMeta in_x_meta = {
        in_x_tensor.dtype(), in_x_tensor.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta out_meta = {
        out_tensor.dtype(), out_tensor.dims(), phi::DataLayout::kNHWC};
    in_x_tensor.set_meta(in_x_meta);
    out_tensor.set_meta(out_meta);
  } else {
    data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
    out_data_dims = phi::slice_ddim(out_dims, 2, out_dims.size());
    ksize_vec[2] = ksize[0];
    ksize_vec[3] = ksize[1];
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
  }

  if (data_dims[0] == 1 && data_dims[1] == 1) {
    TensorCopy(dev_ctx, in_x, false, out);
    return;
  }

  UpdatePadding(&paddings,
                global_pooling,
                adaptive,
                padding_algorithm,
                data_dims,
                strides,
                ksize);

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, in_x);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);

    std::vector<int64_t> strides_v = {strides_t.begin(), strides_t.end()};
    std::vector<int64_t> paddings_v = {
        paddings[0], paddings[2], paddings[1], paddings[3]};
    std::vector<int64_t> kernel_size_v = kernel_size.GetData();

    int divisor_override = 1;
    for (const auto& v : kernel_size_v) {
      divisor_override *= v;
    }

    if (DataPdCustomNHWC(in_x)) {
      PdCustomNHWCRepresentAsAtenNHWC(input_x);
      PdCustomNHWCRepresentAsAtenNHWC(output, true);
    }

    if (pooling_type == "avg") {
      int input_rank = in_x.dims().size();
      int kernel_rank = kernel_size_v.size();
      bool is_exe_reduce_mean = true;
      // kernel = 1 && strides = 1 && padding = 0
      if (divisor_override == 1) {
        for (int i = 0; i < input_rank - kernel_rank; ++i) {
          if (strides_v.at(i) != 1 || paddings_v.at(i) != 0) {
            is_exe_reduce_mean = false;
            break;
          }
        }
      }
      // output = 1
      if (is_exe_reduce_mean) {
        for (int i = input_rank - kernel_rank; i < input_rank; ++i) {
          if (output.dims().at(i) != 1) {
            is_exe_reduce_mean = false;
            break;
          }
        }
      }

      // A special adaptive_avg_pool2d, indicated by the parameter adaptive
      if (is_exe_reduce_mean) {
        std::vector<int64_t> dims;
        for (int i = input_rank - kernel_rank; i < input_rank; ++i) {
          dims.push_back(i);
        }
        LAUNCH_TOPSATENOP(
            topsatenMean, dev_ctx, output, input_x, dims, true, out->dtype());
      } else {
        auto divisor_override_none =
            topsatenScalar_t({TOPSATEN_DATA_NONE, {.ival = 0}});
        LAUNCH_TOPSATENOP(topsatenAvgPool2d,
                          dev_ctx,
                          output,
                          input_x,
                          kernel_size_v,
                          strides_v,
                          paddings_v,
                          ceil_mode,
                          !exclusive,
                          divisor_override_none);
      }
    } else if (pooling_type == "max") {
      std::vector<int64_t> dilation = {1};
      LAUNCH_TOPSATENOP(topsatenMaxPool2d,
                        dev_ctx,
                        output,
                        input_x,
                        kernel_size_v,
                        strides_v,
                        paddings_v,
                        dilation,
                        ceil_mode);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupport pooling_type string: %s.", pooling_type));
    }

    if (DataPdCustomNHWC(in_x)) {
      AtenNHWCRepresentAsPdCustomNHWC(output);
      AtenNHWCRepresentAsPdCustomNHWC(*out, true);
    }
    MaybeTransResult(dev_ctx, output, out);
  } else {  // kernel impl base on JIT
    std::vector<int> ksize(kernel_size.GetData().begin(),
                           kernel_size.GetData().end());

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&in_x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["adaptive"] = adaptive;
    attrs["global_pooling"] = global_pooling;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["pooling_type"] = pooling_type;
    attrs["data_format"] = data_format;
    attrs["ksize"] = ksize;
    attrs["strides"] = strides_t;
    attrs["paddings"] = paddings_t;
    attrs["ceil_mode"] = ceil_mode;
    attrs["exclusive"] = exclusive;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "pool2d", dev_ctx);
  }
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& in_x,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& out_grad,
                      const phi::IntArray& kernel_size,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      phi::DenseTensor* in_x_grad) {
  PADDLE_GCU_KERNEL_TRACE("pool2d_grad");
  dev_ctx.template Alloc<T>(in_x_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    std::vector<int> ksize(kernel_size.GetData().begin(),
                           kernel_size.GetData().end());

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&in_x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&out)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"in_x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {in_x_grad};

    GcuAttributeMap attrs;
    attrs["adaptive"] = adaptive;
    attrs["global_pooling"] = global_pooling;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["pooling_type"] = pooling_type;
    attrs["data_format"] = data_format;
    attrs["ksize"] = ksize;
    attrs["strides"] = strides_t;
    attrs["paddings"] = paddings_t;
    attrs["ceil_mode"] = ceil_mode;
    attrs["exclusive"] = exclusive;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "pool2d_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pool2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dGradKernel,
                          float,
                          phi::dtype::float16) {}
