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

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& shape,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("expand");

  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor as_strides_out;
    auto x_tensor = CreateTopsatenTensor(x);
    auto out_tensor = CreateTopsatenTensor(*out);
    auto view_out_tensor = CreateTopsatenTensor(as_strides_out);

    auto expand_shape = shape.GetData();
    auto expand_size = IntArrayToTopsatenSize(expand_shape);
    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "expand_topsatenExpand", as_strides_out, x, expand_shape);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenExpand,
                                        dev_ctx,
                                        abstract_info,
                                        view_out_tensor,
                                        x_tensor,
                                        expand_size);
    abstract_info = custom_kernel::GetAbstractInfo(
        "expand_topsatenCopy", *out, as_strides_out, false);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenCopy,
                                        dev_ctx,
                                        abstract_info,
                                        out_tensor,
                                        view_out_tensor,
                                        false);

    // auto input_shape = phi::vectorize(x.dims());
    // std::vector<int64_t> expand_shape = shape.GetData();
    // PADDLE_ENFORCE_GE(
    //     expand_shape.size(),
    //     input_shape.size(),
    //     phi::errors::InvalidArgument(
    //         "ExpandKernel check shape failed, input_shape:%s,
    //         expand_shape:%s", VectorToStr<int64_t>(input_shape).c_str(),
    //         VectorToStr<int64_t>(expand_shape).c_str()));

    // std::vector<int64_t> result_shape;
    // const int64_t dims_diff = expand_shape.size() - input_shape.size();
    // for (int64_t i = 0; i < dims_diff; ++i) {
    //   result_shape.emplace_back(expand_shape[i]);
    // }
    // for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
    //   int64_t dim = expand_shape[dims_diff + i];
    //   if (dim < input_shape[i]) {
    //     result_shape.emplace_back(input_shape[i]);
    //   } else {
    //     result_shape.emplace_back(dim);
    //   }
    // }
    // PADDLE_ENFORCE_EQ(
    //     out->dims(),
    //     phi::make_ddim(result_shape),
    //     phi::errors::InvalidArgument(
    //         "ExpandKernel check dims failed, expect %s, but get %s",
    //         out->dims().to_str().c_str(),
    //         VectorToStr<int64_t>(result_shape).c_str()));
    // VLOG(3) << "ExpandKernel:" << VectorToStr<int64_t>(result_shape)
    //         << ", out:" << out->dims();
    // dev_ctx.template Alloc<T>(out);
    // Broadcast(dev_ctx, x, out);

  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(out);
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    std::vector<int> expand_shape = GetIntList(shape.GetData());

    GcuAttributeMap attrs;
    attrs["shape"] = expand_shape;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "expand_v2",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ExpandKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
