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
void ReduceBaseKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      phi::DenseTensor* out,
                      const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;
  auto dim = dims.GetData();
  attrs["dim"] = std::vector<int>(dim.begin(), dim.end());
  attrs["keep_dim"] = keep_dim;
  attrs["reduce_all"] = reduce_all;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void ReduceGradBaseKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& out_grad,
                          const phi::IntArray& dims_array,
                          bool keep_dim,
                          bool reduce_all,
                          phi::DenseTensor* x_grad,
                          const std::string& op_type) {
  dev_ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  TensorValueMap outputs;

  output_names[GradVarName("X")] = {"x_grad"};
  outputs[GradVarName("X")] = {x_grad};

  GcuAttributeMap attrs;
  auto dim = dims_array.GetData();
  attrs["dim"] = std::vector<int>(dim.begin(), dim.end());
  attrs["keep_dim"] = keep_dim;
  attrs["reduce_all"] = reduce_all;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void AnyKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("any");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    if (dims.size() == 1) {
      int64_t dim = dims[0];
      if (dim < 0) {
        dim += x.dims().size();
      }
      LAUNCH_TOPSATENOP(topsatenAny, dev_ctx, *out, x, dim, keep_dim);
    } else {
      ContextPinnedGuard<Context> ctx_pinned_guard(dev_ctx);
      // fallback to CPU
      // 1. Copy x to CPU
      phi::DenseTensor x_cpu;
      x_cpu.set_meta(x.meta());
      TensorCopy(dev_ctx, x, false, &x_cpu, phi::CPUPlace());
      dev_ctx.Wait();

      // 2. Call the CPU implementation
      phi::CPUContext dev_ctx_cpu;
      dev_ctx_cpu.SetAllocator(&(dev_ctx.GetHostAllocator()));
      dev_ctx_cpu.SetHostAllocator(&(dev_ctx.GetHostAllocator()));
      phi::DenseTensor out_cpu;
      out_cpu.set_meta(out->meta());
      phi::AnyKernel<T, phi::CPUContext>(
          dev_ctx_cpu, x_cpu, dims, keep_dim, &out_cpu);
      dev_ctx.Wait();

      // 3. Copy result to device
      TensorCopy(dev_ctx, out_cpu, false, out);
      dev_ctx.Wait();
    }

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("max");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = dims.GetData();
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    LAUNCH_TOPSATENOP(topsatenMax, dev_ctx, *out, x, reduce_axis, keep_dim);

  } else {  // kernel impl base on JIT
    ReduceBaseKernel<T, Context>(dev_ctx,
                                 x,
                                 dims,
                                 keep_dim,
                                 ((dims.size() == 0) ? true : false),
                                 out,
                                 "reduce_max");
  }
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("min");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = dims.GetData();
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    LAUNCH_TOPSATENOP(topsatenMin, dev_ctx, *out, x, reduce_axis, keep_dim);

  } else {  // kernel impl base on JIT
    ReduceBaseKernel<T, Context>(dev_ctx,
                                 x,
                                 dims,
                                 keep_dim,
                                 ((dims.size() == 0) ? true : false),
                                 out,
                                 "reduce_min");
  }
}

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& axes,
                bool keep_dim,
                bool reduce_all,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("prod");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = axes.GetData();
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    LAUNCH_TOPSATENOP(
        topsatenProd, dev_ctx, *out, x, reduce_axis, keep_dim, out->dtype());

  } else {  // kernel impl base on JIT
    ReduceBaseKernel<T, Context>(
        dev_ctx, x, axes, keep_dim, reduce_all, out, "reduce_prod");
  }
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sum");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = dims.GetData();
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor output =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    LAUNCH_TOPSATENOP(topsatenSum,
                      dev_ctx,
                      output,
                      input_x,
                      reduce_axis,
                      keep_dim,
                      out->dtype());
    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
    ReduceBaseKernel<T, Context>(dev_ctx,
                                 x,
                                 dims,
                                 keep_dim,
                                 ((dims.size() == 0) ? true : false),
                                 out,
                                 "reduce_sum");
  }
}

template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& dims_array,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("sum_grad");
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    return;
  }

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(x_grad);
    ReduceGradBaseKernel<T, Context>(dev_ctx,
                                     x,
                                     out_grad,
                                     dims_array,
                                     keep_dim,
                                     reduce_all,
                                     x_grad,
                                     "reduce_sum_grad");
  }
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("mean");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = dims.GetData();
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    LAUNCH_TOPSATENOP(
        topsatenMean, dev_ctx, *out, x, reduce_axis, keep_dim, out->dtype());

  } else {  // kernel impl base on JIT
    ReduceBaseKernel<T, Context>(dev_ctx,
                                 x,
                                 dims,
                                 keep_dim,
                                 ((dims.size() == 0) ? true : false),
                                 out,
                                 "reduce_mean");
  }
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& axes,
                    bool keep_dim,
                    bool reduce_all,
                    phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("mean_grad");
  dev_ctx.template Alloc<T>(x_grad);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    return;
  }
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ReduceGradBaseKernel<T, Context>(dev_ctx,
                                     x,
                                     out_grad,
                                     axes,
                                     keep_dim,
                                     reduce_all,
                                     x_grad,
                                     "reduce_mean_grad");
  }
}

template <typename T, typename Context>
void AMaxKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("amax");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = dims;
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }
    LAUNCH_TOPSATENOP(topsatenAmax, dev_ctx, *out, x, reduce_axis, keep_dim);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void AMinKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("amin");
  if (LaunchAOTKernel()) {
    dev_ctx.template Alloc<T>(out);
    auto reduce_axis = dims;
    int64_t rank = x.dims().size();
    if (reduce_axis.empty()) {
      reduce_axis.assign(rank, 0);
      std::iota(reduce_axis.begin(), reduce_axis.end(), 0);
    } else {
      for (size_t i = 0; i < reduce_axis.size(); ++i) {
        if (reduce_axis[i] < 0) {
          reduce_axis[i] += rank;
        }
      }
    }

    auto x_tensor = CreateTopsatenTensor(x);
    auto out_tensor = CreateTopsatenTensor(*out);
    auto reduce_dims = IntArrayToTopsatenSize(reduce_axis);
    std::string abstract_info = custom_kernel::GetAbstractInfo(
        "topsatenAmin", *out, x, reduce_axis, keep_dim);
    LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(topsatenAmin,
                                        dev_ctx,
                                        abstract_info,
                                        out_tensor,
                                        x_tensor,
                                        reduce_dims,
                                        keep_dim);
    // LAUNCH_TOPSATENOP(topsatenAmin, dev_ctx, *out, x, reduce_axis, keep_dim);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(any,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AnyKernel,
                          float,
                          int,
                          bool,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(max,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          int32_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(min,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MinKernel,
                          int32_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(prod,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ProdKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(mean,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeanGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(amax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AMaxKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(amin,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AMinKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
