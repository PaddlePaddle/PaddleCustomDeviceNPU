// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"
namespace custom_kernel {

struct CastParams {
  phi::DataType src_type;
  phi::DataType dst_type;
};

class Cast : public HpuOperator {
 public:
  Cast() : HpuOperator("cast") {}

  void AddNode(ConvertTensors& ct) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                        inputs[i].type,
                                        inputs[i].dims,
                                        true,
                                        inputs[i].name));
    }

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    guid_ = guid_ + "_" + SynDataTypeToStr(inputs[0].type) + "_to_";
    guid_ = guid_ + SynDataTypeToStr(outputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     "CAST",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void CastKernelF32(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DataType dtype,
                   phi::DenseTensor* out) {
  // from fp32 or to fp32
  if (x.dtype() == dtype) {
    dev_ctx.template Alloc<T>(out);
    TensorCopy(dev_ctx, x, true, out);
    return;
  }

  if (dtype == phi::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(out);
  } else if (dtype == phi::DataType::FLOAT16) {
    dev_ctx.template Alloc<phi::dtype::float16>(out);
  } else if (dtype == phi::DataType::BFLOAT16) {
    dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
  } else if (dtype == phi::DataType::INT16) {
    dev_ctx.template Alloc<int16_t>(out);
  } else if (dtype == phi::DataType::INT32) {
    dev_ctx.template Alloc<int32_t>(out);
  } else if (dtype == phi::DataType::INT64) {
    dev_ctx.template Alloc<int64_t>(out);
  } else if (dtype == phi::DataType::UINT8) {
    dev_ctx.template Alloc<uint8_t>(out);
  } else if (dtype == phi::DataType::BOOL) {
    dev_ctx.template Alloc<bool>(out);
  } else if (dtype == phi::DataType::INT8) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
  }

  OpCacheOperator op_info;
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  CastParams params;
  params.src_type = x.dtype();
  params.dst_type = dtype;

  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, CastParams>("CastKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Cast op;

    op.AddNode(ct);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  if (x.dtype() == dtype) {
    dev_ctx.template Alloc<T>(out);
    TensorCopy(dev_ctx, x, true, out);
    return;
  }

  if (x.dtype() == phi::DataType::FLOAT64) {
    // copy float64 from device to host
    phi::DenseTensor x_double_host;
    phi::DenseTensorMeta x_double_meta(phi::DataType::FLOAT64, x.dims());
    x_double_host.set_meta(x_double_meta);
    double* p_x_double_host =
        dev_ctx.template HostAlloc<double>(&x_double_host);
    TensorCopy(dev_ctx, x, true, &x_double_host, phi::CPUPlace());

    // cast float64 to float32
    phi::DenseTensor x_float_host;
    phi::DenseTensorMeta x_float_meta(phi::DataType::FLOAT32, x.dims());
    x_float_host.set_meta(x_float_meta);
    float* p_x_float_host = dev_ctx.template HostAlloc<float>(&x_float_host);
    auto numel = x_double_host.numel();
    for (auto i = 0; i < numel; ++i) {
      p_x_float_host[i] = static_cast<float>(p_x_double_host[i]);
    }

    // do cast from float32 to other
    if (dtype == phi::DataType::FLOAT32) {
      dev_ctx.template Alloc<T>(out);
      TensorCopy(dev_ctx, x_float_host, true, out);
      return;
    }

    // copy float32 from host to device
    phi::DenseTensor x_float_device;
    x_float_device.set_meta(x_float_meta);
    dev_ctx.template Alloc<float>(&x_float_device);
    TensorCopy(dev_ctx, x_float_host, true, &x_float_device);

    custom_kernel::CastKernelF32<float, Context>(
        dev_ctx, x_float_device, dtype, out);
    return;
  }

  if (dtype == phi::DataType::FLOAT64) {
    phi::DenseTensor out_float_device;
    phi::DenseTensorMeta float_meta(phi::DataType::FLOAT32, x.dims());
    out_float_device.set_meta(float_meta);
    custom_kernel::CastKernelF32<T, Context>(
        dev_ctx, x, phi::DataType::FLOAT32, &out_float_device);

    phi::DenseTensor out_float_host;
    out_float_host.set_meta(float_meta);
    auto p_x_host = dev_ctx.template HostAlloc<float>(&out_float_host);
    TensorCopy(
        dev_ctx, out_float_device, true, &out_float_host, phi::CPUPlace());

    phi::DenseTensor out_double_host;
    phi::DenseTensorMeta double_meta(phi::DataType::FLOAT64, x.dims());
    out_double_host.set_meta(double_meta);
    double* p_double_host =
        dev_ctx.template HostAlloc<double>(&out_double_host);
    auto numel = x.numel();
    for (auto i = 0; i < numel; ++i) {
      p_double_host[i] = static_cast<double>(static_cast<float>(p_x_host[i]));
    }

    dev_ctx.template Alloc<double>(out);
    TensorCopy(dev_ctx, out_double_host, true, out);
    return;
  }

  custom_kernel::CastKernelF32<T, Context>(dev_ctx, x, dtype, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cast,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::CastKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
