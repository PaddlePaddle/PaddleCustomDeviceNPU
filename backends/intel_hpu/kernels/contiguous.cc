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

#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {
struct ContiguousParams {
  std::vector<int32_t> input_strides;
  synStridedOpParams params;
  std::vector<int64_t> input_dims;
};

class Contiguous : public HpuOperator {
 public:
  Contiguous() : HpuOperator("strided_view") {}

  void AddNode(ConvertTensors& ct, ContiguousParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);
    std::vector<synTensor> syn_inputs;
    if (params.params.baseOffset != 0) {
      syn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                        inputs[0].type,
                                        params.input_dims,
                                        true,
                                        inputs[0].name));
    } else {
      syn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                        inputs[0].type,
                                        inputs[0].dims,
                                        true,
                                        inputs[0].name));
    }

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "Contiguous",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);
  const T* input_data = input.data<T>();
  auto* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  auto dims = input.dims();
  auto input_stride = input.strides();
  auto numel = input.numel();

  for (int64_t i = 0; i < numel; i++) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t mod = index_tmp % dims[dim];
      index_tmp = index_tmp / dims[dim];
      input_offset += mod * input_stride[dim];
    }

    C_Device_st device{input.place().GetDeviceId()};
    C_Stream stream = static_cast<C_Stream>(dev_ctx.stream());
    auto* dst_ptr = &output_data[i];
    auto* src_ptr = &input_data[input_offset];
    AsyncMemCpyD2D(
        &device, stream, dst_ptr, src_ptr, phi::SizeOf(input.dtype()));
  }
  dev_ctx.Wait();

  // phi::DenseTensorMeta meta = input.meta();
  // ContiguousParams params;
  // params.params.baseOffset = meta.offset / sizeof(T);
  // LOG(INFO) << "== " << params.params.baseOffset;

  // params.input_strides = phi::vectorize<int32_t>(meta.strides);
  // auto rank = params.input_strides.size();
  // for (size_t i = 0; i < params.input_strides.size(); i++) {
  //   params.params.strides[rank - 1 - i] = params.input_strides[i];
  // }
  // // calculate inputs dim
  // params.input_dims = phi::vectorize<int64_t>(meta.dims);
  // int64_t count = 1;
  // for (int i = rank - 2; i >= 0; i--) {
  //   params.input_dims[i + 1] = meta.strides[i] / count;
  //   count = count * params.input_dims[i + 1];
  // }

  // meta.strides = meta.calc_strides(meta.dims);
  // meta.offset = 0;
  // out->set_meta(meta);
  // dev_ctx.template Alloc<T>(out);

  // ConvertTensors ct;
  // ct.Add(input);
  // ct.Add(out, false);

  // std::vector<DIMS> inputs_dims = ct.GetDims();
  // OpCacheOperator op_info;
  // op_info.prepareOpInfo<T, ContiguousParams>(
  //     "ContiguousKernel", inputs_dims, &params);
  // auto recipe = op_info.GetRecipe();

  // if (recipe == nullptr) {
  //   Contiguous op;
  //   op.AddNode(ct, params);
  //   op.Compile();
  //   op_info.setOp(op);
  //   recipe = op_info.GetRecipe();
  // }

  // RecipeRunner runner(recipe);
  // auto tensors = ct.GetDeviceAddr();
  // runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(contiguous,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ContiguousKernel,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
