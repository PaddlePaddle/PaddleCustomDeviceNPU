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
#include "funcs.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
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
    CHKSTATUS("synNodeCreate reshape failed!");
  }
};

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

  if (dtype == phi::DataType::FLOAT32) {
    dev_ctx.template Alloc<float>(out);
  } else if (dtype == phi::DataType::FLOAT64) {
    dev_ctx.template Alloc<double>(out);
    return;
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
                          bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}