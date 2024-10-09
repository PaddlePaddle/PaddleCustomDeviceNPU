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
#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

struct FullParams {
  phi::DataType dst_type;
  ns_ConstantKernel::Params params;
};

class Full : public HpuOperator {
 public:
  Full() : HpuOperator("constant_") {}

  void AddNode(ConvertTensors& ct, FullParams params) {
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }
    if (outputs[0].type == syn_type_int64)
      guid_ = guid_ + "i32";
    else
      guid_ = guid_ + SynDataTypeToStr(outputs[0].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     syn_outputs.data(),
                                     0,
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "constant",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out) {
  VLOG(6) << "HPU FullKernel with val = " << val;
  auto int_shape = shape.GetData();
  out->Resize(phi::make_ddim(int_shape));
  if (out->dims().size() == 0) {
    out->Resize({1});
  }
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(out, false);

  OpCacheOperator op_info;
  FullParams params;
  params.dst_type = dtype;
  if (dtype == phi::DataType::FLOAT32 || dtype == phi::DataType::FLOAT16 ||
      dtype == phi::DataType::BFLOAT16) {
    params.params.constant.f = val.to<float>();
  } else if (dtype == phi::DataType::INT32 || dtype == phi::DataType::INT8 ||
             dtype == phi::DataType::INT64 || dtype == phi::DataType::UINT8) {
    params.params.constant.i = val.to<int>();
  } else if (dtype == phi::DataType::BOOL) {
    params.params.constant.i = -val.to<int>();
  } else {
    phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
  }
  std::vector<DIMS> outputs_dims = ct.GetDims(false);
  op_info.prepareOpInfo<T, FullParams>("FullKernel", outputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Full op;

    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  VLOG(6) << "HPU FullLikeKernel with val = " << val;
  std::vector<int64_t> shape_vec = phi::vectorize(x.dims());
  phi::IntArray out_shape(shape_vec);
  custom_kernel::FullKernel<T, Context>(dev_ctx, out_shape, val, dtype, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(full,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          float,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          float,
                          uint8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
