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
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

class IndexCopy : public HpuOperator {
 public:
  explicit IndexCopy(synDataType dtype)
      : HpuOperator("index_copy_fwd"), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, ns_IndexCopy::Params params) {
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

    std::string guid = guid_ + "_" + SynDataTypeToStr(outputs[0].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params,
                                     sizeof(params),
                                     guid.c_str(),
                                     "index_copy",
                                     nullptr,
                                     nullptr);

    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void IndexCopyKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const phi::Scalar& dim,
                     const phi::DenseTensor& index,
                     const phi::DenseTensor& source,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(input);
  ct.Add(index);
  ct.Add(source);

  ct.Add(out, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  ns_IndexCopy::Params params{};
  params.axis = dim.to<unsigned>();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_IndexCopy::Params>(
      "index_copy_kernel", inputs_dims, &params);

  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    IndexCopy op(op_info.datatype_);
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  RecipeRunner runner(recipe);
  auto tensors = ct.GetDeviceAddr();
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

template <typename Context>
void CallIndexCopyKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::Scalar& dim,
                         const phi::DenseTensor& index,
                         const phi::DenseTensor& source,
                         phi::DenseTensor* out) {
  if (input.dtype() == phi::DataType::FLOAT32) {
    custom_kernel::IndexCopyKernel<float>(
        dev_ctx, input, dim, index, source, out);
  } else if (input.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::IndexCopyKernel<phi::dtype::float16>(
        dev_ctx, input, dim, index, source, out);
  } else if (input.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::IndexCopyKernel<phi::dtype::bfloat16>(
        dev_ctx, input, dim, index, source, out);
  } else {
    throw std::runtime_error("Unsupported data type for IndexCopyKernel");
  }
}

std::vector<paddle::Tensor> IndexCopyForward(const paddle::Tensor& input,
                                             const int dim,
                                             const paddle::Tensor& index,
                                             const paddle::Tensor& source) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));

  auto input_tensor = static_cast<phi::DenseTensor*>(input.impl().get());
  auto index_tensor = static_cast<const phi::DenseTensor*>(index.impl().get());
  auto source_tensor =
      static_cast<const phi::DenseTensor*>(source.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(input_tensor->dims());

  CallIndexCopyKernel(*dev_ctx,
                      *input_tensor,
                      phi::Scalar(dim),
                      *index_tensor,
                      *source_tensor,
                      out_tensor.get());

  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(index_copy)
    .Inputs({"input", "index", "source"})
    .Outputs({"out"})
    .Attrs({"dim: int"})
    .SetKernelFn(PD_KERNEL(IndexCopyForward));
