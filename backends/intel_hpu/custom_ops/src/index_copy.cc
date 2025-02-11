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

    synSectionHandle section = createSection();

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(createTensor(inputs[0].dims.size(),
                                      inputs[0].type,
                                      inputs[0].dims,
                                      true,
                                      inputs[0].name,
                                      section));

    syn_inputs.push_back(createTensor(inputs[1].dims.size(),
                                      inputs[1].type,
                                      inputs[1].dims,
                                      true,
                                      inputs[1].name));

    syn_inputs.push_back(createTensor(inputs[2].dims.size(),
                                      inputs[2].type,
                                      inputs[2].dims,
                                      true,
                                      inputs[2].name));

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name,
                                       section));

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
                     const phi::DenseTensor& source) {
  ConvertTensors ct;
  ct.Add(input);
  ct.Add(index);
  ct.Add(source);

  ct.Add(input, false);

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
                         const phi::DenseTensor& source) {
  if (input.dtype() == phi::DataType::FLOAT32) {
    custom_kernel::IndexCopyKernel<float>(dev_ctx, input, dim, index, source);
  } else if (input.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::IndexCopyKernel<phi::dtype::float16>(
        dev_ctx, input, dim, index, source);
  } else if (input.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::IndexCopyKernel<phi::dtype::bfloat16>(
        dev_ctx, input, dim, index, source);
  } else {
    throw std::runtime_error("Unsupported data type for IndexCopyKernel");
  }
}

void IndexCopyForward(const paddle::Tensor& input,
                      const int dim,
                      const paddle::Tensor& index,
                      const paddle::Tensor& source) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));

  auto input_tensor = static_cast<phi::DenseTensor*>(input.impl().get());
  auto index_tensor = static_cast<const phi::DenseTensor*>(index.impl().get());
  auto source_tensor =
      static_cast<const phi::DenseTensor*>(source.impl().get());

  CallIndexCopyKernel(
      *dev_ctx, *input_tensor, phi::Scalar(dim), *index_tensor, *source_tensor);
}

std::vector<std::vector<int64_t>> IndexCopyInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& index_shape,
    const std::vector<int64_t>& source_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> IndexCopyInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& index_dtype,
    const paddle::DataType& source_dtype) {
  return {input_dtype};
}

PD_BUILD_OP(index_copy)
    .Inputs({"input", "index", "source"})
    .Outputs({"out"})
    .Attrs({"dim: int"})
    .SetInplaceMap({{"input", "out"}})
    .SetKernelFn(PD_KERNEL(IndexCopyForward))
    .SetInferShapeFn(PD_INFER_SHAPE(IndexCopyInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(IndexCopyInferDtype));
