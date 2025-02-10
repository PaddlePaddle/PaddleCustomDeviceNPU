// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

struct EinsumParams {
  synEinsumParams params;
};

class Einsum : public HpuOperator {
 public:
  Einsum() : HpuOperator("einsum") {}

  void AddNode(ConvertTensors& ct, EinsumParams& params) {
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

    guid_ = guid_ + "_" + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "Einsum",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

class EinsumF32Mult : public HpuOperator {
 public:
  EinsumF32Mult() : HpuOperator("einsum") {}

  void AddNode(ConvertTensors& ct, EinsumParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    synStatus status = synFail;
    std::string guid_unsqueeze = "expand_dims";

    auto src1 = createTensor(inputs[0].dims.size(),
                             inputs[0].type,
                             inputs[0].dims,
                             true,
                             inputs[0].name);
    std::vector<synTensor> src1_inputs;
    src1_inputs.push_back(src1);

    auto src1_dims = inputs[0].dims;
    src1_dims.push_back(1);

    auto src1_ = createTensor(
        src1_dims.size(), inputs[0].type, src1_dims, false, "src1_unsqueezed");
    std::vector<synTensor> src1_outputs;
    src1_outputs.push_back(src1_);

    synExpandDimsParams expandDimsParams1;
    expandDimsParams1.axis = 0;
    status = synNodeCreate(graphHandle_,
                           src1_inputs.data(),
                           src1_outputs.data(),
                           1,
                           1,
                           &expandDimsParams1,
                           sizeof(expandDimsParams1),
                           guid_unsqueeze.c_str(),
                           "x_unsqueeze",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] EinsumF32Mult synNodeCreate (x_unsqueeze) failed = ",
             status);

    auto src2 = createTensor(inputs[1].dims.size(),
                             inputs[1].type,
                             inputs[1].dims,
                             true,
                             inputs[1].name);
    std::vector<synTensor> src2_inputs;
    src2_inputs.push_back(src2);

    auto src2_dims = inputs[1].dims;
    src2_dims.insert(src2_dims.begin(), 1);

    auto src2_ = createTensor(
        src2_dims.size(), inputs[1].type, src2_dims, false, "src2_unsqueezed");
    std::vector<synTensor> src2_outputs;
    src2_outputs.push_back(src2_);

    synExpandDimsParams expandDimsParams2;
    expandDimsParams2.axis = inputs[1].dims.size();
    status = synNodeCreate(graphHandle_,
                           src2_inputs.data(),
                           src2_outputs.data(),
                           1,
                           1,
                           &expandDimsParams2,
                           sizeof(expandDimsParams2),
                           guid_unsqueeze.c_str(),
                           "y_unsqueeze",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] EinsumF32Mult synNodeCreate (y_unsqueeze) failed = ",
             status);

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));

    std::string guid_mult = "mult_fwd_f32";

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(src1_);
    syn_inputs.push_back(src2_);
    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           syn_outputs.data(),
                           syn_inputs.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           guid_mult.c_str(),
                           "multiply",
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] EinsumF32Mult synNodeCreate (multiply) failed = ",
             status);
  }
};

template <typename T, typename Context>
void EinsumKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& inputs,
                  const std::string& equation,
                  phi::DenseTensor* out,
                  std::vector<phi::DenseTensor*> cache,
                  std::vector<phi::DenseTensor*> xshape UNUSED) {
  VLOG(6) << "Call HPU EinsumKernel with equation = " << equation.c_str();
  std::string out_prod_2D = "i,j->ij";
  std::string out_prod_3D = "ij,k->ijk";

  dev_ctx.template Alloc<T>(out);
  ConvertTensors ct;
  for (size_t i = 0; i < inputs.size(); i++) {
    ct.Add(inputs[i]);
  }

  ct.Add(out, false);

  OpCacheOperator op_info;
  EinsumParams params;
  params.params = synEinsumParams(equation.c_str());
  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, EinsumParams>("EinsumKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if ((inputs[0]->dtype() == phi::DataType::FLOAT32) &&
        ((equation == out_prod_2D) || (equation == out_prod_3D))) {
      EinsumF32Mult op;

      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
    } else {
      Einsum op;

      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
    }
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(einsum,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::EinsumKernel,
                          float,
                          int32_t,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
