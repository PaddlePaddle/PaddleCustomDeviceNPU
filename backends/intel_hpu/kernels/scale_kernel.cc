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

struct ScaleParams {
  ns_ConstantKernel::Params scalerParams;
  ns_ConstantKernel::Params biasParams;
};

class Scale : public HpuOperator {
 public:
  Scale() : HpuOperator("scale_", false) {}

  void AddNode(ConvertTensors& ct, ScaleParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    synSectionHandle section = nullptr;
    if (inputs[0].device_addr == outputs[0].device_addr) {
      section = createSection();
    }

    std::string guid_full = "constant_f32";
    std::string name_scaler = guid_ + "full_scaler";
    std::string name_bias = guid_ + "full_bias";

    std::string guid_mul = "mult_fwd_f32";
    std::string name_mul = guid_ + "mul";

    std::string guid_add = "add_fwd_f32";
    std::string name_add = guid_ + "add";

    std::vector<int64_t> scaler_dims = {1};

    std::vector<synTensor> scaler_out;
    auto scaler_tensor = createTensor(scaler_dims.size(),
                                      syn_type_single,
                                      scaler_dims,
                                      false,
                                      "scaler_tensor");
    scaler_out.push_back(scaler_tensor);

    std::vector<synTensor> bias_out;
    auto bias_tensor = createTensor(
        scaler_dims.size(), syn_type_single, scaler_dims, false, "bias_tensor");
    bias_out.push_back(bias_tensor);

    synStatus status = synFail;

    status = synNodeCreate(graphHandle_,
                           nullptr,
                           scaler_out.data(),
                           0,
                           scaler_out.size(),
                           &params.scalerParams,
                           sizeof(params.scalerParams),
                           guid_full.c_str(),
                           name_scaler.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/full_scaler) failed = ",
             status);

    status = synNodeCreate(graphHandle_,
                           nullptr,
                           bias_out.data(),
                           0,
                           bias_out.size(),
                           &params.biasParams,
                           sizeof(params.biasParams),
                           guid_full.c_str(),
                           name_bias.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/full_bias) failed = ",
             status);

    std::vector<synTensor> mul_in;
    auto x_tensor = createTensor(inputs[0].dims.size(),
                                 inputs[0].type,
                                 inputs[0].dims,
                                 true,
                                 inputs[0].name,
                                 section);
    mul_in.push_back(x_tensor);
    mul_in.push_back(scaler_tensor);

    std::vector<synTensor> mul_out;
    auto x_mul = createTensor(
        inputs[0].dims.size(), syn_type_single, inputs[0].dims, false, "x_mul");
    mul_out.push_back(x_mul);

    status = synNodeCreate(graphHandle_,
                           mul_in.data(),
                           mul_out.data(),
                           2,
                           1,
                           nullptr,
                           0,
                           guid_mul.c_str(),
                           name_mul.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/mul) failed = ",
             status);

    std::vector<synTensor> add_in;
    add_in.push_back(x_mul);
    add_in.push_back(bias_tensor);

    std::vector<synTensor> add_out;
    auto out_tensor = createTensor(outputs[0].dims.size(),
                                   outputs[0].type,
                                   outputs[0].dims,
                                   true,
                                   outputs[0].name,
                                   section);
    add_out.push_back(out_tensor);

    status = synNodeCreate(graphHandle_,
                           add_in.data(),
                           add_out.data(),
                           2,
                           1,
                           nullptr,
                           0,
                           guid_add.c_str(),
                           name_add.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/mul) failed = ",
             status);
  }
};

class ScaleCast : public HpuOperator {
 public:
  ScaleCast() : HpuOperator("scale_", false) {}
  void AddNode(ConvertTensors& ct, ScaleParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    synSectionHandle section = nullptr;
    if (inputs[0].device_addr == outputs[0].device_addr) {
      section = createSection();
    }

    std::string guid_full = "constant_f32";
    std::string name_scaler = guid_ + "full_scaler";
    std::string name_bias = guid_ + "full_bias";

    std::string guid_cast = "cast_";
    std::string guid_cast_i = guid_cast + SynDataTypeToStr(inputs[0].type) +
                              "_to_" + SynDataTypeToStr(syn_type_single);
    std::string guid_cast_o = guid_cast + SynDataTypeToStr(syn_type_single) +
                              "_to_" + SynDataTypeToStr(inputs[0].type);
    std::string name_cast_i = guid_ + "cast_in";
    std::string name_cast_o = guid_ + "cast_out";

    std::string guid_mul = "mult_fwd_f32";
    std::string name_mul = guid_ + "mul";

    std::string guid_add = "add_fwd_f32";
    std::string name_add = guid_ + "add";

    std::vector<int64_t> scaler_dims = {1};

    std::vector<synTensor> scaler_out;
    auto scaler_tensor = createTensor(scaler_dims.size(),
                                      syn_type_single,
                                      scaler_dims,
                                      false,
                                      "scaler_tensor");
    scaler_out.push_back(scaler_tensor);

    std::vector<synTensor> bias_out;
    auto bias_tensor = createTensor(
        scaler_dims.size(), syn_type_single, scaler_dims, false, "bias_tensor");
    bias_out.push_back(bias_tensor);

    synStatus status = synFail;

    status = synNodeCreate(graphHandle_,
                           nullptr,
                           scaler_out.data(),
                           0,
                           scaler_out.size(),
                           &params.scalerParams,
                           sizeof(params.scalerParams),
                           guid_full.c_str(),
                           name_scaler.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/full_scaler) failed = ",
             status);

    status = synNodeCreate(graphHandle_,
                           nullptr,
                           bias_out.data(),
                           0,
                           bias_out.size(),
                           &params.biasParams,
                           sizeof(params.biasParams),
                           guid_full.c_str(),
                           name_bias.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/full_bias) failed = ",
             status);

    std::vector<synTensor> cast_in;
    auto x_tensor = createTensor(inputs[0].dims.size(),
                                 inputs[0].type,
                                 inputs[0].dims,
                                 true,
                                 inputs[0].name,
                                 section);
    cast_in.push_back(x_tensor);

    std::vector<synTensor> cast_out;
    auto x_cast = createTensor(inputs[0].dims.size(),
                               syn_type_single,
                               inputs[0].dims,
                               false,
                               "x_cast");
    cast_out.push_back(x_cast);

    status = synNodeCreate(graphHandle_,
                           cast_in.data(),
                           cast_out.data(),
                           cast_in.size(),
                           cast_out.size(),
                           nullptr,
                           0,
                           guid_cast_i.c_str(),
                           name_cast_i.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/cast_in) failed = ",
             status);

    std::vector<synTensor> mul_in;
    mul_in.push_back(x_cast);
    mul_in.push_back(scaler_tensor);

    std::vector<synTensor> mul_out;
    auto x_mul = createTensor(
        inputs[0].dims.size(), syn_type_single, inputs[0].dims, false, "x_mul");
    mul_out.push_back(x_mul);

    status = synNodeCreate(graphHandle_,
                           mul_in.data(),
                           mul_out.data(),
                           2,
                           1,
                           nullptr,
                           0,
                           guid_mul.c_str(),
                           name_mul.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/mul) failed = ",
             status);

    std::vector<synTensor> add_in;
    add_in.push_back(x_mul);
    add_in.push_back(bias_tensor);

    std::vector<synTensor> add_out;
    auto x_add = createTensor(
        inputs[0].dims.size(), syn_type_single, inputs[0].dims, false, "x_add");
    add_out.push_back(x_add);

    status = synNodeCreate(graphHandle_,
                           add_in.data(),
                           add_out.data(),
                           2,
                           1,
                           nullptr,
                           0,
                           guid_add.c_str(),
                           name_add.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/mul) failed = ",
             status);

    std::vector<synTensor> final_cast_in;
    final_cast_in.push_back(x_add);

    std::vector<synTensor> final_cast_out;
    auto out_tensor = createTensor(outputs[0].dims.size(),
                                   outputs[0].type,
                                   outputs[0].dims,
                                   true,
                                   outputs[0].name,
                                   section);
    final_cast_out.push_back(out_tensor);

    status = synNodeCreate(graphHandle_,
                           final_cast_in.data(),
                           final_cast_out.data(),
                           final_cast_in.size(),
                           final_cast_out.size(),
                           nullptr,
                           0,
                           guid_cast_o.c_str(),
                           name_cast_o.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (scale/cast_in) failed = ",
             status);
  }
};

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 const phi::Scalar& in_bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  VLOG(6) << "call HPU ScaleKernel";

  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  ScaleParams params;
  auto scale = in_scale.to<float>();
  auto bias = in_bias.to<float>();

  if (!bias_after_scale) {
    bias = bias * scale;
  }
  params.scalerParams.constant.f = scale;
  params.biasParams.constant.f = bias;

  OpCacheOperator op_info;
  std::string op_name =
      (x.data() == out->data()) ? "_scaleKernel" : "scaleKernel";
  op_info.prepareOpInfo<T, ScaleParams>(op_name, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    if (x.dtype() != phi::DataType::FLOAT32) {
      ScaleCast op;

      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    } else {
      Scale op;

      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);

      recipe = op_info.GetRecipe();
    }
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          float,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
