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
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

class FusedMlp : public HpuOperator {
 public:
  explicit FusedMlp(synDataType dtype)
      : HpuOperator("fused_mlp_fwd", false), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    synTensor hidden_states = createTensor(
        ins[0].dims.size(), dtype_, ins[0].dims, true, ins[0].name);
    synTensor gate_weight = createTensor(
        ins[1].dims.size(), dtype_, ins[1].dims, true, ins[1].name);
    synTensor up_weight = createTensor(
        ins[2].dims.size(), dtype_, ins[2].dims, true, ins[2].name);
    synTensor down_weight = createTensor(
        ins[3].dims.size(), dtype_, ins[3].dims, true, ins[3].name);

    std::vector<int64_t> proj_dims = {
        ins[0].dims[0], ins[0].dims[1], ins[1].dims[1]};
    synTensor gate_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "gate_out");
    synTensor up_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "up_out");
    synTensor silu_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "silu_out");
    synTensor mul_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "mul_out");

    synTensor mlp_out = createTensor(
        outs[0].dims.size(), dtype_, outs[0].dims, true, outs[0].name);

    std::vector<synTensor> gate_inputs;
    gate_inputs.push_back(hidden_states);
    gate_inputs.push_back(gate_weight);
    std::vector<synTensor> gate_outputs;
    gate_outputs.push_back(gate_out);

    std::vector<synTensor> up_inputs;
    up_inputs.push_back(hidden_states);
    up_inputs.push_back(up_weight);
    std::vector<synTensor> up_outputs;
    up_outputs.push_back(up_out);

    std::vector<synTensor> silu_inputs;
    silu_inputs.push_back(gate_out);
    std::vector<synTensor> silu_outputs;
    silu_outputs.push_back(silu_out);

    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(silu_out);
    mul_inputs.push_back(up_out);
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(mul_out);

    std::vector<synTensor> down_inputs;
    down_inputs.push_back(mul_out);
    down_inputs.push_back(down_weight);
    std::vector<synTensor> down_outputs;
    down_outputs.push_back(mlp_out);

    std::string matmul = "gemm";
    std::string silu = "silu_fwd_";
    std::string mul = "mult_fwd_";
    if (dtype_ == syn_type_fp16) {
      silu = silu + "f16";
      mul = mul + "f16";
    } else if (dtype_ == syn_type_bf16) {
      silu = silu + "bf16";
      mul = mul + "bf16";
    } else if (dtype_ == syn_type_single) {
      silu = silu + "f32";
      mul = mul + "f32";
    }

    std::string silu_name = guid_ + "_silu";
    std::string mul_name = guid_ + "_mul";
    std::string gate_name = guid_ + "_gate_proj";
    std::string up_name = guid_ + "_up_proj";
    std::string down_name = guid_ + "_down_proj";

    synStatus status = synNodeCreate(graphHandle_,
                                     gate_inputs.data(),
                                     gate_outputs.data(),
                                     gate_inputs.size(),
                                     gate_outputs.size(),
                                     nullptr,
                                     0,
                                     matmul.c_str(),
                                     gate_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           up_inputs.data(),
                           up_outputs.data(),
                           up_inputs.size(),
                           up_outputs.size(),
                           nullptr,
                           0,
                           matmul.c_str(),
                           up_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           silu_inputs.data(),
                           silu_outputs.data(),
                           silu_inputs.size(),
                           silu_outputs.size(),
                           nullptr,
                           0,
                           silu.c_str(),
                           silu_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           mul_inputs.data(),
                           mul_outputs.data(),
                           mul_inputs.size(),
                           mul_outputs.size(),
                           nullptr,
                           0,
                           mul.c_str(),
                           mul_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           down_inputs.data(),
                           down_outputs.data(),
                           down_inputs.size(),
                           down_outputs.size(),
                           nullptr,
                           0,
                           matmul.c_str(),
                           down_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

class FusedGateUpMlp : public HpuOperator {
 public:
  explicit FusedGateUpMlp(synDataType dtype)
      : HpuOperator("fused_gate_up_mlp_fwd", false), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, synSplitParams params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    synTensor hidden_states = createTensor(
        ins[0].dims.size(), dtype_, ins[0].dims, true, ins[0].name);
    synTensor proj_weight = createTensor(
        ins[1].dims.size(), dtype_, ins[1].dims, true, ins[1].name);
    std::vector<int64_t> proj_dims = {
        ins[0].dims[0], ins[0].dims[1], ins[1].dims[1]};
    synTensor proj_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "proj_out");

    std::vector<int64_t> split_out_dims = {
        proj_dims[0], proj_dims[1], proj_dims[2] / 2};
    synTensor gate_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "gate_out");
    synTensor up_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "up_out");

    synTensor down_weight = createTensor(
        ins[2].dims.size(), dtype_, ins[2].dims, true, ins[2].name);

    synTensor silu_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "silu_out");
    synTensor mul_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "mul_out");

    synTensor mlp_out = createTensor(
        outs[0].dims.size(), dtype_, outs[0].dims, true, outs[0].name);

    std::vector<synTensor> proj_inputs;
    proj_inputs.push_back(hidden_states);
    proj_inputs.push_back(proj_weight);
    std::vector<synTensor> proj_outputs;
    proj_outputs.push_back(proj_out);

    std::vector<synTensor> split_inputs;
    split_inputs.push_back(proj_out);
    std::vector<synTensor> split_outputs;
    split_outputs.push_back(gate_out);
    split_outputs.push_back(up_out);

    std::vector<synTensor> silu_inputs;
    silu_inputs.push_back(gate_out);
    std::vector<synTensor> silu_outputs;
    silu_outputs.push_back(silu_out);

    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(silu_out);
    mul_inputs.push_back(up_out);
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(mul_out);

    std::vector<synTensor> down_inputs;
    down_inputs.push_back(mul_out);
    down_inputs.push_back(down_weight);
    std::vector<synTensor> down_outputs;
    down_outputs.push_back(mlp_out);

    std::string split = "split";
    std::string matmul = "gemm";
    std::string silu = "silu_fwd_";
    std::string mul = "mult_fwd_";
    if (dtype_ == syn_type_fp16) {
      silu = silu + "f16";
      mul = mul + "f16";
    } else if (dtype_ == syn_type_bf16) {
      silu = silu + "bf16";
      mul = mul + "bf16";
    } else if (dtype_ == syn_type_single) {
      silu = silu + "f32";
      mul = mul + "f32";
    }

    std::string proj_name = guid_ + "_proj";
    std::string split_name = guid_ + "_split_proj";
    std::string silu_name = guid_ + "_silu";
    std::string mul_name = guid_ + "_mul";
    std::string down_name = guid_ + "_down_proj";

    synStatus status = synNodeCreate(graphHandle_,
                                     proj_inputs.data(),
                                     proj_outputs.data(),
                                     proj_inputs.size(),
                                     proj_outputs.size(),
                                     nullptr,
                                     0,
                                     matmul.c_str(),
                                     proj_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           split_inputs.data(),
                           split_outputs.data(),
                           split_inputs.size(),
                           split_outputs.size(),
                           &params,
                           sizeof(params),
                           split.c_str(),
                           split_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           silu_inputs.data(),
                           silu_outputs.data(),
                           silu_inputs.size(),
                           silu_outputs.size(),
                           nullptr,
                           0,
                           silu.c_str(),
                           silu_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           mul_inputs.data(),
                           mul_outputs.data(),
                           mul_inputs.size(),
                           mul_outputs.size(),
                           nullptr,
                           0,
                           mul.c_str(),
                           mul_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           down_inputs.data(),
                           down_outputs.data(),
                           down_inputs.size(),
                           down_outputs.size(),
                           nullptr,
                           0,
                           matmul.c_str(),
                           down_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedMlpKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& gate_weight,
                    const phi::DenseTensor& up_weight,
                    const phi::DenseTensor& down_weight,
                    phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(gate_weight);
  ct.Add(up_weight);
  ct.Add(down_weight);
  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("FusedMlpKernel", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedMlp op(op_info.datatype_);
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
void FusedGateUpMlpKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& proj_weight,
                          const phi::DenseTensor& down_weight,
                          phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  const phi::Scalar axis_scalar = proj_weight.dims().size() - 1;
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = proj_weight.dims().size() + axis;
  }
  synSplitParams params = {{0}};
  params.axis = proj_weight.dims().size() - 1 - axis;

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(proj_weight);
  ct.Add(down_weight);
  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synSplitParams>(
      "FusedGateUpMlpKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedGateUpMlp op(op_info.datatype_);
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

template <typename Context>
void CallFusedMlpKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& gate_weight,
                        const phi::DenseTensor& up_weight,
                        const phi::DenseTensor& down_weight,
                        phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedMlpKernel<phi::dtype::bfloat16>(
        dev_ctx, x, gate_weight, up_weight, down_weight, out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedMlpKernel");
  }
}

template <typename Context>
void CallFusedGateUpMlpKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& proj_weight,
                              const phi::DenseTensor& down_weight,
                              phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedGateUpMlpKernel<phi::dtype::bfloat16>(
        dev_ctx, x, proj_weight, down_weight, out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedGateUpMlpKernel");
  }
}

std::vector<paddle::Tensor> FusedMlpForward(
    const paddle::Tensor& x,
    const paddle::Tensor& proj_weight,
    const paddle::optional<paddle::Tensor>& up_weight,
    const paddle::Tensor& down_weight) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());

  auto down_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());

  if (up_weight) {
    auto gate_tensor =
        static_cast<const phi::DenseTensor*>(proj_weight.impl().get());
    auto up_tensor =
        static_cast<const phi::DenseTensor*>(up_weight->impl().get());

    CallFusedMlpKernel(*dev_ctx,
                       *x_tensor,
                       *gate_tensor,
                       *up_tensor,
                       *down_tensor,
                       out_tensor.get());
  } else {
    auto proj_tensor =
        static_cast<const phi::DenseTensor*>(proj_weight.impl().get());

    CallFusedGateUpMlpKernel(
        *dev_ctx, *x_tensor, *proj_tensor, *down_tensor, out_tensor.get());
  }

  paddle::Tensor out(out_tensor);

  return {out};
}

PD_BUILD_OP(fused_mlp)
    .Inputs({"x", "proj_weight", paddle::Optional("up_weight"), "down_weight"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(FusedMlpForward));
