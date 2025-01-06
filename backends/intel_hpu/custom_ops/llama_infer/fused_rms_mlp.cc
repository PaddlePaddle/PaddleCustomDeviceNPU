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

struct FusedRmsMlpParams {
  ns_LayerNormKernel::Params rmsnorm_params;
  synSplitParams split_params;
};

class FusedRmsMlp : public HpuOperator {
 public:
  explicit FusedRmsMlp(synDataType dtype)
      : HpuOperator("fused_rms_mlp_fwd", false), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, FusedRmsMlpParams params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    std::string guid_rmsnorm = "rms_norm_ex_fwd_";
    std::string guid_split = "split";
    std::string guid_matmul = "gemm";
    std::string guid_silu = "silu_fwd_";
    std::string guid_multi = "mult_fwd_";
    if (dtype_ == syn_type_fp16) {
      guid_rmsnorm = guid_rmsnorm + "f16";
      guid_silu = guid_silu + "f16";
      guid_multi = guid_multi + "f16";
    } else if (dtype_ == syn_type_bf16) {
      guid_rmsnorm = guid_rmsnorm + "bf16";
      guid_silu = guid_silu + "bf16";
      guid_multi = guid_multi + "bf16";
    } else if (dtype_ == syn_type_single) {
      guid_rmsnorm = guid_rmsnorm + "f32";
      guid_silu = guid_silu + "f32";
      guid_multi = guid_multi + "f32";
    }

    std::string name_rmsnorm = guid_ + "_rmsnorm";
    std::string name_proj = guid_ + "_proj";
    std::string name_split = guid_ + "_split_proj";
    std::string name_silu = guid_ + "_silu";
    std::string name_multi = guid_ + "_multi";
    std::string name_down = guid_ + "_down_proj";

    auto hidden_states = createTensor(
        ins[0].dims.size(), dtype_, ins[0].dims, true, ins[0].name);
    auto ln_scales = createTensor(
        ins[1].dims.size(), dtype_, ins[1].dims, true, ins[1].name);

    std::vector<synTensor> rmsnorm_inputs;
    rmsnorm_inputs.push_back(hidden_states);
    rmsnorm_inputs.push_back(ln_scales);

    auto tmp_dims = ins[0].dims;
    tmp_dims[2] = 1;
    auto norm_out = createTensor(
        ins[0].dims.size(), dtype_, ins[0].dims, false, "norm_out");
    auto norm_var =
        createTensor(tmp_dims.size(), dtype_, tmp_dims, false, "norm_var");

    std::vector<synTensor> rmsnorm_outputs;
    rmsnorm_outputs.push_back(norm_out);
    rmsnorm_outputs.push_back(norm_var);

    synStatus status = synNodeCreate(graphHandle_,
                                     rmsnorm_inputs.data(),
                                     rmsnorm_outputs.data(),
                                     rmsnorm_inputs.size(),
                                     rmsnorm_outputs.size(),
                                     &params.rmsnorm_params,
                                     sizeof(params.rmsnorm_params),
                                     guid_rmsnorm.c_str(),
                                     name_rmsnorm.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] FusedRmsMlpKernel synNodeCreate () failed = ",
             status);

    auto proj_weight = createTensor(
        ins[2].dims.size(), dtype_, ins[2].dims, true, ins[2].name);
    std::vector<int64_t> proj_dims = {
        ins[0].dims[0], ins[0].dims[1], ins[2].dims[1]};
    auto proj_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "proj_out");

    std::vector<synTensor> proj_inputs;
    proj_inputs.push_back(norm_out);
    proj_inputs.push_back(proj_weight);
    std::vector<synTensor> proj_outputs;
    proj_outputs.push_back(proj_out);

    status = synNodeCreate(graphHandle_,
                           proj_inputs.data(),
                           proj_outputs.data(),
                           proj_inputs.size(),
                           proj_outputs.size(),
                           nullptr,
                           0,
                           guid_matmul.c_str(),
                           name_proj.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "FusedRmsMlpKernel synNodeCreate () failed = %d",
             status);

    std::vector<int64_t> split_out_dims = {
        proj_dims[0], proj_dims[1], proj_dims[2] / 2};
    auto gate_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "gate_out");
    auto up_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "up_out");

    auto down_weight = createTensor(
        ins[3].dims.size(), dtype_, ins[3].dims, true, ins[3].name);

    std::vector<synTensor> split_inputs;
    split_inputs.push_back(proj_out);
    std::vector<synTensor> split_outputs;
    split_outputs.push_back(gate_out);
    split_outputs.push_back(up_out);

    status = synNodeCreate(graphHandle_,
                           split_inputs.data(),
                           split_outputs.data(),
                           split_inputs.size(),
                           split_outputs.size(),
                           &params.split_params,
                           sizeof(params.split_params),
                           guid_split.c_str(),
                           name_split.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "FusedRmsMlpKernel synNodeCreate () failed = %d",
             status);

    auto silu_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "silu_out");

    std::vector<synTensor> silu_inputs;
    silu_inputs.push_back(gate_out);
    std::vector<synTensor> silu_outputs;
    silu_outputs.push_back(silu_out);

    status = synNodeCreate(graphHandle_,
                           silu_inputs.data(),
                           silu_outputs.data(),
                           silu_inputs.size(),
                           silu_outputs.size(),
                           nullptr,
                           0,
                           guid_silu.c_str(),
                           name_silu.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "FusedRmsMlpKernel synNodeCreate () failed = %d",
             status);

    auto multi_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "multi_out");

    std::vector<synTensor> multi_inputs;
    multi_inputs.push_back(silu_out);
    multi_inputs.push_back(up_out);
    std::vector<synTensor> multi_outputs;
    multi_outputs.push_back(multi_out);

    status = synNodeCreate(graphHandle_,
                           multi_inputs.data(),
                           multi_outputs.data(),
                           multi_inputs.size(),
                           multi_outputs.size(),
                           nullptr,
                           0,
                           guid_multi.c_str(),
                           name_multi.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "FusedRmsMlpKernel synNodeCreate () failed = %d",
             status);

    auto mlp_out = createTensor(
        outs[0].dims.size(), dtype_, outs[0].dims, true, outs[0].name);

    std::vector<synTensor> down_inputs;
    down_inputs.push_back(multi_out);
    down_inputs.push_back(down_weight);
    std::vector<synTensor> down_outputs;
    down_outputs.push_back(mlp_out);

    status = synNodeCreate(graphHandle_,
                           down_inputs.data(),
                           down_outputs.data(),
                           down_inputs.size(),
                           down_outputs.size(),
                           nullptr,
                           0,
                           guid_matmul.c_str(),
                           name_down.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "FusedRmsMlpKernel synNodeCreate () failed = %d",
             status);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedRmsMlpKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& ln_scales,
                       const phi::DenseTensor& proj_weight,
                       const phi::DenseTensor& down_weight,
                       const phi::Scalar& epsilon,
                       phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  std::vector<int64_t> ln_scales_dims =
      phi::vectorize<int64_t>(ln_scales.dims());

  const phi::Scalar axis_scalar = proj_weight.dims().size() - 1;
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = proj_weight.dims().size() + axis;
  }
  FusedRmsMlpParams params;
  memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedRmsMlpParams));
  params.rmsnorm_params.epsValid = true;
  params.rmsnorm_params.eps = epsilon.to<float>();

  params.split_params = {{0}};
  params.split_params.axis = proj_weight.dims().size() - 1 - axis;

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(ln_scales);
  ct.Add(proj_weight);
  ct.Add(down_weight);
  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, FusedRmsMlpParams>(
      "FusedRmsMlpKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedRmsMlp op(op_info.datatype_);
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
void CallFusedRmsMlpKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& ln_scales,
                           const phi::DenseTensor& proj_weight,
                           const phi::DenseTensor& down_weight,
                           const phi::Scalar& epsilon,
                           phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedRmsMlpKernel<phi::dtype::bfloat16>(
        dev_ctx, x, ln_scales, proj_weight, down_weight, epsilon, out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedRmsMlpKernel");
  }
}

std::vector<paddle::Tensor> FusedRmsMlpForward(
    const paddle::Tensor& x,
    const paddle::Tensor& ln_scales,
    const paddle::Tensor& proj_weight,
    const paddle::Tensor& down_weight,
    const float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto ln_scales_tensor =
      static_cast<const phi::DenseTensor*>(ln_scales.impl().get());

  auto down_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());

  auto proj_tensor =
      static_cast<const phi::DenseTensor*>(proj_weight.impl().get());

  CallFusedRmsMlpKernel(*dev_ctx,
                        *x_tensor,
                        *ln_scales_tensor,
                        *proj_tensor,
                        *down_tensor,
                        phi::Scalar(epsilon),
                        out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedRmsMlpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& ln_scales_shape,
    const std::vector<int64_t>& proj_weight_shape,
    const std::vector<int64_t>& down_weight_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> FusedRmsMlpInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& ln_scales_dtype,
    const paddle::DataType& proj_weight_dtype,
    const paddle::DataType& down_weight_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(fused_rms_mlp)
    .Inputs({"x", "ln_scales", "proj_weight", "down_weight"})
    .Outputs({"out"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedRmsMlpForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedRmsMlpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedRmsMlpInferDtype));
