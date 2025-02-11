// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.
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
#include "utils/utils.h"

namespace custom_kernel {

struct ReduceAnyParams {
  // ns_Reduction::ParamsV2 params;
  // std::vector<int64_t> dims;
  ns_Reduction::Params params;
  int dim;
  bool keep_dim;
};

class ReduceAny : public HpuOperator {
 public:
  ReduceAny() : HpuOperator("any_") {}

  void AddNode(ConvertTensors& ct, ReduceAnyParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::string guid_cast = "cast_i8_to_i32";
    std::string guid_sum = "reduce_sum_fwd_i32";
    std::string guid_full = "constant_i32";
    std::string guid_not_eq = "not_equal_fwd_i32";
    std::string guid_reshape = "reshape";

    std::string name_cast = guid_ + "cast";
    std::string name_sum = guid_ + "sum";
    std::string name_full = guid_ + "full";
    std::string name_not_eq = guid_ + "not_eq";
    std::string name_reshape = guid_ + "reshape";

    synStatus status = synFail;

    std::vector<synTensor> x_i8;
    x_i8.push_back(createTensor(inputs[0].dims.size(),
                                inputs[0].type,
                                inputs[0].dims,
                                true,
                                inputs[0].name));
    std::vector<synTensor> x_i32;
    x_i32.push_back(createTensor(inputs[0].dims.size(),
                                 syn_type_int32,
                                 inputs[0].dims,
                                 false,
                                 "x_cast"));

    status = synNodeCreate(graphHandle_,
                           x_i8.data(),
                           x_i32.data(),
                           x_i8.size(),
                           x_i32.size(),
                           nullptr,
                           0,
                           guid_cast.c_str(),
                           name_cast.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (any/cast) failed = ",
             status);

    auto reshape_dims = inputs[0].dims;
    auto reduced_dims = inputs[0].dims;
    if (params.dim == -1) {
      reshape_dims =
          std::vector<int64_t>({static_cast<int64_t>(inputs[0].num_elements)});
      reduced_dims = std::vector<int64_t>({1});
    } else {
      reduced_dims[params.dim] = 1;
    }
    std::vector<synTensor> reduce_inputs;

    reduce_inputs.push_back(createTensor(
        reshape_dims.size(), syn_type_int32, reshape_dims, false, "reduce_in"));

    status = synNodeCreate(graphHandle_,
                           x_i32.data(),
                           reduce_inputs.data(),
                           x_i32.size(),
                           reduce_inputs.size(),
                           nullptr,
                           0,
                           guid_reshape.c_str(),
                           name_reshape.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (any/reshape) failed = ",
             status);

    std::vector<synTensor> reduce_outputs;

    reduce_outputs.push_back(createTensor(reduced_dims.size(),
                                          syn_type_int32,
                                          reduced_dims,
                                          false,
                                          "reduce_out"));

    status = synNodeCreate(graphHandle_,
                           reduce_inputs.data(),
                           reduce_outputs.data(),
                           reduce_inputs.size(),
                           reduce_outputs.size(),
                           &params.params,
                           sizeof(params.params),
                           guid_sum.c_str(),
                           name_sum.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (any/sum) failed = ",
             status);

    std::vector<synTensor> reshape_outputs;
    auto reshape_out = createTensor(outputs[0].dims.size(),
                                    syn_type_int32,
                                    outputs[0].dims,
                                    false,
                                    "reshape_out");
    reshape_outputs.push_back(reshape_out);
    status = synNodeCreate(graphHandle_,
                           reduce_outputs.data(),
                           reshape_outputs.data(),
                           reduce_outputs.size(),
                           reshape_outputs.size(),
                           nullptr,
                           0,
                           guid_reshape.c_str(),
                           name_reshape.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (any/reshape) failed = ",
             status);

    std::vector<int64_t> zero_dims = {1};

    std::vector<synTensor> zero_out;
    auto zero_tensor = createTensor(
        zero_dims.size(), syn_type_int32, zero_dims, false, "zero_tensor");
    zero_out.push_back(zero_tensor);

    ns_ConstantKernel::Params zeroParams;
    zeroParams.constant.i = 0;
    status = synNodeCreate(graphHandle_,
                           nullptr,
                           zero_out.data(),
                           0,
                           zero_out.size(),
                           &zeroParams,
                           sizeof(zeroParams),
                           guid_full.c_str(),
                           name_full.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (any/full) failed = ",
             status);

    std::vector<synTensor> syn_inputs;
    syn_inputs.push_back(reshape_out);
    syn_inputs.push_back(zero_tensor);

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name));

    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           syn_outputs.data(),
                           syn_inputs.size(),
                           syn_outputs.size(),
                           nullptr,
                           0,
                           guid_not_eq.c_str(),
                           name_not_eq.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synNodeCreate (any/not_equal) failed = ",
             status);
  }
};

template <typename T, typename Context>
void AnyKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);

  ReduceAnyParams params;
  // params.params.keepDim = keep_dim;
  // params.params.reductionDimensionMask = 0;
  params.keep_dim = keep_dim;
  params.dim = -1;
  params.params.reductionDimension = 0;

  if (dims.size() != 0) {
    PD_CHECK(dims.size() == 1,
             "Any / Reduction only support axis = 1 but got ",
             dims.size());
    auto rank = static_cast<int32_t>(x.dims().size());
    // for (auto dim : dims) {
    //   if(dim < 0)
    //     dim = rank + dim;
    //   params.dims.push_back(dim);
    //   params.params.reductionDimensionMask |= (1 << (rank - 1 - dim));
    // }
    auto dim = CanonicalAxis(static_cast<int64_t>(dims[0]),
                             static_cast<int64_t>(rank));
    params.dim = dim;
    params.params.reductionDimension = rank - 1 - dim;
  }

  OpCacheOperator op_info;
  std::vector<DIMS> in_out_dims = ct.GetDims();
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(), out_dims.begin(), out_dims.end());
  op_info.prepareOpInfo<T, ReduceAnyParams>("AnyKernel", in_out_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    ReduceAny op;
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

PD_REGISTER_PLUGIN_KERNEL(
    any, intel_hpu, ALL_LAYOUT, custom_kernel::AnyKernel, bool) {}
