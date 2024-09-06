/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utills.h"

namespace custom_kernel {

struct ArgMinMaxParams {
  ns_Reduction::Params params;
  phi::DataType type;
};

class ArgMaxMinOperator : public HpuOperator {
 public:
  ArgMaxMinOperator(std::string guid_prefix, std::string node_name)
      : HpuOperator(guid_prefix), pName_(node_name) {}
  void AddNode(const std::vector<DIMS>& ins,
               const std::vector<DIMS>& outs,
               synDataType datatype,
               ArgMinMaxParams params) {
    assert(ins.size() == 1 && "input size should be 1");
    assert(outs.size() == 1 && "output size should be 1");

    synTensor inputs[ins.size()] = {
        createTensor(ins[0].size(), datatype, ins[0], true, "input")};
    synDataType out_datatype = syn_type_int32;
    if (params.type == phi::DataType::INT64) {
      out_datatype = syn_type_int64;
    }
    synTensor outputs[outs.size()] = {
        createTensor(outs[0].size(), out_datatype, outs[0], true, "output")};

    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     ins.size(),
                                     outs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     pName_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
  std::string pName_;
};

template <typename T, typename Context>
void doArgMaxMinTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       int axis,
                       bool arg_max,
                       phi::DataType out_datatype,
                       phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn argmax/argmin kernel";

  std::vector<int64_t> x_dims = phi::vectorize<int64_t>(x.dims());
  std::vector<int64_t> out_dims = phi::vectorize<int64_t>(out->dims());
  if (!out_dims.size()) {
    out_dims = {1};
  }

  std::string guid_prefix;
  std::string node_name;
  if (arg_max) {
    guid_prefix = "argmax_fwd";
    node_name = "ArgMax_op";
  } else {
    guid_prefix = "argmin_fwd";
    node_name = "ArgMin_op";
  }

  OpCacheOperator op_info;
  ArgMinMaxParams params;
  params.params.reductionDimension = axis;
  params.type = out_datatype;
  op_info.prepareOpInfo<T, ArgMinMaxParams>(guid_prefix, {x_dims}, &params);

  // const int siz_ar = op_info.key_creator_.GetKey().size();
  // for (int i = 0; i < siz_ar; ++i)
  //   printf("%d ", op_info.key_creator_.GetKey()[i]);
  // std::cout << std::endl;

  auto recipe = op_info.GetRecipe();
  if (recipe == nullptr) {
    // compile
    ArgMaxMinOperator op(op_info.guid_, node_name);
    op.AddNode({x_dims}, {out_dims}, op_info.datatype_, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  // runtime
  std::map<std::string, uint64_t> tensors;
  auto out_addr = out->data();
  tensors["input"] = reinterpret_cast<uint64_t>(x.data<T>());
  tensors["output"] = reinterpret_cast<uint64_t>(out_addr);

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void ArgMaxMin(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& axis,
               bool keepdims,
               bool flatten,
               phi::DataType dtype,
               bool arg_max,
               phi::DenseTensor* out) {
  int axis_ = axis.to<int>();
  if (x.numel() == 0) return;

  PADDLE_ENFORCE_EQ(
      (dtype == phi::DataType::INT64 || dtype == phi::DataType::INT32),
      true,
      phi::errors::InvalidArgument(
          "The attribute of dtype in argmax op must be [%s] or [%s], "
          "but received [%s]",
          phi::DataType::INT64,
          phi::DataType::INT32,
          dtype));

  if (dtype == phi::DataType::INT32) {
    dev_ctx.template Alloc<int32_t>(out);
  } else {
    dev_ctx.template Alloc<int64_t>(out);
    // PADDLE_THROW(phi::errors::InvalidArgument(
    //     "Intel HPU only support the output's dtype is int32."));
  }

  auto x_dims = x.dims().size();
  if (axis_ < 0) {
    axis_ += x_dims;
  }

  phi::DenseTensor transformed_x;
  std::vector<int> out_shape;
  if (flatten) {
    phi::DenseTensor transformed_x(x);
    transformed_x.Resize(phi::make_ddim({x.numel()}));
    out->Resize(phi::make_ddim({x.numel()}));
    axis_ = 0;
    out_shape = {1};
  } else {
    transformed_x = x;
    std::vector<int> out_keepdims_shape;
    auto in_dims = x.dims();
    auto vec_in_dims = phi::vectorize<int>(in_dims);

    for (int i = 0; i < x_dims; ++i) {
      if (i == axis_) {
        out_keepdims_shape.push_back(1);
        continue;
      }
      out_shape.push_back(vec_in_dims[i]);
      out_keepdims_shape.push_back(vec_in_dims[i]);
    }
    axis_ = x_dims - 1 - axis_;

    out->Resize(phi::make_ddim(out_keepdims_shape));
  }
  custom_kernel::doArgMaxMinTensor<T, Context>(
      dev_ctx, transformed_x, axis_, arg_max, dtype, out);
  if (!keepdims) {
    out->Resize(phi::make_ddim(out_shape));
  }
}

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  custom_kernel::ArgMaxMin<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, false, out);
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  custom_kernel::ArgMaxMin<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, true, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argmin,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(argmax,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          int,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
