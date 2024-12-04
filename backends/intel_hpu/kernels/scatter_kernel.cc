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
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& shape,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out);

struct ScatterParams {
  ns_ScatterKernel::Params params;
};

class Scatter : public HpuOperator {
 public:
  Scatter() : HpuOperator("scatter_fwd_") {}

  void AddNode(ConvertTensors& ct, ScatterParams params, bool is_inplace) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_inputs;
    synSectionHandle section_shared = nullptr;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (i == 0 && is_inplace) {
        section_shared = createSection();
        syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                          inputs[i].type,
                                          inputs[i].dims,
                                          true,
                                          inputs[i].name,
                                          section_shared));
      } else {
        syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                          inputs[i].type,
                                          inputs[i].dims,
                                          true,
                                          inputs[i].name));
      }
    }

    std::vector<synTensor> syn_outputs;
    syn_outputs.push_back(createTensor(outputs[0].dims.size(),
                                       outputs[0].type,
                                       outputs[0].dims,
                                       true,
                                       outputs[0].name,
                                       section_shared));

    guid_ = guid_ + SynDataTypeToStr(inputs[0].type);
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "Scatter",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

class ScatterAdd : public HpuOperator {
 public:
  ScatterAdd() : HpuOperator("unsorted_scatter_add_fwd_") {}

  void AddNode(ConvertTensors& ct, ScatterParams params) {
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

    guid_ = guid_ + SynDataTypeToStr(inputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "ScatterAdd",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void ScatterKernelOverwrite(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& index,
                            const phi::DenseTensor& update,
                            phi::DenseTensor* out) {
  PD_CHECK(index.dtype() == phi::DataType::INT32 ||
               index.dtype() == phi::DataType::INT64,
           "Scatter requires the index type be either int32 or int64");

  auto index_dims = phi::vectorize<int>(index.dims());
  auto update_dims = phi::vectorize<int>(update.dims());
  PD_CHECK(update_dims[0] == index_dims[0],
           "Scatter requires the 1st dim of update match the 1st dim of index");

  if (index_dims.size() == 2) {
    PD_CHECK(index_dims[1] != 1,
             "Scatter's index 2nd dim must be 1 for 2D index");
  } else if (index_dims.size() == 1) {
    index_dims.push_back(1);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Scatter requires the index type "
                                     "be either int32 or int64."));
  }

  phi::DenseTensor index_i32;
  phi::DenseTensor fake_index(index);
  phi::DenseTensor* expand_src = &fake_index;
  phi::DenseTensorMeta fake_meta({index.dtype(), {phi::make_ddim(index_dims)}});
  fake_index.set_meta(fake_meta);

  if (index.dtype() == phi::DataType::INT64) {
    index_i32.Resize(phi::make_ddim(index_dims));
    dev_ctx.template Alloc<int32_t>(&index_i32);

    custom_kernel::CastKernel<int64_t, Context>(
        dev_ctx, fake_index, phi::DataType::INT32, &index_i32);
    expand_src = &index_i32;
  }

  phi::IntArray out_shape(update_dims);
  phi::DenseTensor index_expand;
  index_expand.Resize(phi::make_ddim(update_dims));
  dev_ctx.template Alloc<int32_t>(&index_expand);

  custom_kernel::ExpandKernel<int32_t, Context>(
      dev_ctx, *expand_src, out_shape, &index_expand);

  dev_ctx.template Alloc<T>(out);
  bool is_inplace = (out->data() == x.data());

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(index_expand);
  ct.Add(update);
  ct.Add(out, false);

  OpCacheOperator op_info;
  ScatterParams params;
  params.params.axis = x.dims().size() - 1;
  std::vector<DIMS> inputs_dims = ct.GetDims();
  // need to add different nodes for inplace and non-inplace scatter
  if (is_inplace) {
    op_info.prepareOpInfo<T, ScatterParams>(
        "ScatterKernel_", inputs_dims, &params);
  } else {
    op_info.prepareOpInfo<T, ScatterParams>(
        "ScatterKernel", inputs_dims, &params);
  }
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Scatter op;
    op.AddNode(ct, params, is_inplace);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void ScatterKernelAdd(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& update,
                      phi::DenseTensor* out) {
  PD_CHECK(index.dtype() == phi::DataType::INT32 ||
               index.dtype() == phi::DataType::INT64,
           "ScatterAdd requires the index type be either int32 or int64");

  auto index_dims = phi::vectorize<int>(index.dims());
  auto update_dims = phi::vectorize<int>(update.dims());
  PD_CHECK(
      update_dims[0] == index_dims[0],
      "ScatterAdd requires the 1st dim of update match the 1st dim of index");

  if (index_dims.size() == 2) {
    PD_CHECK(index_dims[1] != 1,
             "ScatterAdd's index 2nd dim must be 1 for 2D index");
  } else if (index_dims.size() == 1) {
    index_dims.push_back(1);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Scatter requires the index type "
                                     "be either int32 or int64."));
  }

  phi::DenseTensor index_i32;
  phi::DenseTensor fake_index(index);
  phi::DenseTensor* expand_src = &fake_index;
  phi::DenseTensorMeta fake_meta({index.dtype(), {phi::make_ddim(index_dims)}});
  fake_index.set_meta(fake_meta);

  if (index.dtype() == phi::DataType::INT64) {
    index_i32.Resize(phi::make_ddim(index_dims));
    dev_ctx.template Alloc<int32_t>(&index_i32);

    custom_kernel::CastKernel<int64_t, Context>(
        dev_ctx, fake_index, phi::DataType::INT32, &index_i32);
    expand_src = &index_i32;
  }

  phi::IntArray out_shape(update_dims);
  phi::DenseTensor index_expand;
  index_expand.Resize(phi::make_ddim(update_dims));
  dev_ctx.template Alloc<int32_t>(&index_expand);

  custom_kernel::ExpandKernel<int32_t, Context>(
      dev_ctx, *expand_src, out_shape, &index_expand);

  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(index_expand);
  ct.Add(update);
  ct.Add(out, false);

  OpCacheOperator op_info;
  ScatterParams params;
  params.params.axis = x.dims().size() - 1;
  std::vector<DIMS> inputs_dims = ct.GetDims();
  op_info.prepareOpInfo<T, ScatterParams>(
      "ScatterAddKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    ScatterAdd op;

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
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& update,
                   bool overwrite,
                   phi::DenseTensor* out) {
  if (overwrite) {
    ScatterKernelOverwrite<T, Context>(dev_ctx, x, index, update, out);
  } else {
    auto value = static_cast<T>(0);

    phi::DenseTensor zero;
    phi::DenseTensorMeta zero_meta = {update.dtype(), update.dims()};
    zero.set_meta(zero_meta);
    custom_kernel::FullLikeKernel<T, Context>(
        dev_ctx, update, phi::Scalar(value), zero.dtype(), &zero);

    phi::DenseTensor x1;
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &x1);

    phi::DenseTensor x2;
    phi::DenseTensorMeta x2_meta = {x.dtype(), x.dims()};
    x2.set_meta(x2_meta);
    ScatterKernelOverwrite<T, Context>(dev_ctx, x1, index, zero, &x2);

    ScatterKernelAdd<T, Context>(dev_ctx, x2, index, update, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
