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
#include "paddle/phi/common/amp_type_traits.h"
#include "utils/utills.h"

namespace custom_kernel {

struct GaussianParams {
  ns_RandomNormal::Params params;
};

class Gaussian : public HpuOperator {
 public:
  explicit Gaussian(std::string guid_prefix) : HpuOperator(guid_prefix) {}

  void AddNode(ConvertTensors& ct, GaussianParams params) {
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      syn_outputs.push_back(createTensor(outputs[i].dims.size(),
                                         outputs[i].type,
                                         outputs[i].dims,
                                         true,
                                         outputs[i].name));
    }

    guid_ = guid_ + "_" + SynDataTypeToStr(outputs[0].type);

    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     syn_outputs.data(),
                                     0,
                                     syn_outputs.size(),
                                     &params.params,
                                     sizeof(params.params),
                                     guid_.c_str(),
                                     "Gaussian",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void GaussianKernel(const Context& dev_ctx,
                    const phi::IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(out, false);

  GaussianParams params;
  params.params.seed = seed;
  params.params.mean = mean;
  params.params.stddev = std;

  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, GaussianParams>(
      "GaussianKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    Gaussian op("random_normal_fwd");
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

PD_REGISTER_PLUGIN_KERNEL(gaussian,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::GaussianKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float) {}
