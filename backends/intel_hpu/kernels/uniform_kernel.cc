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
#include "utils/utills.h"

namespace custom_kernel {

struct uniformParams {
  ns_RandomUniform::ParamsV2 params;
};

class Uniform : public HpuOperator {
 public:
  explicit Uniform(std::string guid_prefix) : HpuOperator(guid_prefix) {}

  void AddNode(ConvertTensors& ct, uniformParams params) {
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
                                     "Uniform",
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T>
inline void UniformRealDistribution(T* data,
                                    const int64_t& size,
                                    const float& min,
                                    const float& max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<float> dist(static_cast<float>(min),
                                             static_cast<float>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(dist(*engine));
  }
}

template <typename T, typename Context>
void UniformRawKernel(const Context& dev_ctx,
                      const phi::IntArray& shape,
                      phi::DataType dtype,
                      const phi::Scalar& min,
                      const phi::Scalar& max,
                      int seed,
                      int diag_num,
                      int diag_step,
                      float diag_val,
                      phi::DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  VLOG(4) << out->dims() << ", "
          << "diag_num = " << diag_num;
  dev_ctx.template Alloc<T>(out);

  if (diag_num > 0) {
    auto size = out->numel();

    // 1. CPU implement
    phi::DenseTensor cpu_out;
    phi::DenseTensorMeta cpu_out_meta = {out->dtype(), out->dims()};
    cpu_out.set_meta(cpu_out_meta);
    T* cpu_data = dev_ctx.template HostAlloc<T>(&cpu_out);

    std::shared_ptr<std::mt19937_64> engine;
    if (seed) {
      engine = std::make_shared<std::mt19937_64>();
      engine->seed(seed);
    } else {
      engine = dev_ctx.GetGenerator()->GetCPUEngine();
    }
    UniformRealDistribution<T>(
        cpu_data, size, min.to<float>(), max.to<float>(), engine);

    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        phi::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            size));
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      cpu_data[pos] = diag_val;
    }
    // 2. CPU Copy to INTEL HPU
    TensorCopy(dev_ctx, cpu_out, false, out);
  } else {
    ConvertTensors ct;
    ct.Add(out, false);

    uniformParams params;

    if (dtype == phi::DataType::FLOAT32 || dtype == phi::DataType::FLOAT16 ||
        dtype == phi::DataType::BFLOAT16) {
      params.params.low.f = min.to<float>();
      params.params.high.f = max.to<float>();
    } else if (dtype == phi::DataType::INT32) {
      params.params.low.f = min.to<int>();
      params.params.high.f = max.to<int>();
    } else {
      phi::errors::InvalidArgument("Unsupported cast dtype %s", dtype);
    }
    params.params.seed = seed;

    std::vector<DIMS> inputs_dims = ct.GetDims();

    OpCacheOperator op_info;
    op_info.prepareOpInfo<T, uniformParams>(
        "UniformRawKernel", inputs_dims, &params);
    auto recipe = op_info.GetRecipe();

    if (recipe == nullptr) {
      Uniform op("random_uniform_fwd");
      op.AddNode(ct, params);
      op.Compile();
      op_info.setOp(op);
      recipe = op_info.GetRecipe();
    }

    RecipeRunner runner(recipe);
    auto tensors = ct.GetDeviceAddr();
    runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
  }
}

template <typename T, typename Context>
void UniformKernel(const Context& dev_ctx,
                   const phi::IntArray& shape,
                   phi::DataType dtype,
                   const phi::Scalar& min,
                   const phi::Scalar& max,
                   int seed,
                   phi::DenseTensor* out) {
  custom_kernel::UniformRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(uniform_raw,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::UniformRawKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int32_t) {}

PD_REGISTER_PLUGIN_KERNEL(uniform,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::UniformKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int32_t) {}
