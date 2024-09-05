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
#include "funcs.h"
#include "hpu_operator.h"
#include "perf_lib_layer_params.h"
#include "utils/utills.h"

namespace custom_kernel {

class Expand : public HpuOperator {
 public:
  Expand() : HpuOperator("broadcast") {}

  void AddNode(ConvertTensors& ct) {
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
    
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs.data(),
                                     syn_outputs.data(),
                                     syn_inputs.size(),
                                     syn_outputs.size(),
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     "Expand",
                                     nullptr,
                                     nullptr);
    PD_CHECK( status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void ExpandKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& shape,
                  phi::DenseTensor* out) {
  VLOG(6) << "HPU ExpandKernel";

  auto expand_shape = shape.GetData();
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  auto diff = expand_shape.size() - x_dims.size();
  x_dims.insert(x_dims.begin(), diff, 1);

  std::vector<int> final_expand_shape(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        phi::errors::InvalidArgument("The expanded size cannot be zero."));

    if (i < diff) {  // expand_shape = [3, 4, -1, -1], x = [10, 2] -->
                     // final_expand_shape = [3, 4, 10, 2]
      PADDLE_ENFORCE_GT(expand_shape[i],
                        0,
                        phi::errors::InvalidArgument(
                            "The expanded size (%d) for non-existing "
                            "dimensions must be positive for expand_v2 op",
                            expand_shape[i]));

      final_expand_shape[i] = expand_shape[i];
    } else if (expand_shape[i] >
               0) {  // expand_shape = [3, 4, 10, 2], x = [10, 1] -->
                     // final_expand_shape = [3, 4, 10, 2]
      if (x_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(x_dims[i],
                          expand_shape[i],
                          phi::errors::InvalidArgument(
                              "The value (%d) of the non-singleton dimensions "
                              "does not much the corresponding value (%d) in "
                              "shape for expand_v2 op.",
                              x_dims[i],
                              expand_shape[i]));

        final_expand_shape[i] = expand_shape[i];
      } else {
        final_expand_shape[i] = expand_shape[i];
      }
    } else {  // expand_shape = [3, 4, -1, -1], x = [10, 2] -->
              // final_expand_shape = [3, 4, 10, 2]
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          phi::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));

      final_expand_shape[i] = x_dims[i];
    }
  }

  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument(
          "The rank of the input 'x' for expand_v2 op must be positive, "
          "but the value received is %d.",
          rank));

  auto shape_size = final_expand_shape.size();
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      phi::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2 op must "
          "be "
          "greater than or equal to the rank (%d) of the input 'x'.",
          shape_size,
          rank));

  out->Resize(phi::make_ddim(final_expand_shape));
  dev_ctx.template Alloc<T>(out);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(out, false);
  std::vector<DIMS> in_out_dims = ct.GetDims(true);
  std::vector<DIMS> out_dims = ct.GetDims(false);
  in_out_dims.insert(in_out_dims.end(),out_dims.begin(),out_dims.end());

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("broadcast", in_out_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  // const int siz_ar = op_info.key_creator_.GetKey().size();
  // for (int i = 0; i < siz_ar; ++i)
  //   printf("%d ", op_info.key_creator_.GetKey()[i]);
  // std::cout << std::endl;
  
  if (recipe == nullptr) {
    Expand op;

    op.AddNode(ct);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}


}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::ExpandKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}
