// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/gcu_kernel_funcs.h"

#include "paddle/phi/kernels/cpu/conv_util.h"

namespace custom_kernel {

constexpr char kGradVarSuffix[] = "@GRAD";
constexpr size_t kGradVarSuffixSize = 5U;

std::string GradVarName(const std::string& var_name) {
  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

std::vector<int> GetIntList(const phi::IntArray& int_array) {
  std::vector<int64_t> data = int_array.GetData();
  std::vector<int> data_ret =
      std::move(std::vector<int>(data.begin(), data.end()));
  return data_ret;
}

void UpdatePaddingAndDilation(const common::DDim& input_dims,
                              const common::DDim& filter_dims,
                              const std::string& padding_algorithm,
                              const std::vector<int>& strides,
                              std::vector<int>& paddings,     // NOLINT
                              std::vector<int>& dilations) {  // NOLINT
  // update paddings and dilations according to padding_algorithm
  common::DDim in_data_dims =
      common::slice_ddim(input_dims, 2, input_dims.size());
  common::DDim filter_data_dims =
      common::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);
  size_t expected_size = ((input_dims.size() == 5) ? 6 : 4);
  PADDLE_ENFORCE_EQ(
      paddings.size(),
      expected_size,
      phi::errors::Fatal("Paddings size should be the same as %zu after update "
                         "padding and dilation process.",
                         expected_size));
}

}  // namespace custom_kernel
