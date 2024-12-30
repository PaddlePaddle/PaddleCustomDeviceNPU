// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef BACKENDS_INTEL_HPU_KERNELS_HPU_OPERATOR_H_
#define BACKENDS_INTEL_HPU_KERNELS_HPU_OPERATOR_H_

#include <assert.h>

#include <memory>

#include "glog/logging.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/extension.h"
#include "utils/hpu_helper.h"

class HpuOperator {
 public:
  explicit HpuOperator(const std::string guid, bool is_eager = true)
      : guid_(guid), is_eager_(is_eager) {
    if (is_eager_) {
      synStatus status = synGraphCreateEager(&graphHandle_, synDeviceGaudi2);
      PD_CHECK(status == synSuccess,
               "synGraphCreateEager() ",
               guid_,
               " failed = ",
               status);
    } else {
      synStatus status = synGraphCreate(&graphHandle_, synDeviceGaudi2);
      PD_CHECK(status == synSuccess,
               "synGraphCreate() ",
               guid_,
               " failed = ",
               status);
    }
  }

  void Compile();
  virtual ~HpuOperator() {}
  synSectionHandle createSection();
  synTensor createTensor(unsigned dims,
                         synDataType data_type,
                         DIMS tensor_size,
                         bool is_presist,
                         std::string name,
                         synSectionHandle section = nullptr);

 public:
  synRecipeHandle GetRecipe() { return recipeHandle_; }

 protected:
  std::string guid_;
  synGraphHandle graphHandle_;
  synRecipeHandle recipeHandle_;
  std::vector<synSectionHandle> sectons_;
  bool is_eager_;

  std::map<std::string, synTensor> tensors_;
};

class RecipeRunner {
 public:
  explicit RecipeRunner(synRecipeHandle h) : recipeHandle_(h) {}
  ~RecipeRunner() {}

  void prepareTensorInfo(synRecipeHandle recipe,
                         synLaunchTensorInfo* tensorInfo,
                         uint32_t totalNumOfTensors);

  void Run(C_Stream stream, std::map<std::string, uint64_t> tensors);

 protected:
  synRecipeHandle recipeHandle_;
};

#endif  // BACKENDS_INTEL_HPU_KERNELS_HPU_OPERATOR_H_
