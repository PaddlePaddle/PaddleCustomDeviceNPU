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
#include "kernels/hpu_operator.h"

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

#define TOTAL_NUMBER_OF_TENSORS 1024

FLAGS_DEFINE_bool(intel_hpu_sync_execute, false, "set sync execute mode");

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;

static uint64_t cached_workspaceSize = 0;
static uint64_t cached_workspaceAddress = 0;
static uint32_t recipe_count = 0;

void HpuOperator::Compile() {
  std::string recipe_name =
      guid_ + "_" + std::to_string(recipe_count) + ".recipe";
  synStatus status =
      synGraphCompile(&recipeHandle_, graphHandle_, recipe_name.c_str(), 0);

  PD_CHECK(status == synSuccess,
           "synGraphCompile() ",
           guid_,
           " failed = ",
           status,
           " eager = ",
           is_eager_);

  VLOG(9) << " synGraphCompile =" << guid_ << ", count = " << recipe_count;
  recipe_count += 1;
  // cleanup
  status = synGraphDestroy(graphHandle_);
  LOG_IF(ERROR, status != synSuccess)
      << "synGraphDestroy() failed = " << status;

  for (auto it = tensors_.begin(); it != tensors_.end(); ++it) {
    status = synTensorDestroy(it->second);
    LOG_IF(ERROR, status != synSuccess)
        << "synTensorDestroy() failed = " << status;
  }
  for (size_t i = 0; i < sectons_.size(); i++) {
    status = synSectionDestroy(sectons_[i]);
    LOG_IF(ERROR, status != synSuccess)
        << "synSectionDestroy() failed = " << status;
  }
}

synSectionHandle HpuOperator::createSection() {
  synStatus status;
  synSectionHandle sectionHandle = nullptr;
  status = synSectionCreate(&sectionHandle, 0, graphHandle_);

  PD_CHECK(status == synSuccess, "synSectionCreate() failed = ", status);
  sectons_.push_back(sectionHandle);
  return sectionHandle;
}

synTensor HpuOperator::createTensor(unsigned dims,
                                    synDataType data_type,
                                    DIMS tensor_size,
                                    bool is_presist,
                                    std::string name,
                                    synSectionHandle section) {
  synStatus status;
  synTensorDescriptor desc{};
  // input
  desc.m_dataType = data_type;
  desc.m_dims = dims;
  desc.m_name = name.c_str();
  memset(desc.m_strides, 0, sizeof(desc.m_strides));

  for (unsigned i = 0; i < dims; ++i) {
    desc.m_sizes[i] = tensor_size[dims - 1 - i];
    VLOG(6) << "name = " << name << ", " << tensor_size[dims - 1 - i];
  }

  synSectionHandle sectionHandle = section;
  if (is_presist && sectionHandle == nullptr) {
    status = synSectionCreate(&sectionHandle, 0, graphHandle_);

    PD_CHECK(status == synSuccess, "synSectionCreate() failed = ", status);
    sectons_.push_back(sectionHandle);
  }

  synTensor tensor = nullptr;
  status = synTensorCreate(&tensor, &desc, sectionHandle, 0);
  PD_CHECK(status == synSuccess, "synTensorCreate() failed = ", status);
  tensors_.insert({name, tensor});
  return tensor;
}

void RecipeRunner::prepareTensorInfo(synRecipeHandle recipe,
                                     synLaunchTensorInfo* tensorInfo,
                                     uint32_t totalNumOfTensors) {
  const char* tensorNames[TOTAL_NUMBER_OF_TENSORS];
  uint64_t tensorIds[TOTAL_NUMBER_OF_TENSORS] = {0};
  uint32_t i = 0;

  for (i = 0; i < totalNumOfTensors; ++i) {
    tensorNames[i] = tensorInfo[i].tensorName;
  }
  synStatus status =
      synTensorRetrieveIds(recipe, tensorNames, tensorIds, totalNumOfTensors);
  PD_CHECK(status == synSuccess, "synTensorRetrieveIds() failed = ", status);
  for (i = 0; i < totalNumOfTensors; i++) {
    tensorInfo[i].tensorId = tensorIds[i];
  }
}

void RecipeRunner::Run(C_Stream stream,
                       std::map<std::string, uint64_t> tensors) {
  uint64_t request_workspace_size = 0;
  synStatus status =
      synWorkspaceGetSize(&request_workspace_size, recipeHandle_);
  PD_CHECK(status == synSuccess, "synWorkspaceGetSize() failed = ", status);

  if (request_workspace_size > cached_workspaceSize) {
    if (cached_workspaceSize != 0) {
      VLOG(6) << "workspace size changed, sync... from " << cached_workspaceSize
              << " to " << request_workspace_size;
      status = synStreamSynchronize(reinterpret_cast<synStreamHandle>(stream));
      PD_CHECK(
          status == synSuccess, "synStreamSynchronize() failed = ", status);

      status = synDeviceFree(0, cached_workspaceAddress, 0);
      PD_CHECK(status == synSuccess, "synDeviceFree() failed = ", status);
    }

    cached_workspaceSize = request_workspace_size;
    VLOG(6) << "malloc device workspace " << cached_workspaceSize;
    status = synDeviceMalloc(
        0, cached_workspaceSize, 0, 0, &cached_workspaceAddress);
    PD_CHECK(status == synSuccess, "synDeviceMalloc() failed = ", status);
  }

  VLOG(6) << "workspace size = " << cached_workspaceSize
          << ", stream = " << stream << ", recipe = " << recipeHandle_;

  std::vector<synLaunchTensorInfo> concatTensors;
  for (auto& tensor : tensors) {
    concatTensors.push_back({tensor.first.c_str(), tensor.second});
  }
  prepareTensorInfo(recipeHandle_, &concatTensors[0], concatTensors.size());
  status = synLaunch(reinterpret_cast<synStreamHandle>(stream),
                     concatTensors.data(),
                     concatTensors.size(),
                     cached_workspaceAddress,
                     recipeHandle_,
                     0);

  PD_CHECK(status == synSuccess, "synLaunch() failed = ", status);
  VLOG(6) << "synLaunch called ";
  if (FLAGS_intel_hpu_sync_execute) {
    status = synStreamSynchronize(reinterpret_cast<synStreamHandle>(stream));
    PD_CHECK(status == synSuccess, "synStreamSynchronize() failed = ", status);
  }
}
