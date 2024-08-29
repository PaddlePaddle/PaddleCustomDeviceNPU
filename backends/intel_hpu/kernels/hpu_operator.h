#include <assert.h>

#include <memory>

#include "funcs.h"
#include "glog/logging.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/extension.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/hpu_helper.h"

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;

static uint64_t cached_workspaceSize = 0;
static uint64_t cached_workspaceAddress = 0;
static uint32_t recipe_count = 0;

class HpuOperator {
 public:
  HpuOperator(const std::string guid) : guid_(guid) {
    synStatus status = synGraphCreate(&graphHandle_, synDeviceGaudi2);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synGraphCreate() failed = %d", status);
  }

  void Compile() {
    synStatus status = synGraphCompile(
        &recipeHandle_,
        graphHandle_,
        (guid_ + "_" + std::to_string(recipe_count) + ".recipe").c_str(),
        0);

    PD_CHECK(status == synSuccess,
             "[RUNTIME] synGraphCompile() failed = %d",
             status);

    LOG(INFO) << " synGraphCompile =" << guid_ << ", count = " << recipe_count;
    recipe_count += 1;
  }

  void prepareTensorInfo(synRecipeHandle recipe,
                         synLaunchTensorInfo* tensorInfo,
                         uint32_t totalNumOfTensors) {
    const char* tensorNames[totalNumOfTensors];
    uint64_t tensorIds[totalNumOfTensors];
    uint32_t i = 0;

    for (i = 0; i < totalNumOfTensors; ++i) {
      tensorNames[i] = tensorInfo[i].tensorName;
    }
    assert(synTensorRetrieveIds(
               recipe, tensorNames, tensorIds, totalNumOfTensors) ==
           synSuccess);
    for (i = 0; i < totalNumOfTensors; i++) {
      tensorInfo[i].tensorId = tensorIds[i];
    }
  }

  void Execute(C_Stream stream, std::map<std::string, uint64_t>& tensors) {
    synStatus status;

    uint64_t request_workspace_size = 0;
    status = synWorkspaceGetSize(&request_workspace_size, recipeHandle_);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synWorkspaceGetSize() failed = %d",
             status);

    if (request_workspace_size > cached_workspaceSize) {
      if (cached_workspaceSize != 0) {
        status = synDeviceFree(0, cached_workspaceAddress, 0);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synDeviceFree() failed = %d",
                 status);
      }

      cached_workspaceSize = request_workspace_size;
      // malloc the new one
      LOG(INFO) << "malloc device workspace " << cached_workspaceSize;
      status = synDeviceMalloc(
          0, cached_workspaceSize, 0, 0, &cached_workspaceAddress);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synDeviceMalloc() failed = %d",
               status);
    }

    LOG(INFO) << "workspace size = " << cached_workspaceSize;

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

    PD_CHECK(status == synSuccess, "[RUNTIME] synLaunch() failed = %d", status);
    status = synStreamSynchronize(reinterpret_cast<synStreamHandle>(stream));
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synStreamSynchronize() failed = %d",
             status);
  }

  virtual ~HpuOperator() {
    synStatus status = synGraphDestroy(graphHandle_);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synGraphDestroy() failed = %d",
             status);

    for (auto it = tensors_.begin(); it != tensors_.end(); ++it) {
      status = synTensorDestroy(it->second);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synTensorDestroy() failed = %d",
               status);
    }
    for (size_t i = 0; i < sectons_.size(); i++) {
      status = synSectionDestroy(sectons_[i]);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synTensorDestroy() failed = %d",
               status);
    }
  }

  // protected:
  synTensor createTensor(unsigned dims,
                         synDataType data_type,
                         DIMS tensor_size, /*const unsigned* tensor_size,*/
                         bool is_presist,
                         std::string name) {
    synStatus status;
    synTensorDescriptor desc{};
    // input
    desc.m_dataType = data_type;
    desc.m_dims = dims;
    desc.m_name = name.c_str();
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i) {
      desc.m_sizes[i] = tensor_size[dims - 1 - i];
      LOG(INFO) << "name = " << name << ", " << tensor_size[dims - 1 - i];
    }

    synSectionHandle sectionHandle = nullptr;
    if (is_presist) {
      status = synSectionCreate(&sectionHandle, 0, graphHandle_);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synSectionCreate() failed = %d",
               status);
      sectons_.push_back(sectionHandle);
    }

    synTensor tensor = nullptr;
    status = synTensorCreate(&tensor, &desc, sectionHandle, 0);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synTensorCreate() failed = %d",
             status);
    tensors_.insert({name, tensor});
    return tensor;
  }

 public:
  synRecipeHandle GetRecipe() { return recipeHandle_; }

  std::string guid_;
  synGraphHandle graphHandle_;
  synRecipeHandle recipeHandle_;
  std::vector<synSectionHandle> sectons_;

  std::map<std::string, synTensor> tensors_;
};

class RecipeRunner {
 public:
  RecipeRunner(synRecipeHandle h) : recipeHandle_(h) {}
  ~RecipeRunner() {}

  void prepareTensorInfo(synRecipeHandle recipe,
                         synLaunchTensorInfo* tensorInfo,
                         uint32_t totalNumOfTensors) {
    const char* tensorNames[totalNumOfTensors];
    uint64_t tensorIds[totalNumOfTensors];
    uint32_t i = 0;

    for (i = 0; i < totalNumOfTensors; ++i) {
      tensorNames[i] = tensorInfo[i].tensorName;
    }
    synStatus status =
        synTensorRetrieveIds(recipe, tensorNames, tensorIds, totalNumOfTensors);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synTensorRetrieveIds() failed = %d",
             status);
    for (i = 0; i < totalNumOfTensors; i++) {
      tensorInfo[i].tensorId = tensorIds[i];
    }
  }

  void Run(C_Stream stream, std::map<std::string, uint64_t>& tensors) {
    uint64_t request_workspace_size = 0;
    synStatus status =
        synWorkspaceGetSize(&request_workspace_size, recipeHandle_);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synWorkspaceGetSize() failed = %d",
             status);

    if (request_workspace_size > cached_workspaceSize) {
      if (cached_workspaceSize != 0) {
        status = synDeviceFree(0, cached_workspaceAddress, 0);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synDeviceFree() failed = " << status;
      }

      cached_workspaceSize = request_workspace_size;
      LOG(INFO) << "malloc device workspace " << cached_workspaceSize;
      status = synDeviceMalloc(
          0, cached_workspaceSize, 0, 0, &cached_workspaceAddress);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synDeviceMalloc() failed = %d",
               status);
    }

    LOG(INFO) << "workspace size = " << cached_workspaceSize;

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

    PD_CHECK(status == synSuccess, "[RUNTIME] synLaunch() failed = %d", status);
    status = synStreamSynchronize(reinterpret_cast<synStreamHandle>(stream));
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synStreamSynchronize() failed = %d",
             status);
  }

 protected:
  synRecipeHandle recipeHandle_;
};