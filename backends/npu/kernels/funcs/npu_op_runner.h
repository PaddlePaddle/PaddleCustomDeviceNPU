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

#pragma once

#include <dlfcn.h>
#include <hccl/hccl.h>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "glog/logging.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_op_prepare.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/extension.h"
#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"

using NPUAttribute = paddle::variant<paddle::blank,
                                     int,
                                     float,
                                     std::string,
                                     std::vector<int>,
                                     std::vector<float>,
                                     std::vector<std::string>,
                                     bool,
                                     std::vector<bool>,
                                     int64_t,
                                     std::vector<int64_t>,
                                     std::vector<double>,
                                     std::vector<std::vector<int64_t>>>;

using NPUAttributeMap = std::unordered_map<std::string, NPUAttribute>;

// clang-format off
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef aclTensor* (*_aclCreateTensor)(const int64_t* view_dims,
                                       uint64_t view_dims_num,
                                       aclDataType data_type,
                                       const int64_t* stride,
                                       int64_t offset,
                                       aclFormat format,
                                       const int64_t* storage_dims,
                                       uint64_t storage_dims_num,
                                       void* tensor_data);
typedef aclScalar* (*_aclCreateScalar)(void* value,
                                       aclDataType data_type);
typedef aclIntArray* (*_aclCreateIntArray)(const int64_t* value,
                                           uint64_t size);
typedef aclFloatArray* (*_aclCreateFloatArray)
    (const float* value, uint64_t size);
typedef aclBoolArray* (*_aclCreateBoolArray)(const bool* value,
                                             uint64_t size);
typedef aclTensorList* (*_aclCreateTensorList)
        (const aclTensor* const* value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor* tensor);
typedef int (*_aclDestroyScalar)(const aclScalar* scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray* array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray* array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray* array);
typedef int (*_aclDestroyTensorList)(const aclTensorList* array);

typedef int (*InitHugeMemThreadLocal)(void*, bool);
typedef void (*UnInitHugeMemThreadLocal)(void*, bool);
typedef void (*ReleaseHugeMem)(void*, bool);
typedef aclOpExecutor*(*PTAGetExecCache) (uint64_t, uint64_t*);

#define GET_OP_API_FUNC(apiName)                           \
  reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName)) \

inline const char* GetOpApiLibName(void) {
  return "libopapi.so";
}

inline const char* GetCustOpApiLibName(void) {
  return "libcust_opapi.so";
}

inline void* GetOpApiFuncAddrInLib(void* handler,
                                   const char* libName,
                                   const char* apiName) {
  auto funcAddr = dlsym(handler, apiName);
  if (funcAddr == nullptr) {
    printf("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
  }
  return funcAddr;
}

inline void* GetOpApiLibHandler(const char* libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  if (handler == nullptr && libName != "libcust_opapi.so") {
    printf("dlopen %s failed, error:%s.", libName, dlerror());
  }
  return handler;
}

inline void* GetOpApiFuncAddr(const char* apiName) {
  static auto custOpApiHandler = GetOpApiLibHandler(GetCustOpApiLibName());
  if (custOpApiHandler != nullptr) {
    auto funcAddr =
        GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
  if (opApiHandler == nullptr) {
    return nullptr;
  }
  return GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
}

inline void Release(aclTensor* p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

inline void Release(aclScalar* p) {
  static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
  if (aclDestroyScalar == nullptr) {
    return;
  }
  aclDestroyScalar(p);
}

inline void Release(aclIntArray* p) {
  static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
  if (aclDestroyIntArray == nullptr) {
    return;
  }
  aclDestroyIntArray(p);
}

inline void Release(aclBoolArray* p) {
  static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
  if (aclDestroyBoolArray == nullptr) {
    return;
  }
  aclDestroyBoolArray(p);
}

inline void Release(aclTensorList* p) {
  static const auto aclDestroyTensorList =
    GET_OP_API_FUNC(aclDestroyTensorList);
  if (aclDestroyTensorList == nullptr) {
    return;
  }

  aclDestroyTensorList(p);
}

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(const Tuple& t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template <std::size_t N>
inline aclBoolArray* ConvertType(const std::array<bool, N> &value) {
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
      return nullptr;
    }
    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclScalar* ConvertType(const phi::Scalar& at_scalar) {
  static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
  if (aclCreateScalar == nullptr) {
    return nullptr;
  }
  auto scalar_data_type = at_scalar.dtype();
  aclDataType acl_data_type = ConvertToNpuDtype(scalar_data_type);
  aclScalar* acl_scalar = nullptr;
  switch (scalar_data_type) {
    case phi::DataType::FLOAT32: {
      float value = at_scalar.to<float>();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case phi::DataType::FLOAT64: {
      double value = at_scalar.to<double>();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case phi::DataType::BOOL: {
      bool value = at_scalar.to<bool>();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case phi::DataType::INT32: {
      int value = at_scalar.to<int>();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    case phi::DataType::INT64: {
      int64_t value = at_scalar.to<int64_t>();
      acl_scalar = aclCreateScalar(&value, acl_data_type);
      break;
    }
    default:
      acl_scalar = nullptr;
      break;
  }
  return acl_scalar;
}

inline aclTensor* ConvertType(const phi::DenseTensor& at_tensor) {
  // create aclDataBuffer
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (at_tensor.numel() == 0 && (at_tensor.dims().size() < 0)) {
    return nullptr;
  }
  auto at_tensor_dtype = at_tensor.dtype();

  if (at_tensor_dtype == phi::DataType::FLOAT64) {
    VLOG(2) << "Kernel is running on the AICPU."
            << "For better performance. use other dtypes.";
  }
  auto acl_data_type = ConvertToNpuDtype(at_tensor_dtype);
  const auto dimNum =
      at_tensor.dims().size() == 0 ? 1 : at_tensor.dims().size();
  std::vector<int64_t> storageDims(dimNum - 1);
  if (acl_data_type != ACL_STRING) {
    storageDims.push_back(at_tensor.numel() * sizeof(at_tensor_dtype));
  }
  aclFormat format = ACL_FORMAT_ND;
  switch (dimNum) {
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  auto origin_dims = phi::vectorize(at_tensor.dims());
  auto origin_strides = phi::vectorize(at_tensor.strides());

  auto acl_tensor = aclCreateTensor(origin_dims.data(),
                                    origin_dims.size(),
                                    acl_data_type,
                                    origin_strides.data(),
                                    0,
                                    format,
                                    origin_dims.data(),
                                    storageDims.size(),
                                    const_cast<void*>(at_tensor.data()));
  return acl_tensor;
}

template <typename T>
inline aclTensor* ConvertType(const std::pair<T*,
  std::vector<int64_t>>& tp_tensor) {
  // Ptr ConvertType to aclTensor
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    LOG(WARNING) << "aclCreateTensor execute failed, return nullptr.";
    return nullptr;
  }
  if (tp_tensor.first == nullptr || tp_tensor.second.size() < 1) {
    LOG(WARNING) << "received nullptr or empty tensor, return nullptr.";
    return nullptr;
  }
  auto at_tensor_dtype = phi::DataType::FLOAT16;
  if (std::is_same<T, const phi::float16>::value ||
    std::is_same<T, phi::float16>::value) {
    at_tensor_dtype = phi::DataType::FLOAT16;
  } else if (std::is_same<T, const int64_t>::value ||
    std::is_same<T, int64_t>::value) {
    at_tensor_dtype = phi::DataType::INT64;
  } else if (std::is_same<T, const int32_t>::value ||
    std::is_same<T, int32_t>::value) {
    at_tensor_dtype = phi::DataType::INT32;
  } else if (std::is_same<T, const int8_t>::value ||
    std::is_same<T, int8_t>::value) {
    at_tensor_dtype = phi::DataType::INT8;
  }
  auto acl_data_type = ConvertToNpuDtype(at_tensor_dtype);
  const auto dimNum =
    tp_tensor.second.size() == 0 ? 1 : tp_tensor.second.size();
  std::vector<int64_t> storageDims(dimNum - 1);
  int64_t tp_tensor_numel = 1;
  for (auto &num : tp_tensor.second) {
    tp_tensor_numel *= num;
  }
  if (acl_data_type != ACL_STRING) {
    storageDims.push_back(tp_tensor_numel * sizeof(at_tensor_dtype));
  }
  aclFormat format = ACL_FORMAT_ND;
  switch (dimNum) {
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }

  std::vector<int64_t> origin_dims;
  std::vector<int64_t> origin_strides;
  if (tp_tensor.second.size() == 2) {
    origin_dims = phi::vectorize({tp_tensor.second[0], tp_tensor.second[1]});
    origin_strides = phi::vectorize({tp_tensor.second[1], 1});
  } else if (tp_tensor.second.size() == 1) {
    origin_dims = phi::vectorize({tp_tensor.second[0]});
    origin_strides = phi::vectorize({1});
  } else {
    PADDLE_ENFORCE_GE(
        tp_tensor.second.size(), 2,
        phi::errors::InvalidArgument("Only support 1-d and 2-d input"));
        return nullptr;
  }
  auto acl_tensor = aclCreateTensor(origin_dims.data(),
                                    origin_dims.size(),
                                    acl_data_type,
                                    origin_strides.data(),
                                    0,
                                    format,
                                    origin_dims.data(),
                                    storageDims.size(),
                                    const_cast<void*>(
                                      static_cast<const void*>(
                                        tp_tensor.first)));
  return acl_tensor;
}

inline aclTensorList *ConvertType(
  const std::vector<const phi::DenseTensor*> &phi_tensor_list) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  std::vector<const aclTensor *> tensor_list(phi_tensor_list.size());
  for (size_t i = 0; i < phi_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(*phi_tensor_list[i]);
  }
  auto acl_tensor_list = aclCreateTensorList(tensor_list.data(),
                                             tensor_list.size());
  return acl_tensor_list;
}

inline aclTensorList *ConvertType(
  const std::vector<phi::DenseTensor*> &phi_tensor_list) {
  static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
  if (aclCreateTensorList == nullptr) {
    return nullptr;
  }

  std::vector<const aclTensor *> tensor_list(phi_tensor_list.size());
  for (size_t i = 0; i < phi_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(*phi_tensor_list[i]);
  }
  auto acl_tensor_list = aclCreateTensorList(tensor_list.data(),
                                             tensor_list.size());
  return acl_tensor_list;
}

inline aclIntArray* ConvertType(const std::vector<int64_t>& at_array) {
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }
  auto array = aclCreateIntArray(at_array.data(), at_array.size());
  return array;
}

inline aclIntArray *ConvertType(const phi::IntArray &phi_array) {
  static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
  if (aclCreateIntArray == nullptr) {
    return nullptr;
  }
  std::vector<int64_t> temp_array(
    phi_array.GetData().begin(), phi_array.GetData().end());
  auto array = aclCreateIntArray(temp_array.data(), temp_array.size());
  return array;
}

template <typename T>
T ConvertType(T value) {
  return value;
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args) {
  return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr,
                        std::index_sequence<I...>) {
  typedef int (*OpApiFunc)
    (typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr,
    std::make_index_sequence<size>{});
}

inline void SetDeterministic() {
#if (CANN_VERSION_CODE >= 700000)
  bool env_deterministic_flag = std::getenv("npu_deterministic") != nullptr && \
    std::string(std::getenv("npu_deterministic")) == "true";
  static std::once_flag deterministic_flag;
  std::call_once(deterministic_flag, [&]() {
    if (env_deterministic_flag) {
      aclSetCompileopt(ACL_OP_DETERMINISTIC, "1");
      aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1);
      HcclConfigValue configValue = {1};
      HcclSetConfig(HcclConfig::HCCL_DETERMINISTIC, configValue);
    }
  });
#endif
}

#define EXEC_NPU_CMD(aclnn_api, dev_ctx, ...)                             \
  do {                                                                    \
    VLOG(1) << "NpuAclnnOpRunner: " << #aclnn_api;                        \
    SetDeterministic();                                                   \
    static const auto getWorkspaceSizeFuncAddr =                          \
    GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);       \
    static const auto initMemAddr =                                       \
      GetOpApiFuncAddr("InitHugeMemThreadLocal");                         \
    static const auto unInitMemAddr =                                     \
      GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                       \
    static const auto releaseMemAddr =                                    \
      GetOpApiFuncAddr("ReleaseHugeMem");                                 \
    auto acl_stream = (dev_ctx).stream();                                 \
    uint64_t workspace_size = 0;                                          \
    uint64_t* workspace_size_addr = &workspace_size;                      \
    aclOpExecutor* executor = nullptr;                                    \
    aclOpExecutor** executor_addr = &executor;                            \
    InitHugeMemThreadLocal initMemFunc =                                  \
      reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);              \
    UnInitHugeMemThreadLocal unInitMemFunc =                              \
      reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);          \
    if (initMemFunc) {                                                    \
      initMemFunc(nullptr, false);                                        \
    }                                                                     \
    auto converted_params =                                               \
      ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);      \
    static auto getWorkspaceSizeFunc =                                    \
      ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);     \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params); \
    void* workspace_addr = nullptr;                                       \
    if (workspace_size != 0) {                                            \
      phi::Allocator::AllocationPtr workspace_tensor =                    \
        const_cast<phi::Allocator &>((dev_ctx).GetAllocator()).           \
          Allocate(workspace_size);                                       \
      workspace_addr = const_cast<void*>(workspace_tensor->ptr());        \
    }                                                                     \
    if (unInitMemFunc) {                                                  \
      unInitMemFunc(nullptr, false);                                      \
    }                                                                     \
    typedef int                                                           \
      (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);   \
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);     \
    auto api_ret =                                                        \
      opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
    ReleaseConvertTypes(converted_params);                                \
    ReleaseHugeMem releaseMemFunc =                                       \
      reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
    if (releaseMemFunc) {                                                 \
      releaseMemFunc(nullptr, false);                                     \
    }                                                                     \
  } while (false)

#define DO_COMPATIBILITY(aclnn_api, originCallExpression)                 \
  do {                                                                    \
    static const auto getWorkspaceSizeFuncAddr =                          \
      GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                    \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);       \
    if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr) {\
      VLOG(3) <<"aclop exexuted";                                         \
      return originCallExpression;                                        \
    }                                                                     \
  } while (0)                                                             \
// clang-format on

class NpuOpRunner {
 public:
  NpuOpRunner();
  explicit NpuOpRunner(const std::string &op_type);
  NpuOpRunner(const std::string &op_type,
              const std::vector<phi::DenseTensor> &inputs = {},
              const std::vector<phi::DenseTensor> &outputs = {},
              const NPUAttributeMap &attrs = {});

  NpuOpRunner(const NpuOpRunner &runner) = delete;
  NpuOpRunner &operator=(const NpuOpRunner &runner) = delete;

  ~NpuOpRunner();

  const std::string &Type();

  NpuOpRunner &SetType(const std::string &name);

  NpuOpRunner &AddAttr(const std::string &name, const NPUAttribute &attr);

  NpuOpRunner &AddAttrDataType(const std::string &name,
                               const NPUAttribute &attr);

  NpuOpRunner &AddAttrs(const NPUAttributeMap &attrs);

  NpuOpRunner &AddInput(const phi::DenseTensor &tensor);

  NpuOpRunner &AddInput(const phi::DenseTensor &tensor, aclMemType mem_type);

  template <typename T>
  NpuOpRunner &AddInput(const phi::CustomContext &dev_ctx,
                        const std::vector<T> &&values,
                        const bool is_const = true);

  NpuOpRunner &AddOutput(const phi::DenseTensor &tensor);

  NpuOpRunner &AddInputs(const std::vector<phi::DenseTensor> &tensors);

  NpuOpRunner &AddInputNames(const std::vector<std::string> &names);

  NpuOpRunner &AddOutputs(const std::vector<phi::DenseTensor> &tensors);

  aclTensorDesc *GetInputDesc(size_t index);

  aclTensorDesc *GetOutputDesc(size_t index);

  std::vector<aclTensorDesc *> &GetInputDescs();

  std::vector<aclTensorDesc *> &GetOutputDescs();

  std::vector<aclDataBuffer *> &GetInputBuffers();

  std::vector<aclDataBuffer *> &GetOutputBuffers();

  void Run(aclrtStream stream = nullptr, bool sync = false) const;

  static void TypeAdapter(
      const std::vector<phi::DenseTensor> &inputs,
      const std::vector<phi::DenseTensor> &outputs,
      const NPUAttributeMap &attrs,
      const phi::CustomContext &dev_ctx,
      std::function<void(const std::vector<phi::DenseTensor> &,
                         const std::vector<phi::DenseTensor> &,
                         const NPUAttributeMap &,
                         const phi::CustomContext &)> op_runner,
      const std::vector<phi::DataType> &input_type,
      const std::vector<phi::DataType> &output_type) {
    std::function<void(const std::vector<phi::DenseTensor> &,
                       const std::vector<phi::DenseTensor> &,
                       const NPUAttributeMap &,
                       const phi::CustomContext &,
                       const std::vector<std::vector<int>> &)>
        new_op_runner =
            [&](const std::vector<phi::DenseTensor> &inputs,
                const std::vector<phi::DenseTensor> &outputs,
                const NPUAttributeMap &attrs,
                const phi::CustomContext &dev_ctx,
                const std::vector<std::vector<int>> &host_vecs = {}) {
              op_runner(inputs, outputs, attrs, dev_ctx);
            };
    TypeAdapter<int>(inputs,
                     outputs,
                     attrs,
                     dev_ctx,
                     new_op_runner,
                     input_type,
                     output_type);
  }

  template <typename T>
  static void TypeAdapter(
      const std::vector<phi::DenseTensor> &inputs,
      const std::vector<phi::DenseTensor> &outputs,
      const NPUAttributeMap &attrs,
      const phi::CustomContext &dev_ctx,
      std::function<void(const std::vector<phi::DenseTensor> &,
                         const std::vector<phi::DenseTensor> &,
                         const NPUAttributeMap &,
                         const phi::CustomContext &,
                         const std::vector<std::vector<T>> &)> op_runner,
      const std::vector<phi::DataType> &input_type,
      const std::vector<phi::DataType> &output_type,
      const std::vector<std::vector<T>> &&host_vecs = {}) {
    std::vector<phi::DenseTensor> tmp_inputs(inputs.size());
    std::vector<phi::DenseTensor> tmp_outputs(outputs.size());

    for (size_t i = 0; i < input_type.size(); ++i) {
      bool cast_input = (input_type[i] == phi::DataType::UNDEFINED ||
                         input_type[i] != inputs[i].dtype());
      if (!cast_input) {
        tmp_inputs[i] = inputs[i];
      } else {
        tmp_inputs[i].Resize(inputs[i].dims());
        dev_ctx.Alloc(&(tmp_inputs[i]), input_type[i]);

        CastCall(dev_ctx, inputs[i], input_type[i], &(tmp_inputs[i]));
      }
    }
    for (size_t i = 0; i < output_type.size(); ++i) {
      bool cast_output = (output_type[i] == phi::DataType::UNDEFINED ||
                          output_type[i] != outputs[i].dtype());
      if (!cast_output) {
        tmp_outputs[i] = outputs[i];
      } else {
        tmp_outputs[i].Resize(outputs[i].dims());
        dev_ctx.Alloc(&(tmp_outputs[i]), output_type[i]);
      }
    }

    op_runner(tmp_inputs, tmp_outputs, attrs, dev_ctx, host_vecs);

    for (size_t i = 0; i < output_type.size(); ++i) {
      bool cast_output = (output_type[i] == phi::DataType::UNDEFINED ||
                          output_type[i] != outputs[i].dtype());
      if (cast_output) {
        CastCall(dev_ctx,
                 tmp_outputs[i],
                 outputs[i].dtype(),
                 const_cast<phi::DenseTensor*>(&(outputs[i])));
      }
    }
  }

  static bool GetFloatStatus(aclrtStream stream);
  static void ClearFloatStatus(aclrtStream stream);

 private:
  aclTensorDesc *CreateTensorDesc(phi::DenseTensor tensor,
                                  aclMemType mem_type = ACL_MEMTYPE_DEVICE);
  aclDataBuffer *CreateDataBuffer(phi::DenseTensor tensor);
  void InitFloatStatus(aclrtStream stream) const;
  void AllocFloatStatus(aclrtStream stream) const;
  void PrintOpInfo() const;

  static void AclopCastCall(
      const phi::CustomContext& dev_ctx,
      const phi::DenseTensor& x,
      phi::DataType dtype,
      phi::DenseTensor* out) {
    const auto &cast_runner = NpuOpRunner(
        "Cast",
        {x},
        {*out},
        {{"dst_type", static_cast<int>(ConvertToNpuDtype(dtype))}});
    cast_runner.Run(dev_ctx.stream());
  }

  static void CastCall(
      const phi::CustomContext& dev_ctx,
      const phi::DenseTensor& x,
      phi::DataType dtype,
      phi::DenseTensor* out) {
    DO_COMPATIBILITY(
        aclnnCast,
        (AclopCastCall(dev_ctx, x, dtype, out)));

    int aclDtype = ConvertToNpuDtype(dtype);
    EXEC_NPU_CMD(aclnnCast, dev_ctx, x, aclDtype, *out);
  }

 private:
  std::string op_type_;
  std::vector<aclDataBuffer *> input_buffers_;
  std::vector<aclDataBuffer *> output_buffers_;
  std::vector<aclTensorDesc *> input_descs_;
  std::vector<aclTensorDesc *> output_descs_;
  std::vector<phi::DenseTensor> host_tensors_;
  aclopAttr *attr_{nullptr};
};

template <typename T>
struct cpp_type_to_acl_dtype;

template <>
struct cpp_type_to_acl_dtype<float> {
  static const aclDataType value() { return ACL_FLOAT; }
};

template <>
struct cpp_type_to_acl_dtype<double> {
  static const aclDataType value() { return ACL_DOUBLE; }
};
