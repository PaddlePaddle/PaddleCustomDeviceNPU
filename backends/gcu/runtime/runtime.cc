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

#include "runtime/runtime.h"

#include <eccl.h>

#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend/executor/cast_runner.h"
#include "glog/logging.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "runtime/flags.h"

namespace {
const char *const kDeviceType = "gcu";
const char *const kSubDeviceType = "none";

inline size_t get_devices_count() {
  int count;
  RT_CHECK(topsGetDeviceCount(&count));
  return static_cast<size_t>(count);
}

inline int get_current_device_id() {
  int device = 0;
  RT_CHECK(topsGetDevice(&device));
  return device;
}

class GcuDeviceGuard {
 public:
  explicit GcuDeviceGuard(int device) {
    RT_CHECK(topsGetDevice(&device_));
    if (device_ != device) {
      RT_CHECK(topsSetDevice(device));
      reset_device_ = true;
    }
  }

  ~GcuDeviceGuard() {
    if (reset_device_) {
      RT_CHECK(topsSetDevice(device_));
    }
  }

  GcuDeviceGuard() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(GcuDeviceGuard);

 private:
  int device_;
  bool reset_device_ = false;
};

}  // namespace

static int g_current_device_id(-1);
static size_t total_alloc = 0;
static size_t total_using = 0;
static size_t total_free = 0;
static size_t total_pinned_alloc = 0;
static size_t total_pinned_using = 0;
static size_t total_pinned_free = 0;

// Device
C_Status Init() {
  size_t dev_cnt = get_devices_count();
  if (std::getenv("FLAGS_selected_gcus") != nullptr) {
    g_current_device_id = std::atoi(std::getenv("FLAGS_selected_gcus"));
  } else {
    g_current_device_id = 0;
  }
  VLOG(0) << "Backend GCU Init, get GCU count:" << dev_cnt
          << ", current device id:" << g_current_device_id;
  return C_SUCCESS;
}

C_Status Finalize() {
  VLOG(0) << "Backend GCU Finalize";
  return C_SUCCESS;
}

void OpsInitialize() { TOPSATEN_CHECK(topsatenInit()); }
void OpsFinalize() { TOPSATEN_CHECK(topsatenFinalize()); }

C_Status InitDevice(const C_Device device) {
  RT_CHECK(topsSetDevice(device->id));
  OpsInitialize();
  VLOG(0) << "Backend GCU init device:" << device->id;
  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  RT_CHECK(topsSetDevice(device->id));
  return C_SUCCESS;
}

C_Status GetDevice(const C_Device device) {
  RT_CHECK(topsGetDevice(&(device->id)));
  return C_SUCCESS;
}

C_Status DeInitDevice(const C_Device device) {
  OpsFinalize();
  VLOG(0) << "Backend GCU finalize device:" << device->id;
  return C_SUCCESS;
}

// Stream
C_Status CreateStream(const C_Device device, C_Stream *stream) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamCreate(reinterpret_cast<topsStream_t *>(stream)));
  VLOG(3) << "[GcuRuntime], CreateStream:" << stream;
  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamDestroy(reinterpret_cast<topsStream_t>(stream)));
  VLOG(3) << "[GcuRuntime], DestroyStream:" << stream;
  return C_SUCCESS;
}

static void StreamCallbackFunc(topsStream_t stream,
                               topsError_t status,
                               void *user_data) {
  std::unique_ptr<std::function<void()>> func(
      reinterpret_cast<std::function<void()> *>(user_data));
  (*func)();
}

// user_data is phi::stream::Stream::Callback, which is std::function<void()>
// defined at paddle/phi/backends/stream.h
// For more information about this interface, please refer to
// CustomDevice::AddCallback at paddle/phi/backends/custom/custom_device.cc
C_Status AddCallback(const C_Device device,
                     C_Stream stream,
                     C_Callback callback,
                     void *user_data) {
  RT_CHECK(topsStreamAddCallback(reinterpret_cast<topsStream_t>(stream),
                                 StreamCallbackFunc,
                                 user_data,
                                 topsStreamDefault));
  return C_SUCCESS;
}

// Event
C_Status CreateEvent(const C_Device device, C_Event *event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventCreate(reinterpret_cast<topsEvent_t *>(event)));
  VLOG(3) << "[GcuRuntime], CreateEvent:" << event;
  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventRecord(reinterpret_cast<topsEvent_t>(event),
                           reinterpret_cast<topsStream_t>(stream)));
  VLOG(3) << "[GcuRuntime], RecordEvent:" << event << ", stream:" << stream;
  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventDestroy(reinterpret_cast<topsEvent_t>(event)));
  VLOG(3) << "[GcuRuntime], DestroyEvent:" << event;
  return C_SUCCESS;
}

// Synchronize
C_Status SyncDevice(const C_Device device) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsDeviceSynchronize());
  VLOG(3) << "[GcuRuntime], SyncDevice:" << device->id;
  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamSynchronize(reinterpret_cast<topsStream_t>(stream)));
  VLOG(3) << "[GcuRuntime], SyncStream:" << stream;
  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsEventSynchronize(reinterpret_cast<topsEvent_t>(event)));
  VLOG(3) << "[GcuRuntime], SyncEvent:" << event;
  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsStreamWaitEvent(reinterpret_cast<topsStream_t>(stream),
                               reinterpret_cast<topsEvent_t>(event),
                               0));
  VLOG(3) << "[GcuRuntime], StreamWaitEvent, event:" << event
          << ", stream:" << stream;
  return C_SUCCESS;
}

// Memory
C_Status Allocate(const C_Device device, void **ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  void *tmp_ptr = nullptr;
  topsError_t ret = topsMalloc(&tmp_ptr, size);
  if (ret != topsSuccess) {
    *ptr = nullptr;
    VLOG(1) << "[AllocFromGCU] Failed to alloc hbm, size: " << size;
    return C_FAILED;
  }
  *ptr = tmp_ptr;
  VLOG(6) << "[AllocFromGCU] Alloc gcu hbm size:" << size << ", ptr:" << *ptr;
  total_alloc += size;
  total_using = total_alloc - total_free;
  VLOG(2) << "[AllocFromGCU] Alloc gcu hbm size: " << size << ", total_alloc: "
          << (total_alloc / static_cast<double>(1024 * 1024))
          << " MB, total_using: "
          << (total_using / static_cast<double>(1024 * 1024)) << " MB.";
  return C_SUCCESS;
}

C_Status Deallocate(const C_Device device, void *ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsFree(ptr));
  total_free += size;
  total_using = total_alloc - total_free;

  VLOG(2) << "[FreeToGCU] Free gcu hbm size:" << size << ", total_alloc: "
          << (total_alloc / static_cast<double>(1024 * 1024))
          << " MB, total_using: "
          << (total_using / static_cast<double>(1024 * 1024)) << " MB.";

  return C_SUCCESS;
}

C_Status HostAllocate(const C_Device device, void **ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  void *gcu_host_mem;
  // NOTE: If size is 0, no memory is allocated, *ptr returns nullptr, and
  // topsSuccess is returned.
  auto ret = topsHostMalloc(&gcu_host_mem, size, topsHostMallocDefault);
  if (ret == topsSuccess) {
    *ptr = gcu_host_mem;
    total_pinned_alloc += size;
    total_pinned_using = total_pinned_alloc - total_pinned_free;
    VLOG(3) << "[AllocFromGCU] Alloc gcu host memory size:" << size
            << ", total_pinned_alloc: "
            << (total_pinned_alloc / static_cast<double>(1024 * 1024))
            << " MB, total_pinned_using: "
            << (total_pinned_using / static_cast<double>(1024 * 1024))
            << " MB.";
    return C_SUCCESS;
  }
  *ptr = nullptr;
  return C_FAILED;
}

C_Status PinnedAllocate(void **ptr, size_t size) {
  C_Device_st device;
  device.id = g_current_device_id;
  return HostAllocate(&device, ptr, size);
}

C_Status HostDeallocate(const C_Device device, void *ptr, size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsHostFree(ptr));
  total_pinned_free += size;
  total_pinned_using = total_pinned_alloc - total_pinned_free;
  VLOG(3) << "[FreeToGCU] Free gcu host memory size:" << size
          << ", total_pinned_alloc: "
          << (total_pinned_alloc / static_cast<double>(1024 * 1024))
          << " MB, total_pinned_using: "
          << (total_pinned_using / static_cast<double>(1024 * 1024)) << " MB.";
  return C_SUCCESS;
}

C_Status PinnedDeallocate(void *ptr, size_t size) {
  C_Device_st device;
  device.id = g_current_device_id;
  return HostDeallocate(&device, ptr, size);
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsMemcpy(dst, src, size, topsMemcpyHostToDevice));
  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsMemcpy(dst, src, size, topsMemcpyDeviceToHost));
  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsMemcpy(dst, src, size, topsMemcpyDeviceToDevice));
  return C_SUCCESS;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsMemcpyAsync(dst,
                           src,
                           size,
                           topsMemcpyHostToDevice,
                           reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsMemcpyAsync(dst,
                           src,
                           size,
                           topsMemcpyDeviceToHost,
                           reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  GcuDeviceGuard guard(device->id);
  RT_CHECK(topsMemcpyAsync(dst,
                           src,
                           size,
                           topsMemcpyDeviceToDevice,
                           reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

// Get visible device info
C_Status GetDevicesCount(size_t *count) {
  *count = get_devices_count();
  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *device) {
  size_t count = get_devices_count();
  for (size_t dev_id = 0; dev_id < count; ++dev_id) {
    device[dev_id] = dev_id;
  }
  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 256;
  return C_SUCCESS;
}

C_Status DeviceMaxChunkSize(const C_Device device, size_t *size) {
  *size = 0;
  return C_SUCCESS;
}

C_Status ExtraPaddingSize(const C_Device device, size_t *size) {
  *size = 32;
  return C_SUCCESS;
}

// CCL
namespace {
const std::unordered_map<C_CCLReduceOp, ecclRedOp_t> kEcclOpMap = {
    {C_CCLReduceOp::SUM, ecclSum},
    {C_CCLReduceOp::MAX, ecclMax},
    {C_CCLReduceOp::MIN, ecclMin},
    {C_CCLReduceOp::PRODUCT, ecclProd},
};

ecclRedOp_t ConvertEcclReduceOp(const C_CCLReduceOp &reduce_op) {
  try {
    return kEcclOpMap.at(reduce_op);
  } catch (const std::out_of_range &e) {
    throw std::runtime_error(
        "Failed to ConvertEcclReduceOp, unsupport reduce_op");
  }
}

const std::unordered_map<C_DataType, ecclDataType_t> kEcclDtypeMap = {
    {C_DataType::UINT8, ecclUint8},
    {C_DataType::UINT32, ecclUint32},
    {C_DataType::UINT64, ecclUint64},
    {C_DataType::INT8, ecclInt8},
    {C_DataType::INT32, ecclInt32},
    {C_DataType::INT64, ecclInt64},
    {C_DataType::FLOAT16, ecclFloat16},
    {C_DataType::FLOAT32, ecclFloat32},
    {C_DataType::FLOAT64, ecclFloat64},
    {C_DataType::BFLOAT16, ecclBfloat16},
};

ecclDataType_t ConvertEcclDataType(const C_DataType &dtype) {
  try {
    return kEcclDtypeMap.at(dtype);
  } catch (const std::out_of_range &e) {
    throw std::runtime_error("Failed to ConvertEcclDataType, unsupport dtype");
  }
}

const std::unordered_map<C_DataType, std::string> kDtypeToStr = {
    {C_DataType::UNDEFINED, "UNDEFINED"},
    {C_DataType::BOOL, "BOOL"},
    {C_DataType::UINT8, "UINT8"},
    {C_DataType::UINT16, "UINT16"},
    {C_DataType::UINT32, "UINT32"},
    {C_DataType::UINT64, "UINT64"},
    {C_DataType::INT8, "INT8"},
    {C_DataType::INT16, "INT16"},
    {C_DataType::INT32, "INT32"},
    {C_DataType::INT64, "INT64"},
    {C_DataType::FLOAT16, "FLOAT16"},
    {C_DataType::FLOAT32, "FLOAT32"},
    {C_DataType::FLOAT64, "FLOAT64"},
    {C_DataType::BFLOAT16, "BFLOAT16"},
    {C_DataType::COMPLEX64, "COMPLEX64"},
    {C_DataType::COMPLEX128, "COMPLEX128"},
};

std::string DataTypeToStr(const C_DataType &dtype) {
  try {
    return kDtypeToStr.at(dtype);
  } catch (const std::out_of_range &e) {
    throw std::runtime_error("DataTypeToStr failed, unsupport dtype");
  }
}

std::string SerializeUniqueId(C_CCLRootId *unique_id) {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(unique_id->data);
  std::ostringstream oss;
  for (size_t i = 0; i < unique_id->sz; ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}
}  // namespace

C_Status XcclGetUniqueIdSize(size_t *size) {
  *size = sizeof(ecclUniqueId);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  if (unique_id->sz != sizeof(ecclUniqueId)) {
    VLOG(0) << "unique_id->sz must be equal sizeof(ecclUniqueId)";
    return C_FAILED;
  }
  ECCL_CHECK(
      ecclGetUniqueId(reinterpret_cast<ecclUniqueId *>(unique_id->data)));
  VLOG(0) << "Backend GCU GetUniqueId, UniqueId size:" << unique_id->sz
          << ", UniqueId:" << SerializeUniqueId(unique_id);
  return C_SUCCESS;
}

C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  ECCL_CHECK(
      ecclCommInitRank(reinterpret_cast<ecclComm_t *>(comm),
                       nranks,
                       *(reinterpret_cast<ecclUniqueId *>(unique_id->data)),
                       rank));
  VLOG(0) << "Backend GCU CommInitRank successfully, world size:" << nranks
          << ", rank:" << rank << ", current device id:" << g_current_device_id;
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  ECCL_CHECK(ecclCommDestroy(reinterpret_cast<ecclComm_t>(comm)));
  VLOG(0) << "Backend GCU CommDestroy successfully, current device id:"
          << g_current_device_id;
  return C_SUCCESS;
}

C_DataType CastRealDataType(C_DataType data_type) {
  switch (data_type) {
    case C_DataType::INT64:
      return C_DataType::INT32;
    case C_DataType::UINT64:
      return C_DataType::UINT32;
    case C_DataType::FLOAT64:
      return C_DataType::FLOAT32;

    default:
      return data_type;
  }
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  C_DataType real_data_type = CastRealDataType(data_type);

  if (data_type != real_data_type) {
    phi::DataType dtype = phi::capi::ToPhiDataType(data_type);
    phi::DataType real_dtype = phi::capi::ToPhiDataType(real_data_type);

    VLOG(6) << "XcclAllReduce data_type: " << phi::DataTypeToString(dtype)
            << ", real_data_type: " << phi::DataTypeToString(real_dtype);
    static std::unordered_map<std::string, void *> device_memorys;
    std::string key = phi::DataTypeToString(real_dtype) + " " +
                      std::to_string(count * phi::SizeOf(real_dtype));
    void *tmp_buf = nullptr;
    if (device_memorys.count(key) <= 0) {
      C_Device_st device;
      device.id = get_current_device_id();
      Allocate(&device, &tmp_buf, count * phi::SizeOf(real_dtype));
      device_memorys[key] = tmp_buf;
    } else {
      tmp_buf = device_memorys[key];
    }

    backend::CastRunner(reinterpret_cast<topsStream_t>(stream),
                        {static_cast<int64_t>(count)},
                        dtype,
                        real_dtype,
                        send_buf,
                        tmp_buf);

    ECCL_CHECK(ecclAllReduce(tmp_buf,
                             tmp_buf,
                             count,
                             ConvertEcclDataType(real_data_type),
                             ConvertEcclReduceOp(op),
                             reinterpret_cast<ecclComm_t>(comm),
                             reinterpret_cast<topsStream_t>(stream)));

    backend::CastRunner(reinterpret_cast<topsStream_t>(stream),
                        {static_cast<int64_t>(count)},
                        real_dtype,
                        dtype,
                        tmp_buf,
                        recv_buf);
  } else {
    ECCL_CHECK(ecclAllReduce(send_buf,
                             recv_buf,
                             count,
                             ConvertEcclDataType(data_type),
                             ConvertEcclReduceOp(op),
                             reinterpret_cast<ecclComm_t>(comm),
                             reinterpret_cast<topsStream_t>(stream)));
  }

  VLOG(6) << "Backend GCU XcclAllReduce successfully, data_type:"
          << DataTypeToStr(data_type) << ", count:" << count;
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  ECCL_CHECK(ecclBroadcast(buf,
                           buf,
                           count,
                           ConvertEcclDataType(data_type),
                           static_cast<int>(root),
                           reinterpret_cast<ecclComm_t>(comm),
                           reinterpret_cast<topsStream_t>(stream)));
  VLOG(6) << "Backend GCU XcclBroadcast successfully, data_type:"
          << DataTypeToStr(data_type) << ", count:" << count;
  return C_SUCCESS;
}

C_Status XcclReduce(void *send_buf,
                    void *recv_buf,
                    size_t count,
                    C_DataType data_type,
                    C_CCLReduceOp op,
                    size_t root,
                    C_CCLComm comm,
                    C_Stream stream) {
  ECCL_CHECK(ecclReduce(send_buf,
                        recv_buf,
                        count,
                        ConvertEcclDataType(data_type),
                        ConvertEcclReduceOp(op),
                        static_cast<int>(root),
                        reinterpret_cast<ecclComm_t>(comm),
                        reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  ECCL_CHECK(ecclAllGather(send_buf,
                           recv_buf,
                           count,
                           ConvertEcclDataType(data_type),
                           reinterpret_cast<ecclComm_t>(comm),
                           reinterpret_cast<topsStream_t>(stream)));
  VLOG(6) << "Backend GCU XcclAllGather successfully, data_type:"
          << DataTypeToStr(data_type) << ", count:" << count;
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  ECCL_CHECK(ecclReduceScatter(send_buf,
                               recv_buf,
                               count,
                               ConvertEcclDataType(data_type),
                               ConvertEcclReduceOp(op),
                               reinterpret_cast<ecclComm_t>(comm),
                               reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  ECCL_CHECK(ecclGroupStart());
  return C_SUCCESS;
}

C_Status XcclGroupEnd() {
  ECCL_CHECK(ecclGroupEnd());
  return C_SUCCESS;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  ECCL_CHECK(ecclSend(send_buf,
                      count,
                      ConvertEcclDataType(data_type),
                      static_cast<int>(dest_rank),
                      reinterpret_cast<ecclComm_t>(comm),
                      reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  ECCL_CHECK(ecclRecv(recv_buf,
                      count,
                      ConvertEcclDataType(data_type),
                      static_cast<int>(src_rank),
                      reinterpret_cast<ecclComm_t>(comm),
                      reinterpret_cast<topsStream_t>(stream)));
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->device_type = const_cast<char *>(kDeviceType);
  params->sub_device_type = const_cast<char *>(kSubDeviceType);

  // Device
  params->interface->initialize = Init;
  params->interface->finalize = Finalize;
  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDevice;
  params->interface->deinit_device = DeInitDevice;

  // Stream
  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;
  params->interface->stream_add_callback = AddCallback;

  // Event
  params->interface->create_event = CreateEvent;
  params->interface->record_event = RecordEvent;
  params->interface->destroy_event = DestroyEvent;

  // Synchronize
  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  // Memory
  params->interface->device_memory_allocate = Allocate;
  params->interface->device_memory_deallocate = Deallocate;
  params->interface->host_memory_allocate = HostAllocate;
  params->interface->host_memory_deallocate = HostDeallocate;
  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_p2p = nullptr;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_p2p = nullptr;

  // Get visible device info
  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;
  params->interface->device_max_chunk_size = DeviceMaxChunkSize;
  params->interface->device_extra_padding_size = ExtraPaddingSize;

  // Xccl
  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;
  params->interface->xccl_reduce = XcclReduce;
  params->interface->xccl_all_gather = XcclAllGather;
  params->interface->xccl_reduce_scatter = XcclReduceScatter;
  params->interface->xccl_group_start = XcclGroupStart;
  params->interface->xccl_group_end = XcclGroupEnd;
  params->interface->xccl_send = XcclSend;
  params->interface->xccl_recv = XcclRecv;
  VLOG(0) << "InitPlugin for backend GCU successfully.";
}
