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

#include "runtime/runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "habanalabs/hccl.h"
#include "habanalabs/hccl_types.h"
#include "paddle/phi/common/type_traits.h"

FLAGS_DEFINE_bool(intel_hpu_runtime_debug, false, "runtime debug log");
FLAGS_DEFINE_uint32(
    intel_hpu_profiling_type,
    1,
    "set runtime profiling type, 1=all, 2 = host only, 3 = device only");

inline hcclDataType_t PDDataTypeToHcclDataType(C_DataType type) {
  if (type == C_DataType::FLOAT32) {
    return hcclFloat32;
  } else if (type == C_DataType::FLOAT16) {
    return hcclBfloat16;
  } else if (type == C_DataType::INT32) {
    return hcclInt32;
  } else if (type == C_DataType::INT8) {
    return hcclInt8;
  } else if (type == C_DataType::UINT8) {
    return hcclUint8;
  } else if (type == C_DataType::INT64) {
    return hcclInt64;
  } else {
    PD_CHECK(false, "[RUNTIME] Datatype %d in hccl is not supported.", type);
  }
}

hcclRedOp_t PDReduceOpToHcclReduceOp(C_CCLReduceOp op) {
  if (op == C_CCLReduceOp::MIN) {
    return hcclMin;
  } else if (op == C_CCLReduceOp::MAX) {
    return hcclMax;
  } else if (op == C_CCLReduceOp::SUM) {
    return hcclSum;
  } else if (op == C_CCLReduceOp::PRODUCT) {
    return hcclProd;
  } else {
    PD_CHECK(false, "[RUNTIME] Reduceop %d n hccl is not supported.", op);
  }
}

class RuntimeManager {
 public:
  RuntimeManager() {}
  ~RuntimeManager() {}

  void SetDevice(const C_Device device) {
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "set device id to " << device->id << ", current = " << moduleID;
    synStatus status = synFail;
    auto require_id = static_cast<synModuleId>(device->id);
    if (require_id == moduleID) {
      if (Status == 0) {
        LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
            << "1st ================================= " << require_id;
        status = synDeviceAcquireByModuleId(&deviceId, require_id);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synDeviceAcquireByModuleId() failed = %d",
                 status);
      }
    } else {
      // release the old one and acquire a new one
      if (Status == 1) {
        status = synDeviceRelease(deviceId);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synDeviceRelease() failed = %d",
                 status);
      }
      status = synDeviceAcquireByModuleId(&deviceId, require_id);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synDeviceAcquireByModuleId() failed = %d",
               status);
    }
    Status = 1;
    moduleID = device->id;
  }

  void Release(const C_Device device) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    if (Status == 1) {
      synStatus status = synDeviceRelease(deviceId);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synDeviceRelease() failed = %d",
               status);
    }

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "moduleID =  " << moduleID << " released";
  }

  void GetMemoryStats(const C_Device device,
                      size_t *total_memory,
                      size_t *free_memory) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    uint64_t free = 0;
    uint64_t total = 0;

    synStatus status = synDeviceGetMemoryInfo(deviceId, &free, &total);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synDeviceGetMemoryInfo() failed = %d",
             status);

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "free = " << free;
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "total = " << total;

    *total_memory = static_cast<size_t>(total);
    *free_memory = static_cast<size_t>(free);
  }

  void GetNumDevices(size_t *count) {
    uint32_t num_devices = 0;
    synStatus status = synDeviceGetCount(&num_devices);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synDeviceGetCount() failed = %d",
             status);
    // if (num_devices >= 1) {
    //   LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
    //       << "total device num =" << num_devices
    //       << " actual return 1, only support 1 thread acquire 1 device";
    //   num_devices = 1;
    // }

    *count = static_cast<size_t>(num_devices);
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << " number device = " << *count;
  }

  C_Status CreateStream(const C_Device device, C_Stream *stream) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    auto it = streams.find(reinterpret_cast<synStreamHandle>(*stream));
    if (it == streams.end()) {
      synStreamHandle h = nullptr;
      synStatus status = synStreamCreateGeneric(&h, device->id, 0);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synStreamCreateGeneric() failed = %d",
               status);

      streams.insert(h);
      *stream = reinterpret_cast<C_Stream>(h);
      LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
          << "create stream stream = " << h;
    }

    return C_SUCCESS;
  }

  C_Status DestroyStream(const C_Device device, C_Stream stream) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;
    auto it = streams.find(reinterpret_cast<synStreamHandle>(stream));
    if (it != streams.end()) {
      synStatus status = synStreamDestroy(*it);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synStreamDestroy() failed = %d",
               status);

      streams.erase(*it);
    }
    return C_SUCCESS;
  }

  C_Status Copy(const C_Device device,
                void *dst,
                const void *src,
                size_t size,
                size_t flag = 0 /*0 = h2d, 1 = d2h, 2=d2d*/) {
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "copy: flag = " << flag << ", size = " << size << ", src = " << src
        << ", dst = " << dst;
    synStatus status = synFail;
    if (flag == 0) {
      if (stream_h2d == nullptr) {
        status = synStreamCreateGeneric(
            reinterpret_cast<synStreamHandle *>(&stream_h2d), device->id, 0);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synStreamCreateGeneric() failed = %d",
                 status);

        LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
            << "create builtin stream h2d" << stream_h2d;
      }

      addCache(device, src, size);
      status = synMemCopyAsync(stream_h2d,
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               HOST_TO_DRAM);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synMemCopyAsync(HOST_TO_DRAM) failed = %d",
               status);
      status = synStreamSynchronize(stream_h2d);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synStreamSynchronize(stream_h2d) failed = %d",
               status);

    } else if (flag == 1) {
      if (stream_d2h == nullptr) {
        status = synStreamCreateGeneric(
            reinterpret_cast<synStreamHandle *>(&stream_d2h), device->id, 0);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synStreamCreateGeneric() failed = %d",
                 status);
      }

      addCache(device, dst, size);

      status = synMemCopyAsync(stream_d2h,
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_HOST);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synMemCopyAsync() failed = %d",
               status);
      status = synStreamSynchronize(stream_d2h);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synStreamSynchronize() failed = %d",
               status);

    } else if (flag == 2) {
      if (stream_d2d == nullptr) {
        status = synStreamCreateGeneric(
            reinterpret_cast<synStreamHandle *>(&stream_d2d), device->id, 0);
        PD_CHECK(status == synSuccess,
                 "[RUNTIME] synStreamCreateGeneric() failed = %d",
                 status);
      }
      status = synMemCopyAsync(stream_d2d,
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_DRAM);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synMemCopyAsync() failed = %d",
               status);

      status = synStreamSynchronize(stream_d2d);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synStreamSynchronize() failed = %d",
               status);
    }
    return C_SUCCESS;
  }

  C_Status AsyncCopy(const C_Device device,
                     C_Stream stream,
                     void *dst,
                     const void *src,
                     size_t size,
                     size_t flag = 0 /*0 = h2d, 1 = d2h, 2=d2d*/) {
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "AsyncCopy: flag = " << flag << ", size = " << size
        << ", stream = " << stream << ", src = " << src << ", dst = " << dst;
    synStatus status = synFail;
    if (flag == 0) {
      addCache(device, src, size);
      status = synMemCopyAsync(reinterpret_cast<synStreamHandle>(stream),
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               HOST_TO_DRAM);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synMemCopyAsync() failed = %d",
               status);

    } else if (flag == 1) {
      addCache(device, dst, size);
      status = synMemCopyAsync(reinterpret_cast<synStreamHandle>(stream),
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_HOST);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synMemCopyAsync() failed = %d",
               status);

    } else if (flag == 2) {
      status = synMemCopyAsync(reinterpret_cast<synStreamHandle>(stream),
                               reinterpret_cast<uint64_t>(src),
                               size,
                               reinterpret_cast<uint64_t>(dst),
                               DRAM_TO_DRAM);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synMemCopyAsync() failed = %d",
               status);
    }
    return C_SUCCESS;
  }

  C_Status CreateEvent(const C_Device device, C_Event *event) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;

    auto it = events.find(reinterpret_cast<synEventHandle>(*event));
    if (it == events.end()) {
      synEventHandle e = nullptr;
      synStatus status = synEventCreate(&e, deviceId, 0);
      PD_CHECK(status == synSuccess,
               "[RUNTIME] synEventCreate() failed = %d",
               status);
      events.insert(e);
      *event = reinterpret_cast<C_Event>(e);
    }

    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "device id=" << device->id << " create event = " << *event;

    return C_SUCCESS;
  }

  C_Status DestroyEvent(const C_Device device, C_Event event) {
    LOG_IF(ERROR, static_cast<synModuleId>(device->id) != moduleID)
        << "[RUNTIME] moduleID mismatch : moduleID = " << moduleID
        << ", current = " << moduleID;
    auto it = events.find(reinterpret_cast<synEventHandle>(event));
    if (it != events.end()) {
      synStatus status = synEventDestroy(*it);

      PD_CHECK(status == synSuccess,
               "[RUNTIME] synEventDestroy() failed = %d",
               status);

      events.erase(*it);
    }
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "device id=" << device->id << " remove event = " << event;

    return C_SUCCESS;
  }

  inline synModuleId GetModuleID() { return moduleID; }
  inline synDeviceId GetDeviceID() { return deviceId; }

  inline void addCache(const C_Device device, const void *ptr, size_t size) {
    auto it = hostMappedAddress.find(ptr);
    synStatus status = synFail;
    if (it == hostMappedAddress.end()) {
      // not found, map and cache
      status = synHostMap(device->id, size, ptr);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synHostMap() failed = " << status;
      hostMappedAddress[ptr] = size;
    } else {
      if (it->second != size) {
        // found but size not equal
        // unmap the old one and map a new one
        status = synHostUnmap(device->id, ptr);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synHostUnmap() failed = " << status;

        status = synHostMap(device->id, size, ptr);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synHostMap() failed = " << status;
        hostMappedAddress[ptr] = size;
      }
    }
  }

  void GetUniqueIdSize(size_t *sz) {
    if (uid.length == 0) {
      CHECK_HCCL_STATUS(hcclGetUniqueId(&uid));
    }
    *sz = uid.length;
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "uid size = " << *sz;
  }

  C_Status GetUniqueId(C_CCLRootId *unique_id) {
    if (uid.length == 0) {
      CHECK_HCCL_STATUS(hcclGetUniqueId(&uid));
    }
    unique_id->sz = uid.length;
    memcpy(reinterpret_cast<void *>(unique_id->data),
           reinterpret_cast<void *>(uid.internal),
           uid.length);
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "uid size = " << unique_id->sz;
    return C_SUCCESS;
  }

  C_Status CommInitRank(size_t ranks,
                        C_CCLRootId *unique_id,
                        size_t rank,
                        C_CCLComm *comm) {
    LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
        << "uid size = " << unique_id->sz << ", rank = " << rank;
    uid.length = unique_id->sz;
    memcpy(reinterpret_cast<void *>(uid.internal),
           reinterpret_cast<void *>(unique_id->data),
           unique_id->sz);
    CHECK_HCCL_STATUS(hcclCommInitRank(
        reinterpret_cast<hcclComm_t *>(comm), ranks, uid, rank));
    return C_SUCCESS;
  }

 private:
  synModuleId moduleID = 0;
  std::string busID = "";
  synDeviceId deviceId = 0;
  uint32_t Status = 0;  // 1 acquire, 0 not acquire
  uint32_t count = 0;

  // hccl
  hcclUniqueId uid = hcclUniqueId{.length = 0};

  // user streams
  std::set<synStreamHandle> streams;

  // builtin stream
  synStreamHandle stream_h2d = nullptr;
  synStreamHandle stream_d2h = nullptr;
  synStreamHandle stream_d2d = nullptr;

  // user events
  std::set<synEventHandle> events;

  // cache
  std::unordered_map<const void *, size_t> hostMappedAddress;
};

static RuntimeManager runtimeManager;

C_Status Init() {
  synStatus status = synInitialize();
  PD_CHECK(
      status == synSuccess, "[RUNTIME] synInitialize() failed = %d", status);

  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  runtimeManager.SetDevice(device);

  return C_SUCCESS;
}

C_Status SetDevice(const C_Device device) {
  runtimeManager.SetDevice(device);

  return C_SUCCESS;
}

C_Status GetDeviceID(const C_Device device) {
  device->id = static_cast<int>(runtimeManager.GetModuleID());
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << __FUNCTION__ << " : " << device->id;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) {
  runtimeManager.Release(device);

  return C_SUCCESS;
}

C_Status Finalize() {
  synStatus status = synDestroy();
  PD_CHECK(status == synSuccess, "[RUNTIME] synDestroy() failed = %d", status);

  return C_SUCCESS;
}

C_Status GetDevicesCount(size_t *count) {
  runtimeManager.GetNumDevices(count);

  return C_SUCCESS;
}

C_Status GetDevicesList(size_t *devices) {
  uint32_t count = 0;
  synStatus status = synDeviceGetCount(&count);
  PD_CHECK(status == synSuccess,
           "[RUNTIME] synDeviceGetCount() failed = %d",
           status);
  for (size_t dev_id = 0; dev_id < count; dev_id++) {
    devices[dev_id] = dev_id;
  }
  return C_SUCCESS;
}

C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  runtimeManager.Copy(device, dst, src, size, 0);

  return C_SUCCESS;
}

C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  runtimeManager.Copy(device, dst, src, size, 1);

  return C_SUCCESS;
}

C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size) {
  runtimeManager.Copy(device, dst, src, size, 2);

  return C_SUCCESS;
}

C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  runtimeManager.AsyncCopy(device, stream, dst, src, size, 0);

  return C_SUCCESS;
}

C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  runtimeManager.AsyncCopy(device, stream, dst, src, size, 1);

  return C_SUCCESS;
}

C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size) {
  runtimeManager.AsyncCopy(device, stream, dst, src, size, 2);

  return C_SUCCESS;
}

C_Status Allocate_device(const C_Device device, void **ptr, size_t size) {
  uint64_t p;
  synStatus status =
      synDeviceMalloc(runtimeManager.GetDeviceID(), size, 0, 0, &p);
  PD_CHECK(
      status == synSuccess, "[RUNTIME] synDeviceMalloc() failed = %d", status);
  *ptr = reinterpret_cast<void *>(p);
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "device id = " << runtimeManager.GetDeviceID()
      << " malloc ptr=" << *ptr << " size=" << size;

  return C_SUCCESS;
}

C_Status Deallocate_device(const C_Device device, void *ptr, size_t size) {
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "device id=" << runtimeManager.GetDeviceID() << " free ptr = " << ptr
      << " size=" << size;

  synStatus status = synDeviceFree(
      runtimeManager.GetDeviceID(), *reinterpret_cast<uint64_t *>(ptr), 0);

  PD_CHECK(
      status == synSuccess, "[RUNTIME] synDeviceFree() failed = %d", status);

  return C_SUCCESS;
}

C_Status Allocate_host(const C_Device device, void **ptr, size_t size) {
  synStatus status = synHostMalloc(runtimeManager.GetDeviceID(), size, 0, ptr);

  PD_CHECK(
      status == synSuccess, "[RUNTIME] synHostMalloc() failed = %d", status);

  return C_FAILED;
}

C_Status Deallocate_host(const C_Device device, void *ptr, size_t size) {
  synStatus status = synHostFree(runtimeManager.GetDeviceID(), ptr, 0);
  PD_CHECK(status == synSuccess, "[RUNTIME] synHostFree() failed = %d", status);

  return C_SUCCESS;
}

C_Status CreateStream(const C_Device device, C_Stream *stream) {
  runtimeManager.CreateStream(device, stream);

  return C_SUCCESS;
}

C_Status DestroyStream(const C_Device device, C_Stream stream) {
  runtimeManager.DestroyStream(device, stream);
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "device id=" << device->id << " stream=" << stream;

  return C_SUCCESS;
}

C_Status CreateEvent(const C_Device device, C_Event *event) {
  runtimeManager.CreateEvent(device, event);

  return C_SUCCESS;
}

C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event) {
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;
  synStatus status =
      synEventRecord(reinterpret_cast<synEventHandle>(event),
                     reinterpret_cast<const synStreamHandle>(stream));

  PD_CHECK(
      status == synSuccess, "[RUNTIME] synEventRecord() failed = %d", status);

  return C_SUCCESS;
}

C_Status DestroyEvent(const C_Device device, C_Event event) {
  runtimeManager.DestroyEvent(device, event);

  return C_SUCCESS;
}

C_Status SyncDevice(const C_Device device) {
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status = synDeviceSynchronize(runtimeManager.GetDeviceID());
  PD_CHECK(status == synSuccess,
           "[RUNTIME] synDeviceSynchronize() failed = %d",
           status);

  return C_SUCCESS;
}

C_Status SyncStream(const C_Device device, C_Stream stream) {
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << "SyncStream: " << static_cast<synModuleId>(device->id) << ", "
      << reinterpret_cast<const synStreamHandle>(stream);
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status =
      synStreamSynchronize(reinterpret_cast<const synStreamHandle>(stream));

  PD_CHECK(status == synSuccess,
           "[RUNTIME] synStreamSynchronize() failed = %d",
           status);

  return C_SUCCESS;
}

C_Status SyncEvent(const C_Device device, C_Event event) {
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status =
      synEventSynchronize(reinterpret_cast<const synEventHandle>(event));

  PD_CHECK(status == synSuccess,
           "[RUNTIME] synEventSynchronize() failed = %d",
           status);

  return C_SUCCESS;
}

C_Status StreamWaitEvent(const C_Device device,
                         C_Stream stream,
                         C_Event event) {
  LOG_IF(ERROR,
         static_cast<synModuleId>(device->id) != runtimeManager.GetModuleID())
      << "[RUNTIME] moduleID mismatch : moduleID = "
      << runtimeManager.GetModuleID() << ", current = " << device->id;

  synStatus status =
      synStreamWaitEvent(reinterpret_cast<const synStreamHandle>(stream),
                         reinterpret_cast<synEventHandle>(event),
                         0);

  PD_CHECK(status == synSuccess,
           "[RUNTIME] synStreamWaitEvent() failed = %d",
           status);

  return C_SUCCESS;
}

C_Status DeviceMemStats(const C_Device device,
                        size_t *total_memory,
                        size_t *free_memory) {
  runtimeManager.GetMemoryStats(device, total_memory, free_memory);

  return C_SUCCESS;
}

C_Status DeviceMinChunkSize(const C_Device device, size_t *size) {
  *size = 1;
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug) << "min chunksize=" << *size;

  return C_SUCCESS;
}

C_Status XcclGetUniqueIdSize(size_t *sz) {
  runtimeManager.GetUniqueIdSize(sz);
  return C_SUCCESS;
}

C_Status XcclGetUniqueId(C_CCLRootId *unique_id) {
  return runtimeManager.GetUniqueId(unique_id);
}

C_Status XcclCommInitRank(size_t ranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm) {
  runtimeManager.CommInitRank(ranks, unique_id, rank, comm);
  return C_SUCCESS;
}

C_Status XcclDestroyComm(C_CCLComm comm) {
  CHECK_HCCL_STATUS(hcclCommDestroy(reinterpret_cast<hcclComm_t>(comm)));

  return C_SUCCESS;
}

C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream) {
  LOG_IF(INFO, FLAGS_intel_hpu_runtime_debug)
      << send_buf << ", " << recv_buf << ", " << count << ", " << data_type
      << ", " << op << ", " << comm << ", " << stream;
  CHECK_HCCL_STATUS(hcclAllReduce(static_cast<const void *>(send_buf),
                                  recv_buf,
                                  count,
                                  PDDataTypeToHcclDataType(data_type),
                                  PDReduceOpToHcclReduceOp(op),
                                  reinterpret_cast<hcclComm_t>(comm),
                                  reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream) {
  CHECK_HCCL_STATUS(hcclBroadcast(static_cast<const void *>(buf),
                                  buf,
                                  static_cast<uint64_t>(count),
                                  PDDataTypeToHcclDataType(data_type),
                                  static_cast<int>(root),
                                  reinterpret_cast<hcclComm_t>(comm),
                                  reinterpret_cast<synStreamHandle>(stream)));
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
  CHECK_HCCL_STATUS(hcclReduce(static_cast<const void *>(send_buf),
                               recv_buf,
                               count,
                               PDDataTypeToHcclDataType(data_type),
                               PDReduceOpToHcclReduceOp(op),
                               root,
                               reinterpret_cast<hcclComm_t>(comm),
                               reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream) {
  CHECK_HCCL_STATUS(hcclAllGather(static_cast<const void *>(send_buf),
                                  recv_buf,
                                  count,
                                  PDDataTypeToHcclDataType(data_type),
                                  reinterpret_cast<hcclComm_t>(comm),
                                  reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream) {
  CHECK_HCCL_STATUS(
      hcclReduceScatter(static_cast<const void *>(send_buf),
                        recv_buf,
                        count,
                        PDDataTypeToHcclDataType(data_type),
                        PDReduceOpToHcclReduceOp(op),
                        reinterpret_cast<hcclComm_t>(comm),
                        reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclGroupStart() {
  CHECK_HCCL_STATUS(hcclGroupStart());
  return C_SUCCESS;
}

C_Status XcclGroupEnd() {
  CHECK_HCCL_STATUS(hcclGroupEnd());
  return C_SUCCESS;
}

C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  CHECK_HCCL_STATUS(hcclSend(static_cast<const void *>(send_buf),
                             count,
                             PDDataTypeToHcclDataType(data_type),
                             static_cast<uint32_t>(dest_rank),
                             reinterpret_cast<hcclComm_t>(comm),
                             reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream) {
  CHECK_HCCL_STATUS(hcclRecv(recv_buf,
                             count,
                             PDDataTypeToHcclDataType(data_type),
                             static_cast<uint32_t>(src_rank),
                             reinterpret_cast<hcclComm_t>(comm),
                             reinterpret_cast<synStreamHandle>(stream)));
  return C_SUCCESS;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) { return C_SUCCESS; }

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  // auto type = static_cast<synTraceType>(FLAGS_intel_hpu_profiling_type);
  // synStatus status = synProfilerStart(type, runtimeManager.GetDeviceID());
  // PD_CHECK(status == synSuccess,
  //          "[RUNTIME] start intel hpu profiling failed  = %d",
  //          status);

  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  // auto type = static_cast<synTraceType>(FLAGS_intel_hpu_profiling_type);
  // synStatus status = synProfilerStop(type, runtimeManager.GetDeviceID());
  // PD_CHECK(status == synSuccess,
  //          "[RUNTIME] stop intel hpu profiling failed  = %d",
  //          status);

  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data) {
  return C_SUCCESS;
}

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->version.major = 1;
  params->version.minor = 16;
  params->version.patch = 0;
  params->device_type = const_cast<char *>("intel_hpu");
  params->sub_device_type = const_cast<char *>("intel_hpu");

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = InitDevice;
  params->interface->set_device = SetDevice;
  params->interface->get_device = GetDeviceID;
  params->interface->deinit_device = DestroyDevice;

  params->interface->create_stream = CreateStream;
  params->interface->destroy_stream = DestroyStream;

  params->interface->create_event = CreateEvent;
  params->interface->destroy_event = DestroyEvent;
  params->interface->record_event = RecordEvent;

  params->interface->synchronize_device = SyncDevice;
  params->interface->synchronize_stream = SyncStream;
  params->interface->synchronize_event = SyncEvent;
  params->interface->stream_wait_event = StreamWaitEvent;

  params->interface->memory_copy_h2d = MemCpyH2D;
  params->interface->memory_copy_d2d = MemCpyD2D;
  params->interface->memory_copy_d2h = MemCpyD2H;
  // params->interface->memory_copy_p2p = MemCpyP2P;
  params->interface->async_memory_copy_h2d = AsyncMemCpyH2D;
  params->interface->async_memory_copy_d2d = AsyncMemCpyD2D;
  params->interface->async_memory_copy_d2h = AsyncMemCpyD2H;
  // params->interface->async_memory_copy_p2p = AsyncMemCpyP2P;
  params->interface->device_memory_allocate = Allocate_device;
  params->interface->host_memory_allocate = Allocate_host;
  params->interface->device_memory_deallocate = Deallocate_device;
  params->interface->host_memory_deallocate = Deallocate_host;

  params->interface->get_device_count = GetDevicesCount;
  params->interface->get_device_list = GetDevicesList;
  params->interface->device_memory_stats = DeviceMemStats;
  params->interface->device_min_chunk_size = DeviceMinChunkSize;

  params->interface->xccl_all_gather = XcclAllGather;
  params->interface->xccl_all_reduce = XcclAllReduce;
  params->interface->xccl_broadcast = XcclBroadcast;
  params->interface->xccl_comm_init_rank = XcclCommInitRank;
  params->interface->xccl_destroy_comm = XcclDestroyComm;
  params->interface->xccl_get_unique_id = XcclGetUniqueId;
  params->interface->xccl_get_unique_id_size = XcclGetUniqueIdSize;
  params->interface->xccl_group_end = XcclGroupEnd;
  params->interface->xccl_group_start = XcclGroupStart;
  params->interface->xccl_recv = XcclRecv;
  params->interface->xccl_reduce = XcclReduce;
  params->interface->xccl_reduce_scatter = XcclReduceScatter;
  params->interface->xccl_send = XcclSend;

  params->interface->profiler_collect_trace_data = ProfilerCollectData;
  params->interface->profiler_initialize = ProfilerInitialize;
  params->interface->profiler_finalize = ProfilerFinalize;
  params->interface->profiler_start_tracing = ProfilerStart;
  params->interface->profiler_stop_tracing = ProfilerStop;
  params->interface->profiler_prepare_tracing = ProfilerPrepare;
}
