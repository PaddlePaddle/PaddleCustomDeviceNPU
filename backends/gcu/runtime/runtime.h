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

#pragma once
#include <tops/tops_ext.h>
#include <topsaten/topsaten.h>

#include <string>

#include "paddle/phi/extension.h"

#define RT_DISALLOW_COPY_AND_ASSIGN(TypeName)     \
  TypeName(const TypeName &) = delete;            \
  TypeName(const TypeName &&) = delete;           \
  TypeName &operator=(const TypeName &) = delete; \
  TypeName &operator=(const TypeName &&) = delete

#define CHECK_COMMON(func, success)                                      \
  do {                                                                   \
    auto ret = (func);                                                   \
    if (ret != success) {                                                \
      std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << ", Call " \
                << #func << " failed, ret:" << ret << std::endl;         \
      exit(-1);                                                          \
    }                                                                    \
  } while (false)

#define RT_CHECK(func) CHECK_COMMON(func, topsSuccess)
#define ECCL_CHECK(func) CHECK_COMMON(func, ecclSuccess)
#define TOPSATEN_CHECK(func) CHECK_COMMON(func, TOPSATEN_STATUS_SUCCESS)
#define TOPSOP_CHECK(func) CHECK_COMMON(func, TOPSOP_STATUS_SUCCESS)

#ifdef __cplusplus
extern "C" {
#endif

void InitPlugin(CustomRuntimeParams *params);

C_Status Init();
C_Status Finalize();
C_Status InitDevice(const C_Device device);
C_Status SetDevice(const C_Device device);
C_Status GetDevice(const C_Device device);
C_Status DeInitDevice(const C_Device device);

C_Status CreateStream(const C_Device device, C_Stream *stream);
C_Status DestroyStream(const C_Device device, C_Stream stream);
C_Status AddCallback(const C_Device device,
                     C_Stream stream,
                     C_Callback callback,
                     void *user_data);

C_Status CreateEvent(const C_Device device, C_Event *event);
C_Status RecordEvent(const C_Device device, C_Stream stream, C_Event event);
C_Status DestroyEvent(const C_Device device, C_Event event);

C_Status SyncDevice(const C_Device device);
C_Status SyncStream(const C_Device device, C_Stream stream);
C_Status SyncEvent(const C_Device device, C_Event event);
C_Status StreamWaitEvent(const C_Device device, C_Stream stream, C_Event event);

C_Status Allocate(const C_Device device, void **ptr, size_t size);
C_Status Deallocate(const C_Device device, void *ptr, size_t size);
C_Status HostAllocate(const C_Device device, void **ptr, size_t size);
C_Status HostDeallocate(const C_Device device, void *ptr, size_t size);
C_Status MemCpyH2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2H(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status MemCpyD2D(const C_Device device,
                   void *dst,
                   const void *src,
                   size_t size);
C_Status AsyncMemCpyH2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2H(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);
C_Status AsyncMemCpyD2D(const C_Device device,
                        C_Stream stream,
                        void *dst,
                        const void *src,
                        size_t size);

C_Status GetDevicesCount(size_t *count);
C_Status GetDevicesList(size_t *device);
C_Status DeviceMinChunkSize(const C_Device device, size_t *size);
C_Status DeviceMaxChunkSize(const C_Device device, size_t *size);
C_Status ExtraPaddingSize(const C_Device device, size_t *size);

C_Status XcclGetUniqueIdSize(size_t *size);
C_Status XcclGetUniqueId(C_CCLRootId *unique_id);
C_Status XcclCommInitRank(size_t nranks,
                          C_CCLRootId *unique_id,
                          size_t rank,
                          C_CCLComm *comm);
C_Status XcclDestroyComm(C_CCLComm comm);
C_Status XcclAllReduce(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLReduceOp op,
                       C_CCLComm comm,
                       C_Stream stream);
C_Status XcclBroadcast(void *buf,
                       size_t count,
                       C_DataType data_type,
                       size_t root,
                       C_CCLComm comm,
                       C_Stream stream);
C_Status XcclReduce(void *send_buf,
                    void *recv_buf,
                    size_t count,
                    C_DataType data_type,
                    C_CCLReduceOp op,
                    size_t root,
                    C_CCLComm comm,
                    C_Stream stream);
C_Status XcclAllGather(void *send_buf,
                       void *recv_buf,
                       size_t count,
                       C_DataType data_type,
                       C_CCLComm comm,
                       C_Stream stream);
C_Status XcclReduceScatter(void *send_buf,
                           void *recv_buf,
                           size_t count,
                           C_DataType data_type,
                           C_CCLReduceOp op,
                           C_CCLComm comm,
                           C_Stream stream);
C_Status XcclGroupStart();
C_Status XcclGroupEnd();
C_Status XcclSend(void *send_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t dest_rank,
                  C_CCLComm comm,
                  C_Stream stream);
C_Status XcclRecv(void *recv_buf,
                  size_t count,
                  C_DataType data_type,
                  size_t src_rank,
                  C_CCLComm comm,
                  C_Stream stream);

C_Status PinnedAllocate(void **ptr, size_t size);

C_Status PinnedDeallocate(void *ptr, size_t size);

void OpsInitialize();

void OpsFinalize();

#ifdef __cplusplus
} /* extern "c" */
#endif
