// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>
#include <tops.h>
#include <tops/tops_runtime.h>

__global__ void vec_add_kernel(int *from, int *to, size_t N) {
  printf("====> vec_add_kernel start\n");
  tops_dte_ctx_t ctx;
  tops::dte_scope s(ctx);
  __valigned__ int buffer[128];

  tops::mdspan buf(tops::Private, &buffer, 128);

  for (size_t i = 0; i < N; i += 128) {
    tops::mdspan src(tops::Global, from + i, 128);
    tops::mdspan dst(tops::Global, to + i, 128);
    tops::memcpy(ctx, buf, src);
    for (size_t j = 0; j < 128; j += tops::vlength<vint>()) {
      const auto &v = tops::vload<vint>(buffer + j);
      tops::vstore(tops::vadd(v, v), buffer + j);
    }
    tops::memcpy(ctx, dst, buf);
  }
  printf("====> vec_add_kernel finish\n");
}

void vec_add_cpp(int *from, int *to, size_t N) {
  printf("====> %s:%d, %s\n", __FILE__, __LINE__, __FUNCTION__);
  vec_add_kernel<<<1, 1>>>(from, to, N);
  return;
}
