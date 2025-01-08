// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <omp.h>

#include "kernels/funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T>
void top_p_sampling(const T* probs,
                    const T top_p,
                    const int64_t bs,
                    const int64_t length,
                    int seed,
                    T* top_probs,
                    int64_t* top_ids) {
  std::mt19937 generator(seed);

#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int bi = 0; bi < bs; bi++) {
    const T* start = probs + length * bi;
    std::vector<int> indices(length);
    std::vector<T> probs_sorted(length);

    for (int i = 0; i < length; i++) {
      indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return start[a] > start[b];
    });

    for (int i = 0; i < length; i++) {
      probs_sorted[i] = start[indices[i]];
    }

    T prob_sum = 0.0;
    int prob_end;
    for (int i = 0; i < length; i++) {
      if (prob_sum > top_p) {
        prob_end = i;
        break;
      }
      prob_sum += probs_sorted[i];
    }
    if (prob_end == 0) {
      prob_end = 1;
    }

    std::discrete_distribution<int> distribution(
        probs_sorted.begin(), probs_sorted.begin() + prob_end);

    auto predict = distribution(generator);
    top_ids[bi] = indices[predict];
    top_probs[bi] = probs_sorted[predict];
  }
}

}  // namespace custom_kernel

namespace custom_kernel {
template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& ps,
                        const paddle::optional<phi::DenseTensor>& threshold,
                        const paddle::optional<phi::DenseTensor>& topp_seed,
                        int seed,
                        int k,
                        const std::string& mode,
                        phi::DenseTensor* out,
                        phi::DenseTensor* ids,
                        phi::DenseTensor* topk_scores,
                        phi::DenseTensor* topk_ids) {
  auto x_dims = phi::vectorize<int64_t>(x.dims());
  auto meta = x.meta();
  int bs = x_dims[0];
  int length = x_dims[1];

  auto out_data = dev_ctx.template Alloc<T>(out);
  auto ids_data = dev_ctx.template Alloc<int64_t>(ids);

  top_p_sampling(const_cast<T*>(x.data<T>()),
                 *(const_cast<T*>(ps.data<T>())),
                 bs,
                 length,
                 seed,
                 out_data,
                 ids_data);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    top_p_sampling, CPU, ALL_LAYOUT, custom_kernel::TopPSamplingKernel, float) {
}
