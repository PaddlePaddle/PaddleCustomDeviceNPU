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

#ifdef PADDLE_WITH_ATB

#include "fused_blha_layer_op_utils.h"  // NOLINT

namespace custom_kernel {
template <typename T, typename Context>
void FullKernel(const Context &dev_ctx,
                const phi::IntArray &shape,
                const phi::Scalar &val,
                phi::DataType dtype,
                phi::DenseTensor *out);

template <typename T, typename Context>
void TrilKernel(const Context &ctx,
                const phi::DenseTensor &x,
                int diagonal,
                phi::DenseTensor *out);

template <typename T, typename Context>
void ScaleKernel(const Context &dev_ctx,
                 const phi::DenseTensor &x,
                 const phi::Scalar &in_scale,
                 const phi::Scalar &in_bias,
                 bool bias_after_scale,
                 phi::DenseTensor *out);
}  // namespace custom_kernel

void init_tensor(const phi::CustomContext &dev_ctx,
                 const phi::DataType &dtype,
                 const std::vector<int64_t> &shape,
                 paddle::Tensor *tensor) {
  phi::DenseTensorMeta meta(dtype, phi::make_ddim(shape));
  static_cast<phi::DenseTensor *>(tensor->impl().get())->set_meta(meta);
  dev_ctx.Alloc(static_cast<phi::DenseTensor *>(tensor->impl().get()), dtype);
}

FusedBlhaGlobalVar &FusedBlhaGlobalVar::Instance() {
  static FusedBlhaGlobalVar ins;
  return ins;
}

void FusedBlhaGlobalVar::update_seqlens_encoder(
    const phi::CustomContext &dev_ctx, const paddle::Tensor &seqlen) {
  static void *g_seqlens_encoder_int64 = nullptr;
  // init
  if (!g_seqlens_encoder_int64) {
    g_seqlens_encoder.size = seqlen.numel();
    ACL_CHECK(aclrtMallocHost(&g_seqlens_encoder.host_ptr,
                              g_seqlens_encoder.size * sizeof(int32_t)));
    ACL_CHECK(aclrtMallocHost(&g_seqlens_encoder_int64,
                              g_seqlens_encoder.size * sizeof(int64_t)));
    g_seqlens_encoder.dev_tensor.Resize({g_seqlens_encoder.size});
    dev_ctx.template Alloc<int32_t>(&g_seqlens_encoder.dev_tensor);
    g_seqlens_encoder.dev_ptr = g_seqlens_encoder.dev_tensor.data();
  }

  // update
  if (seqlen.dtype() == phi::DataType::INT32) {
    ACL_CHECK(
        aclrtMemcpyAsync(g_seqlens_encoder.host_ptr,
                         g_seqlens_encoder.size * sizeof(int32_t),
                         seqlen.data(),
                         g_seqlens_encoder.size * sizeof(int32_t),
                         ACL_MEMCPY_DEVICE_TO_HOST,
                         reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    ACL_CHECK(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(dev_ctx.stream())));
  } else {
    ACL_CHECK(
        aclrtMemcpyAsync(g_seqlens_encoder_int64,
                         g_seqlens_encoder.size * sizeof(int64_t),
                         seqlen.data(),
                         g_seqlens_encoder.size * sizeof(int64_t),
                         ACL_MEMCPY_DEVICE_TO_HOST,
                         reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    ACL_CHECK(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    for (auto i = 0; i < g_seqlens_encoder.size; ++i) {
      reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr)[i] =
          static_cast<int32_t>(
              reinterpret_cast<int64_t *>(g_seqlens_encoder_int64)[i]);
    }
  }

  // calc ntokens
  auto *g_seqlens_encoder_host = reinterpret_cast<int32_t *>(
      reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr));
  g_seqlens_encoder.ntokens = 0;
  for (auto i = 0; i < g_seqlens_encoder.size; ++i) {
    if (g_seqlens_encoder_host[i] > 0) {
      g_seqlens_encoder.ntokens += g_seqlens_encoder_host[i];
    }
  }

  // copy to device
  ACL_CHECK(aclrtMemcpyAsync(g_seqlens_encoder.dev_ptr,
                             g_seqlens_encoder.size * sizeof(int32_t),
                             g_seqlens_encoder.host_ptr,
                             g_seqlens_encoder.size * sizeof(int32_t),
                             ACL_MEMCPY_HOST_TO_DEVICE,
                             reinterpret_cast<aclrtStream>(dev_ctx.stream())));
}

void FusedBlhaGlobalVar::update_seqlens_decoder(
    const phi::CustomContext &dev_ctx, const paddle::Tensor &seqlen) {
  static void *g_seqlens_decoder_int64 = nullptr;
  // init
  if (!g_seqlens_decoder_int64) {
    g_seqlens_decoder.size = seqlen.numel();
    ACL_CHECK(aclrtMallocHost(&g_seqlens_decoder.host_ptr,
                              g_seqlens_decoder.size * sizeof(int32_t)));
    g_batch_status.size = seqlen.numel();
    ACL_CHECK(aclrtMallocHost(&g_batch_status.data,
                              g_batch_status.size * sizeof(int32_t)));
    ACL_CHECK(aclrtMallocHost(&g_seqlens_decoder_int64,
                              g_seqlens_decoder.size * sizeof(int64_t)));
    g_seqlens_decoder.dev_tensor.Resize({g_seqlens_decoder.size});
    dev_ctx.template Alloc<int32_t>(&g_seqlens_decoder.dev_tensor);
    g_seqlens_decoder.dev_ptr = g_seqlens_decoder.dev_tensor.data();
  }

  // update
  if (seqlen.dtype() == phi::DataType::INT32) {
    ACL_CHECK(
        aclrtMemcpyAsync(g_seqlens_decoder.host_ptr,
                         g_seqlens_decoder.size * sizeof(int32_t),
                         seqlen.data(),
                         g_seqlens_decoder.size * sizeof(int32_t),
                         ACL_MEMCPY_DEVICE_TO_HOST,
                         reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    ACL_CHECK(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(dev_ctx.stream())));
  } else {
    ACL_CHECK(
        aclrtMemcpyAsync(g_seqlens_decoder_int64,
                         g_seqlens_decoder.size * sizeof(int64_t),
                         seqlen.data(),
                         g_seqlens_decoder.size * sizeof(int64_t),
                         ACL_MEMCPY_DEVICE_TO_HOST,
                         reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    ACL_CHECK(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    for (auto i = 0; i < g_seqlens_decoder.size; ++i) {
      reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr)[i] =
          static_cast<int32_t>(
              reinterpret_cast<int64_t *>(g_seqlens_decoder_int64)[i]);
    }
  }

  // calc ntokens
  auto *g_seqlens_decoder_host = reinterpret_cast<int32_t *>(
      reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr));
  auto *g_batch_status_host = reinterpret_cast<int32_t *>(
      reinterpret_cast<int32_t *>(g_batch_status.data));
  g_seqlens_decoder.ntokens = 0;
  for (auto i = 0; i < g_seqlens_decoder.size; ++i) {
    g_batch_status_host[i] = 0;
    if (g_seqlens_decoder_host[i] > 0) {
      g_seqlens_decoder_host[i] += 1;
      g_seqlens_decoder.ntokens += 1;
      g_batch_status_host[i] = 1;
    }
  }

  // copy to device
  ACL_CHECK(aclrtMemcpyAsync(g_seqlens_decoder.dev_ptr,
                             g_seqlens_decoder.size * sizeof(int32_t),
                             g_seqlens_decoder.host_ptr,
                             g_seqlens_decoder.size * sizeof(int32_t),
                             ACL_MEMCPY_HOST_TO_DEVICE,
                             reinterpret_cast<aclrtStream>(dev_ctx.stream())));
}

void FusedBlhaGlobalVar::update_block_tables(
    const phi::CustomContext &dev_ctx, const paddle::Tensor &block_tables) {
  static void *g_block_tables_int64 = nullptr;
  // init
  if (!g_block_tables_int64) {
    g_block_tables.size = block_tables.numel();
    ACL_CHECK(aclrtMallocHost(&g_block_tables.data,
                              g_block_tables.size * sizeof(int32_t)));
    ACL_CHECK(aclrtMallocHost(&g_block_tables_int64,
                              g_block_tables.size * sizeof(int64_t)));
  }

  // update
  if (block_tables.dtype() == phi::DataType::INT32) {
    ACL_CHECK(
        aclrtMemcpyAsync(g_block_tables.data,
                         g_block_tables.size * sizeof(int32_t),
                         block_tables.data(),
                         g_block_tables.size * sizeof(int32_t),
                         ACL_MEMCPY_DEVICE_TO_HOST,
                         reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    ACL_CHECK(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(dev_ctx.stream())));
  } else {
    ACL_CHECK(
        aclrtMemcpyAsync(g_block_tables_int64,
                         g_block_tables.size * sizeof(int64_t),
                         block_tables.data(),
                         g_block_tables.size * sizeof(int64_t),
                         ACL_MEMCPY_DEVICE_TO_HOST,
                         reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    ACL_CHECK(aclrtSynchronizeStream(
        reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    for (auto i = 0; i < g_block_tables.size; ++i) {
      reinterpret_cast<int32_t *>(g_block_tables.data)[i] =
          static_cast<int32_t>(
              reinterpret_cast<int64_t *>(g_block_tables_int64)[i]);
    }
  }
}

void FusedBlhaGlobalVar::update_rope_encoder(const phi::CustomContext &dev_ctx,
                                             const paddle::Tensor &rope_emb,
                                             int64_t max_seqlen,
                                             int64_t head_dim) {
  if (g_seqlens_encoder.ntokens == 0) {
    return;
  }
  // init
  g_rope_emb_encoder.rope_emb_cos = std::make_shared<phi::DenseTensor>();
  g_rope_emb_encoder.rope_emb_cos->Resize(
      {g_seqlens_encoder.ntokens, head_dim});
  dev_ctx.template Alloc<phi::float16>(g_rope_emb_encoder.rope_emb_cos.get());

  g_rope_emb_encoder.rope_emb_sin = std::make_shared<phi::DenseTensor>();
  g_rope_emb_encoder.rope_emb_sin->Resize(
      {g_seqlens_encoder.ntokens, head_dim});
  dev_ctx.template Alloc<phi::float16>(g_rope_emb_encoder.rope_emb_sin.get());

  // update
  C_Device_st device{dev_ctx.GetPlace().GetDeviceId()};
  C_Stream stream =
      const_cast<C_Stream>(reinterpret_cast<const C_Stream>(dev_ctx.stream()));

  void *new_cos_data = g_rope_emb_encoder.rope_emb_cos->data();
  void *new_sin_data = g_rope_emb_encoder.rope_emb_sin->data();
  void *cos_data = const_cast<void *>(rope_emb.data());
  void *sin_data =
      cos_data + rope_emb.numel() / 2 * phi::SizeOf(phi::DataType::FLOAT16);

  uint64_t out_offset = 0;
  uint64_t in_offset = 0;
  uint64_t numel = 0;
  int32_t *seqlens = reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr);
  uint64_t seqlens_size = g_seqlens_encoder.size;
  for (auto i = 0; i < seqlens_size; ++i) {
    if (seqlens[i] > 0) {
      out_offset += numel;
      numel = seqlens[i] * head_dim;
      AsyncMemCpyD2D(
          &device,
          stream,
          new_cos_data + out_offset * phi::SizeOf(phi::DataType::FLOAT16),
          cos_data + in_offset * phi::SizeOf(phi::DataType::FLOAT16),
          numel * phi::SizeOf(phi::DataType::FLOAT16));
      AsyncMemCpyD2D(
          &device,
          stream,
          new_sin_data + out_offset * phi::SizeOf(phi::DataType::FLOAT16),
          sin_data + in_offset * phi::SizeOf(phi::DataType::FLOAT16),
          numel * phi::SizeOf(phi::DataType::FLOAT16));
    }
  }
}

void FusedBlhaGlobalVar::update_rope_decoder(const phi::CustomContext &dev_ctx,
                                             const paddle::Tensor &rope_emb,
                                             int64_t max_seqlen,
                                             int64_t head_dim) {
  if (g_seqlens_decoder.ntokens == 0) {
    return;
  }
  // init
  g_rope_emb_decoder.rope_emb_cos = std::make_shared<phi::DenseTensor>();
  g_rope_emb_decoder.rope_emb_cos->Resize(
      {g_seqlens_decoder.ntokens, head_dim});
  dev_ctx.template Alloc<phi::float16>(g_rope_emb_decoder.rope_emb_cos.get());

  g_rope_emb_decoder.rope_emb_sin = std::make_shared<phi::DenseTensor>();
  g_rope_emb_decoder.rope_emb_sin->Resize(
      {g_seqlens_decoder.ntokens, head_dim});
  dev_ctx.template Alloc<phi::float16>(g_rope_emb_decoder.rope_emb_sin.get());

  // update
  C_Device_st device{dev_ctx.GetPlace().GetDeviceId()};
  C_Stream stream =
      const_cast<C_Stream>(reinterpret_cast<const C_Stream>(dev_ctx.stream()));

  void *new_cos_data = g_rope_emb_decoder.rope_emb_cos->data();
  void *new_sin_data = g_rope_emb_decoder.rope_emb_sin->data();
  void *cos_data = const_cast<void *>(rope_emb.data());
  void *sin_data =
      cos_data + rope_emb.numel() / 2 * phi::SizeOf(phi::DataType::FLOAT16);

  uint64_t out_offset = 0;
  uint64_t in_offset = 0;
  uint64_t numel = 0;
  int32_t *seqlens = reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr);
  uint64_t seqlens_size = g_seqlens_decoder.size;
  for (auto i = 0; i < seqlens_size; ++i) {
    if (seqlens[i] > 0) {
      in_offset = (seqlens[i] - 1) * head_dim;
      out_offset += numel;
      numel = head_dim;
      AsyncMemCpyD2D(
          &device,
          stream,
          new_cos_data + out_offset * phi::SizeOf(phi::DataType::FLOAT16),
          cos_data + in_offset * phi::SizeOf(phi::DataType::FLOAT16),
          numel * phi::SizeOf(phi::DataType::FLOAT16));
      AsyncMemCpyD2D(
          &device,
          stream,
          new_sin_data + out_offset * phi::SizeOf(phi::DataType::FLOAT16),
          sin_data + in_offset * phi::SizeOf(phi::DataType::FLOAT16),
          numel * phi::SizeOf(phi::DataType::FLOAT16));
    }
  }
}

void FusedBlhaGlobalVar::update_slots_encoder(const phi::CustomContext &dev_ctx,
                                              int64_t block_size,
                                              int64_t max_block_num) {
  if (g_seqlens_encoder.ntokens == 0) {
    return;
  }
  static int32_t *g_slots = nullptr;
  static int64_t g_slots_size = 0;
  // init
  if (g_slots_size < g_seqlens_encoder.ntokens) {
    g_slots_size = g_seqlens_encoder.ntokens;
    if (g_slots != nullptr) {
      ACL_CHECK(aclrtFreeHost(g_slots));
      g_slots = nullptr;
    }
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&g_slots),
                              g_slots_size * sizeof(int32_t)));
    g_slots_encoder = std::make_shared<phi::DenseTensor>();
    g_slots_encoder->Resize({g_slots_size});
    dev_ctx.template Alloc<int32_t>(g_slots_encoder.get());
  }

  // update
  int64_t idx = 0;
  int64_t block_offset = 0;
  int32_t *block_tables = reinterpret_cast<int32_t *>(g_block_tables.data);
  int32_t *seqlens = reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr);
  uint64_t seqlens_size = g_seqlens_encoder.size;
  for (int64_t i = 0; i < seqlens_size; ++i) {
    int64_t len = seqlens[i];
    if (len > 0) {
      int64_t need_block_num = len / block_size;
      int64_t tail_len = len % block_size;
      int64_t slot_offset = 0;
      for (int64_t j = 0; j < need_block_num; ++j) {
        slot_offset = block_tables[block_offset + j] * block_size;
        for (int64_t k = 0; k < block_size; ++k) {
          g_slots[idx++] = slot_offset + k;
        }
        len -= block_size;
      }
      slot_offset = block_tables[block_offset + need_block_num] * block_size;
      for (int64_t k = 0; k < tail_len; ++k) {
        g_slots[idx++] = slot_offset + k;
      }
    }
    block_offset += max_block_num;
  }

  // copy to device
  ACL_CHECK(aclrtMemcpyAsync(g_slots_encoder->data(),
                             g_seqlens_encoder.ntokens * sizeof(int32_t),
                             g_slots,
                             g_seqlens_encoder.ntokens * sizeof(int32_t),
                             ACL_MEMCPY_HOST_TO_DEVICE,
                             reinterpret_cast<aclrtStream>(dev_ctx.stream())));
}

void FusedBlhaGlobalVar::update_slots_decoder(const phi::CustomContext &dev_ctx,
                                              int64_t block_size,
                                              int64_t max_block_num) {
  if (g_seqlens_decoder.ntokens == 0) {
    return;
  }
  static int32_t *g_slots = nullptr;
  static int64_t g_slots_size = 0;
  // init
  if (g_slots_size < g_seqlens_decoder.ntokens) {
    g_slots_size = g_seqlens_decoder.ntokens;
    if (g_slots != nullptr) {
      ACL_CHECK(aclrtFreeHost(g_slots));
      g_slots = nullptr;
    }
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&g_slots),
                              g_slots_size * sizeof(int32_t)));
    g_slots_decoder = std::make_shared<phi::DenseTensor>();
    g_slots_decoder->Resize({g_slots_size});
    dev_ctx.template Alloc<int32_t>(g_slots_decoder.get());
  }

  // update
  int64_t idx = 0;
  int64_t block_offset = 0;
  int32_t *block_tables = reinterpret_cast<int32_t *>(g_block_tables.data);
  int32_t *seqlens = reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr);
  uint64_t seqlens_size = g_seqlens_decoder.size;
  for (int64_t i = 0; i < seqlens_size; i++) {
    int64_t len = seqlens[i];
    if (len > 0) {
      int64_t need_block_num = (len - 1) / block_size;
      int64_t tail_len = (len - 1) % block_size;
      int64_t slot_offset =
          block_tables[block_offset + need_block_num] * block_size;
      g_slots[idx++] = slot_offset + tail_len;
    }
    block_offset += max_block_num;
  }

  // copy to device
  ACL_CHECK(aclrtMemcpyAsync(g_slots_decoder->data(),
                             g_seqlens_decoder.ntokens * sizeof(int32_t),
                             g_slots,
                             g_seqlens_decoder.ntokens * sizeof(int32_t),
                             ACL_MEMCPY_HOST_TO_DEVICE,
                             reinterpret_cast<aclrtStream>(dev_ctx.stream())));
}

void FusedBlhaGlobalVar::update_casual_mask(const phi::CustomContext &dev_ctx,
                                            uint64_t max_seq_len) {
  if (!g_mask.get()) {
    g_mask = std::make_shared<phi::DenseTensor>();
  }
  if (g_mask->numel() != max_seq_len * max_seq_len) {
    LOG(INFO) << "update_mask: max_seq_len=" << max_seq_len;
    g_mask->Resize({max_seq_len, max_seq_len});
    dev_ctx.template Alloc<phi::float16>(g_mask.get());

    phi::DenseTensor ones_tensor;
    custom_kernel::FullKernel<phi::float16>(dev_ctx,
                                            {max_seq_len, max_seq_len},
                                            1.0f,
                                            phi::DataType::FLOAT16,
                                            &ones_tensor);

    phi::DenseTensor tril_ones_tensor;
    tril_ones_tensor.Resize({max_seq_len, max_seq_len});
    custom_kernel::TrilKernel<phi::float16>(
        dev_ctx, ones_tensor, 0, &tril_ones_tensor);

    phi::DenseTensor tmp_mask;
    tmp_mask.Resize({max_seq_len, max_seq_len});
    custom_kernel::ScaleKernel<phi::float16>(
        dev_ctx, tril_ones_tensor, 1.0f, -1.0f, true, &tmp_mask);
    // use 50000 to avoid overflow
    custom_kernel::ScaleKernel<phi::float16>(
        dev_ctx, tmp_mask, 50000.0f, 0.0f, true, g_mask.get());
  }
}

void FusedBlhaGlobalVar::update_in_encoder(const phi::CustomContext &dev_ctx,
                                           const paddle::Tensor &hidden) {
  if (g_seqlens_encoder.ntokens == 0) {
    return;
  }
  auto hidden_shape = hidden.shape();
  auto ntokens = g_seqlens_encoder.ntokens;
  auto emb_dim = hidden_shape[1];
  // init
  g_out_encoder.first = std::make_shared<phi::DenseTensor>();
  g_out_encoder.first->Resize({ntokens, emb_dim});
  dev_ctx.template Alloc<phi::float16>(g_out_encoder.first.get());

  g_out_encoder.second = std::make_shared<phi::DenseTensor>();
  g_out_encoder.second->Resize({ntokens, emb_dim});
  dev_ctx.template Alloc<phi::float16>(g_out_encoder.second.get());

  // udpate
  void *in_data = const_cast<void *>(hidden.data());
  void *out_data = g_out_encoder.first->data();
  int32_t batch_size = g_seqlens_encoder.size;
  int32_t *seqlens_encoder =
      reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr);
  int32_t *seqlens_decoder =
      reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr);

  int64_t in_offset = 0, out_offset = 0, numel = 0;
  for (auto i = 0; i < batch_size; ++i) {
    if (seqlens_encoder[i] > 0) {
      numel = seqlens_encoder[i] * emb_dim;
      ACL_CHECK(
          aclrtMemcpyAsync(out_data + out_offset * sizeof(phi::float16),
                           numel * sizeof(phi::float16),
                           in_data + in_offset * sizeof(phi::float16),
                           numel * sizeof(phi::float16),
                           ACL_MEMCPY_DEVICE_TO_DEVICE,
                           reinterpret_cast<aclrtStream>(dev_ctx.stream())));
      in_offset += numel;
      out_offset += numel;
    } else if (seqlens_decoder[i] > 0) {
      in_offset += emb_dim;
    }
  }
}

void FusedBlhaGlobalVar::update_in_decoder(const phi::CustomContext &dev_ctx,
                                           const paddle::Tensor &hidden) {
  if (g_seqlens_decoder.ntokens == 0) {
    return;
  }
  auto hidden_shape = hidden.shape();
  auto ntokens = g_seqlens_decoder.ntokens;
  auto emb_dim = hidden_shape[1];
  // init
  g_out_decoder.first = std::make_shared<phi::DenseTensor>();
  g_out_decoder.first->Resize({ntokens, emb_dim});
  dev_ctx.template Alloc<phi::float16>(g_out_decoder.first.get());

  g_out_decoder.second = std::make_shared<phi::DenseTensor>();
  g_out_decoder.second->Resize({ntokens, emb_dim});
  dev_ctx.template Alloc<phi::float16>(g_out_decoder.second.get());

  // udpate
  void *in_data = const_cast<void *>(hidden.data());
  void *out_data = g_out_decoder.first->data();
  int32_t batch_size = g_seqlens_decoder.size;
  int32_t *seqlens_encoder =
      reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr);
  int32_t *seqlens_decoder =
      reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr);

  int64_t in_offset = 0, out_offset = 0, numel = 0;
  for (auto i = 0; i < batch_size; ++i) {
    if (seqlens_encoder[i] > 0) {
      in_offset += seqlens_encoder[i] * emb_dim;
    } else if (seqlens_decoder[i] > 0) {
      numel = emb_dim;
      ACL_CHECK(
          aclrtMemcpyAsync(out_data + out_offset * sizeof(phi::float16),
                           numel * sizeof(phi::float16),
                           in_data + in_offset * sizeof(phi::float16),
                           numel * sizeof(phi::float16),
                           ACL_MEMCPY_DEVICE_TO_DEVICE,
                           reinterpret_cast<aclrtStream>(dev_ctx.stream())));
      in_offset += emb_dim;
      out_offset += emb_dim;
    }
  }
}

void FusedBlhaGlobalVar::update_out_encoder(const phi::CustomContext &dev_ctx,
                                            bool first_or_second,
                                            paddle::Tensor *out) {
  if (g_seqlens_encoder.ntokens == 0) {
    return;
  }
  auto out_shape = out->shape();
  auto emb_dim = out_shape[1];

  // udpate
  void *in_data = first_or_second ? g_out_encoder.first->data()
                                  : g_out_encoder.second->data();
  void *out_data = out->data();
  int32_t batch_size = g_seqlens_encoder.size;
  int32_t *seqlens_encoder =
      reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr);
  int32_t *seqlens_decoder =
      reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr);

  int64_t in_offset = 0, out_offset = 0;
  for (auto i = 0; i < batch_size; ++i) {
    if (seqlens_encoder[i] > 0) {
      in_offset += seqlens_encoder[i] * emb_dim;
      out_offset = i * emb_dim;
      ACL_CHECK(aclrtMemcpyAsync(
          out_data + out_offset * sizeof(phi::float16),
          emb_dim * sizeof(phi::float16),
          in_data + (in_offset - emb_dim) * sizeof(phi::float16),
          emb_dim * sizeof(phi::float16),
          ACL_MEMCPY_DEVICE_TO_DEVICE,
          reinterpret_cast<aclrtStream>(dev_ctx.stream())));
    }
  }
}

void FusedBlhaGlobalVar::update_out_decoder(const phi::CustomContext &dev_ctx,
                                            bool first_or_second,
                                            paddle::Tensor *out) {
  if (g_seqlens_decoder.ntokens == 0) {
    return;
  }
  auto out_shape = out->shape();
  auto emb_dim = out_shape[1];

  // udpate
  void *in_data = first_or_second ? g_out_decoder.first->data()
                                  : g_out_decoder.second->data();
  void *out_data = out->data();
  int32_t batch_size = g_seqlens_encoder.size;
  int32_t *seqlens_encoder =
      reinterpret_cast<int32_t *>(g_seqlens_encoder.host_ptr);
  int32_t *seqlens_decoder =
      reinterpret_cast<int32_t *>(g_seqlens_decoder.host_ptr);

  int64_t in_offset = 0, out_offset = 0;
  for (auto i = 0; i < batch_size; ++i) {
    if (seqlens_decoder[i] > 0) {
      out_offset = i * emb_dim;
      ACL_CHECK(
          aclrtMemcpyAsync(out_data + out_offset * sizeof(phi::float16),
                           emb_dim * sizeof(phi::float16),
                           in_data + in_offset * sizeof(phi::float16),
                           emb_dim * sizeof(phi::float16),
                           ACL_MEMCPY_DEVICE_TO_DEVICE,
                           reinterpret_cast<aclrtStream>(dev_ctx.stream())));
      in_offset += emb_dim;
    }
  }
}

void FusedBlhaGlobalVar::update_qkv_deq_offset(
    const phi::CustomContext &dev_ctx, int64_t sz) {
  if (!g_qkv_deq_offset.get()) {
    g_qkv_deq_offset = std::make_shared<phi::DenseTensor>();
    g_qkv_deq_offset->Resize({sz});
    custom_kernel::FullKernel<int32_t>(
        dev_ctx, {sz}, 0, phi::DataType::INT32, g_qkv_deq_offset.get());
  }
}

void FusedBlhaGlobalVar::update_out_deq_offset(
    const phi::CustomContext &dev_ctx, int64_t sz) {
  if (!g_out_deq_offset.get()) {
    g_out_deq_offset = std::make_shared<phi::DenseTensor>();
    g_out_deq_offset->Resize({sz});
    custom_kernel::FullKernel<int32_t>(
        dev_ctx, {sz}, 0, phi::DataType::INT32, g_out_deq_offset.get());
  }
}

void FusedBlhaGlobalVar::update_ffn1_deq_offset(
    const phi::CustomContext &dev_ctx, int64_t sz) {
  if (!g_ffn1_deq_offset.get()) {
    g_ffn1_deq_offset = std::make_shared<phi::DenseTensor>();
    g_ffn1_deq_offset->Resize({sz});
    custom_kernel::FullKernel<int32_t>(
        dev_ctx, {sz}, 0, phi::DataType::INT32, g_ffn1_deq_offset.get());
  }
}

void FusedBlhaGlobalVar::update_ffn2_deq_offset(
    const phi::CustomContext &dev_ctx, int64_t sz) {
  if (!g_ffn2_deq_offset.get()) {
    g_ffn2_deq_offset = std::make_shared<phi::DenseTensor>();
    g_ffn2_deq_offset->Resize({sz});
    custom_kernel::FullKernel<int32_t>(
        dev_ctx, {sz}, 0, phi::DataType::INT32, g_ffn2_deq_offset.get());
  }
}

#endif
