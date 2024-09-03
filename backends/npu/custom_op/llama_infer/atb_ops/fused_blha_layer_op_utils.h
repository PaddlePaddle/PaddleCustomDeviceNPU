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

#pragma once
#ifdef PADDLE_WITH_ATB

#include <iostream>
#include <vector>

#include "atb_layers/fused_blha_layer.h"
#include "atb_layers/runner.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

void init_tensor(const phi::CustomContext &dev_ctx,
                 const phi::DataType &dtype,
                 const std::vector<int64_t> &shape,
                 paddle::Tensor *tensor);

std::shared_ptr<phi::DenseTensor> get_triu_mask(
    const phi::CustomContext &dev_ctx, uint64_t max_seq_len);

class FusedBlhaGlobalVar {
 public:
  struct SeqPack {
    void *dev_ptr = nullptr;
    void *host_ptr = nullptr;
    uint64_t ntokens = 0;
    uint64_t size = 0;
    phi::DenseTensor dev_tensor;
  };

  struct RopePack {
    std::shared_ptr<phi::DenseTensor> rope_emb_cos;
    std::shared_ptr<phi::DenseTensor> rope_emb_sin;
  };

  struct OutPack {
    std::shared_ptr<phi::DenseTensor> first;
    std::shared_ptr<phi::DenseTensor> second;
  };

  struct Pack {
    void *data = nullptr;
    uint64_t size = 0;
  };

  static FusedBlhaGlobalVar &Instance();

  SeqPack *get_seqlens_encoder() { return &g_seqlens_encoder; }

  SeqPack *get_seqlens_decoder() { return &g_seqlens_decoder; }

  Pack *get_block_tables() { return &g_block_tables; }

  Pack *get_batch_status() { return &g_batch_status; }

  RopePack *get_rope_encoder() { return &g_rope_emb_encoder; }

  RopePack *get_rope_decoder() { return &g_rope_emb_decoder; }

  void *get_slots_encoder() { return g_slots_encoder->data(); }

  void *get_slots_decoder() { return g_slots_decoder->data(); }

  void *get_casual_mask() { return g_mask->data(); }

  void *get_alibi_src_mask() { return alibi_src_mask; }
  void *get_alibi_tgt_mask() { return alibi_tgt_mask; }

  OutPack *get_out_encoder() { return &g_out_encoder; }
  OutPack *get_out_decoder() { return &g_out_decoder; }

  void *get_qkv_deq_offset() { return g_qkv_deq_offset->data(); }

  void *get_out_deq_offset() { return g_out_deq_offset->data(); }

  void *get_ffn1_deq_offset() { return g_ffn1_deq_offset->data(); }

  void *get_ffn2_deq_offset() { return g_ffn2_deq_offset->data(); }

  atb_layers::OperationRunner *get_encoder_runner(int64_t idx) {
    return &g_encoder_runners[idx];
  }

  atb_layers::OperationRunner *get_decoder_runner(int64_t idx) {
    return &g_decoder_runners[idx];
  }

  // async d2h + sync + async h2d
  void update_seqlens_encoder(const phi::CustomContext &dev_ctx,
                              const paddle::Tensor &seqlen);

  // async d2h + sync + async h2d
  void update_seqlens_decoder(const phi::CustomContext &dev_ctx,
                              const paddle::Tensor &seqlen);

  // async d2h + sync
  void update_block_tables(const phi::CustomContext &dev_ctx,
                           const paddle::Tensor &block_tables);

  // async d2d
  void update_rope_encoder(const phi::CustomContext &dev_ctx,
                           const paddle::Tensor &rope_emb,
                           int64_t max_seqlen,
                           int64_t head_dim);

  // async d2d
  void update_rope_decoder(const phi::CustomContext &dev_ctx,
                           const paddle::Tensor &rope_emb,
                           int64_t max_seqlen,
                           int64_t head_dim);

  // async h2d
  void update_slots_encoder(const phi::CustomContext &dev_ctx,
                            int64_t block_size,
                            int64_t max_block_num);

  // async h2d
  void update_slots_decoder(const phi::CustomContext &dev_ctx,
                            int64_t block_size,
                            int64_t max_block_num);

  // async d2d
  void update_casual_mask(const phi::CustomContext &dev_ctx,
                          uint64_t max_seq_len);

  void update_alibi_mask(void *src_mask, void *tgt_mask) {
    alibi_src_mask = src_mask;
    alibi_tgt_mask = tgt_mask;
  }

  // async d2d
  void update_in_encoder(const phi::CustomContext &dev_ctx,
                         const paddle::Tensor &hidden);

  // async d2d
  void update_in_decoder(const phi::CustomContext &dev_ctx,
                         const paddle::Tensor &hidden);

  // async d2d
  void update_out_encoder(const phi::CustomContext &dev_ctx,
                          bool,
                          paddle::Tensor *out);

  // async d2d
  void update_out_decoder(const phi::CustomContext &dev_ctx,
                          bool,
                          paddle::Tensor *out);

  void update_qkv_deq_offset(const phi::CustomContext &dev_ctx, int64_t sz);

  void update_out_deq_offset(const phi::CustomContext &dev_ctx, int64_t sz);

  void update_ffn1_deq_offset(const phi::CustomContext &dev_ctx, int64_t sz);

  void update_ffn2_deq_offset(const phi::CustomContext &dev_ctx, int64_t sz);

 private:
  SeqPack g_seqlens_encoder;
  SeqPack g_seqlens_decoder;
  RopePack g_rope_emb_encoder;
  RopePack g_rope_emb_decoder;
  Pack g_batch_status;
  Pack g_block_tables;
  std::shared_ptr<phi::DenseTensor> g_slots_encoder{nullptr};
  std::shared_ptr<phi::DenseTensor> g_slots_decoder{nullptr};
  std::shared_ptr<phi::DenseTensor> g_mask{nullptr};
  OutPack g_out_encoder;
  OutPack g_out_decoder;
  std::shared_ptr<phi::DenseTensor> g_qkv_deq_offset{nullptr};
  std::shared_ptr<phi::DenseTensor> g_out_deq_offset{nullptr};
  std::shared_ptr<phi::DenseTensor> g_ffn1_deq_offset{nullptr};
  std::shared_ptr<phi::DenseTensor> g_ffn2_deq_offset{nullptr};
  std::unordered_map<int64_t, atb_layers::OperationRunner> g_encoder_runners;
  std::unordered_map<int64_t, atb_layers::OperationRunner> g_decoder_runners;
  void *alibi_src_mask;
  void *alibi_tgt_mask;
};

#endif
