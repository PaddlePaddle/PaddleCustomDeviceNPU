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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "paddle/extension.h"

void NgramMatch(const paddle::Tensor& input_ids,
                const paddle::Tensor& input_ids_len,
                const paddle::Tensor& pre_ids,
                const paddle::Tensor& step_idx,
                const paddle::Tensor& draft_token_num,
                const paddle::Tensor& draft_tokens,
                const paddle::Tensor& seq_lens_this_time,
                const paddle::Tensor& seq_lens_encoder,
                const paddle::Tensor& seq_lens_decoder,
                const int real_batch_size,
                const int max_ngram_size,
                const int max_draft_tokens) {}

PD_BUILD_OP(ngram_match)
    .Inputs({"input_ids",
             "input_ids_len",
             "pre_ids",
             "step_idx",
             "draft_token_num",
             "draft_tokens",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder"})
    .Attrs({"real_batch_size: int",
            "max_ngram_size: int",
            "max_draft_tokens: int"})
    .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
    .SetKernelFn(PD_KERNEL(NgramMatch))
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"}});

std::vector<paddle::Tensor> TopPCandidates(
    const paddle::Tensor& probs,
    const paddle::Tensor& top_p,
    const paddle::Tensor& output_padding_offset,
    int candidates_len,
    int max_seq_len) {
  return {top_p};
}

std::vector<std::vector<int64_t>> TopPCandidatesInferShape(
    const std::vector<int64_t>& probs_shape,
    const std::vector<int64_t>& top_p_shape,
    const std::vector<int64_t>& output_padding_offset_shape,
    int max_candidates_len) {
  int token_num = probs_shape[0];
  return {{token_num, max_candidates_len},
          {token_num, max_candidates_len},
          {token_num}};
}

std::vector<paddle::DataType> TopPCandidatesInferDtype(
    const paddle::DataType& probs_dtype,
    const paddle::DataType& top_p_dtype,
    const paddle::DataType& output_padding_offset_dtype) {
  return {probs_dtype, paddle::DataType::INT64, paddle::DataType::INT32};
}

PD_BUILD_OP(top_p_candidates)
    .Inputs({"probs", "top_p", "output_padding_offset"})
    .Outputs({"verify_scores", "verify_tokens", "actual_candidate_lens"})
    .Attrs({"candidates_len: int", "max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(TopPCandidates))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPCandidatesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPCandidatesInferDtype));

void SpeculateVerifyAndUpdate(const paddle::Tensor& accept_tokens,
                              const paddle::Tensor& accept_num,
                              const paddle::Tensor& step_idx,
                              const paddle::Tensor& seq_lens_encoder,
                              const paddle::Tensor& seq_lens_decoder,
                              const paddle::Tensor& stop_flags,
                              const paddle::Tensor& not_need_stop,
                              const paddle::Tensor& draft_tokens,
                              const paddle::Tensor& seq_lens_this_time,
                              const paddle::Tensor& verify_tokens,
                              const paddle::Tensor& verify_scores,
                              const paddle::Tensor& max_dec_len,
                              const paddle::Tensor& end_tokens,
                              const paddle::Tensor& is_block_step,
                              const paddle::Tensor& output_cum_offsets,
                              const paddle::Tensor& actual_candidate_len,
                              const paddle::Tensor& actual_draft_token_nums,
                              const paddle::Tensor& topp,
                              int max_seq_len,
                              int verify_window,
                              bool enable_topp) {}

PD_BUILD_OP(speculate_verify_and_update)
    .Inputs({"accept_tokens",
             "accept_num",
             "step_idx",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "stop_flags",
             "not_need_stop",
             "draft_tokens",
             "seq_lens_this_time",
             "verify_tokens",
             "verify_scores",
             "max_dec_len",
             "end_tokens",
             "is_block_step",
             "output_cum_offsets",
             "actual_candidate_len",
             "actual_draft_token_nums",
             "topp"})
    .Outputs({"accept_tokens_out",
              "accept_num_out",
              "step_idx_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "stop_flags_out",
              "not_need_stop_out",
              "draft_tokens_out"})
    .Attrs({"max_seq_len: int", "verify_window: int", "enable_topp: bool"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"},
                    {"step_idx", "step_idx_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"draft_tokens", "draft_tokens_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateVerifyAndUpdate));

void SpeculateSetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all,
                                    const paddle::Tensor& accept_tokens,
                                    const paddle::Tensor& accept_num,
                                    const paddle::Tensor& stop_flags,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& step_idx) {}

PD_BUILD_OP(speculate_set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all",
             "accept_tokens",
             "accept_num",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateSetValueByFlagsAndIdx));

std::vector<paddle::Tensor> SpeculateGetSeqLensOutput(
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder) {
  return {seq_lens_this_time};
}

std::vector<std::vector<int64_t>> SpeculateGetSeqLensOutputInferShape(
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape) {
  int64_t bsz = seq_lens_this_time_shape[0];
  return {{bsz}};
}

std::vector<paddle::DataType> SpeculateGetSeqLensOutputInferDtype(
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype) {
  return {seq_lens_this_time_dtype};
}

PD_BUILD_OP(speculate_get_seq_lens_output)
    .Inputs({"seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder"})
    .Outputs({"seq_lens_output"})
    .SetKernelFn(PD_KERNEL(SpeculateGetSeqLensOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetSeqLensOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetSeqLensOutputInferDtype));

std::vector<paddle::Tensor> SpeculateGetOutputPaddingOffset(
    const paddle::Tensor& output_cum_offsets_tmp,
    const paddle::Tensor& out_token_num,
    const paddle::Tensor& seq_lens_output,
    const int max_seq_len) {
  return {output_cum_offsets_tmp};
}

std::vector<std::vector<int64_t>> SpeculateGetOutputPaddingOffsetInferShape(
    const std::vector<int64_t>& output_cum_offsets_tmp_shape,
    const std::vector<int64_t>& out_token_num_shape,
    const std::vector<int64_t>& seq_lens_output_shape) {
  int64_t bsz = output_cum_offsets_tmp_shape[0];
  return {{-1}, {bsz}};
}

std::vector<paddle::DataType> SpeculateGetOutputPaddingOffsetInferDtype(
    const paddle::DataType& output_cum_offsets_tmp_dtype,
    const paddle::DataType& out_token_num_dtype,
    const paddle::DataType& seq_lens_output_dtype) {
  return {output_cum_offsets_tmp_dtype, output_cum_offsets_tmp_dtype};
}

PD_BUILD_OP(speculate_get_output_padding_offset)
    .Inputs({"output_cum_offsets_tmp", "out_token_num", "seq_lens_output"})
    .Outputs({"output_padding_offset", "output_cum_offsets"})
    .Attrs({"max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(SpeculateGetOutputPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(SpeculateGetOutputPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SpeculateGetOutputPaddingOffsetInferDtype));

void SaveOutMsg(const paddle::Tensor& x,
                const paddle::Tensor& not_need_stop,
                const paddle::optional<paddle::Tensor>& accept_num,
                int64_t rank_id) {}

PD_BUILD_OP(save_output)
    .Inputs({"x", "not_need_stop", paddle::Optional("accept_num")})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMsg));
