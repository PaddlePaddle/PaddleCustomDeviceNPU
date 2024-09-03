# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.incubate.passes import ir


@ir.RegisterPass
def remove_residual_in_rms_norm():
    def pattern(hidden_state, norm_weight, residual):
        rms_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight, x=hidden_state, residual=residual
        )
        return rms_norm.Output("out")

    def replace(hidden_state, norm_weight, residual):
        rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=norm_weight, x=hidden_state)
        rms_norm.Attr("epsilon").MappedPattern(op="rms_norm", name="epsilon", index=0)
        rms_norm.Attr("quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        rms_norm.Attr("quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        return rms_norm.Output("out")

    return pattern, replace


@ir.RegisterPass
def remove_residual_in_fused_bias_residual_layernorm():
    def pattern(residual, x):
        op = ir.PassDesc.OP.fused_bias_residual_layernorm
        op._outputs.pop("mean")
        op._outputs.pop("residual_out")
        op._outputs.pop("variance")
        result = op(residual=residual, x=x)
        return result.Output("out")[0]

    def replace(residual, x):
        op = ir.PassDesc.OP.fused_bias_residual_layernorm
        op._outputs.pop("mean")
        op._outputs.pop("residual_out")
        op._outputs.pop("variance")
        result = op(x=x)
        return result.Output("out")[0]

    return pattern, replace


@ir.RegisterPass
def remove_blha_get_max_len():
    def pattern(seq_lens_encoder, seq_lens_decoder, batch_size):
        blha_get_max_len = ir.PassDesc.OP.blha_get_max_len(
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            batch_size=batch_size,
        )
        return (
            blha_get_max_len.Output("max_enc_len_this_time")[0],
            blha_get_max_len.Output("max_dec_len_this_time")[0],
        )

    def replace(seq_lens_encoder, seq_lens_decoder, batch_size):
        return seq_lens_encoder, seq_lens_decoder

    return pattern, replace


@ir.RegisterPass
def llama_fuse_attention_layer():
    def pattern(
        hidden_state,
        norm_weight,
        qkv_weight,
        out_weight,
        ffn_norm_weight,
        ffn1_weight,
        ffn2_weight,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        input_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight,
            x=hidden_state,
        )
        qkv_proj = ir.PassDesc.OP.matmul_v2(X=input_norm.Output("out"), Y=qkv_weight)
        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv_proj.Output("Out"),
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            padding_offsets=padding_offsets,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            rope_emb=rope_emb,
        )
        out_proj = ir.PassDesc.OP.matmul_v2(X=blha.Output("fmha_out"), Y=out_weight)
        ffn_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=ffn_norm_weight, x=out_proj.Output("Out")
        )
        ffn1_proj = ir.PassDesc.OP.matmul_v2(X=ffn_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(
            x=ffn1_proj.Output("Out"),
        )
        ffn2_proj = ir.PassDesc.OP.matmul_v2(
            X=fused_bias_act.Output("out"), Y=ffn2_weight
        )
        return ffn2_proj

    def replace(
        hidden_state,
        norm_weight,
        qkv_weight,
        out_weight,
        ffn_norm_weight,
        ffn1_weight,
        ffn2_weight,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        fused_blha_op = ir.PassDesc.OP.fused_blha_layer_op
        fused_blha_op(
            hidden=hidden_state,
            norm_weight=norm_weight,
            qkv_weight=qkv_weight,
            out_linear_weight=out_weight,
            ffn_norm_weight=ffn_norm_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            rope_emb=rope_emb,
            cache_k=key_cache,
            cache_v=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            block_tables=block_tables,
        )
        fused_blha_op.SetAttr("flag", 0)
        fused_blha_op.SetAttr("use_neox_style", False)
        fused_blha_op.Attr("max_seq_len").MappedPattern(
            op="block_multihead_attention", name="max_seq_len", index=0
        )
        fused_blha_op.Attr("block_size").MappedPattern(
            op="block_multihead_attention", name="block_size", index=0
        )
        fused_blha_op.Attr("epsilon").MappedPattern(
            op="rms_norm", name="epsilon", index=0
        )
        fused_blha_op.Attr("qkv_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        fused_blha_op.Attr("out_linear_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=1
        )
        fused_blha_op.Attr("ffn1_quant_scale").MappedPattern(
            op="block_multihead_attention", name="out_scale", index=0
        )
        fused_blha_op.Attr("ffn2_quant_scale").MappedPattern(
            op="fused_bias_act", name="quant_scale", index=0
        )
        fused_blha_op.Attr("trans_qkv").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        fused_blha_op.Attr("trans_out_linear").MappedPattern(
            op="matmul_v2", name="trans_y", index=1
        )
        fused_blha_op.Attr("trans_ffn1").MappedPattern(
            op="matmul_v2", name="trans_y", index=2
        )
        fused_blha_op.Attr("trans_ffn2").MappedPattern(
            op="matmul_v2", name="trans_y", index=3
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def llama_fuse_attention_layer_begin():
    def pattern(
        input_ids,
        embedding_weight,
        norm_weight,
        qkv_weight,
        out_weight,
        ffn_norm_weight,
        ffn1_weight,
        ffn2_weight,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        embedding = ir.PassDesc.OP.lookup_table_v2(Ids=input_ids, W=embedding_weight)
        input_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight,
            x=embedding.Output("Out"),
        )
        qkv_proj = ir.PassDesc.OP.matmul_v2(X=input_norm.Output("out"), Y=qkv_weight)
        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv_proj.Output("Out"),
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            padding_offsets=padding_offsets,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            rope_emb=rope_emb,
        )
        out_proj = ir.PassDesc.OP.matmul_v2(X=blha.Output("fmha_out"), Y=out_weight)
        ffn_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=ffn_norm_weight, x=out_proj.Output("Out")
        )
        ffn1_proj = ir.PassDesc.OP.matmul_v2(X=ffn_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(
            x=ffn1_proj.Output("Out"),
        )
        ffn2_proj = ir.PassDesc.OP.matmul_v2(
            X=fused_bias_act.Output("out"), Y=ffn2_weight
        )
        return ffn2_proj

    def replace(
        input_ids,
        embedding_weight,
        norm_weight,
        qkv_weight,
        out_weight,
        ffn_norm_weight,
        ffn1_weight,
        ffn2_weight,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        embedding = ir.PassDesc.OP.lookup_table_v2(Ids=input_ids, W=embedding_weight)

        fused_blha_op = ir.PassDesc.OP.fused_blha_layer_op
        fused_blha_op(
            hidden=embedding.Output("Out"),
            norm_weight=norm_weight,
            qkv_weight=qkv_weight,
            out_linear_weight=out_weight,
            ffn_norm_weight=ffn_norm_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            rope_emb=rope_emb,
            cache_k=key_cache,
            cache_v=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            block_tables=block_tables,
        )
        fused_blha_op.SetAttr("flag", 1)
        fused_blha_op.SetAttr("use_neox_style", False)
        fused_blha_op.Attr("max_seq_len").MappedPattern(
            op="block_multihead_attention", name="max_seq_len", index=0
        )
        fused_blha_op.Attr("block_size").MappedPattern(
            op="block_multihead_attention", name="block_size", index=0
        )
        fused_blha_op.Attr("epsilon").MappedPattern(
            op="rms_norm", name="epsilon", index=0
        )
        fused_blha_op.Attr("qkv_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        fused_blha_op.Attr("out_linear_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=1
        )
        fused_blha_op.Attr("ffn1_quant_scale").MappedPattern(
            op="block_multihead_attention", name="out_scale", index=0
        )
        fused_blha_op.Attr("ffn2_quant_scale").MappedPattern(
            op="fused_bias_act", name="quant_scale", index=0
        )
        fused_blha_op.Attr("trans_qkv").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        fused_blha_op.Attr("trans_out_linear").MappedPattern(
            op="matmul_v2", name="trans_y", index=1
        )
        fused_blha_op.Attr("trans_ffn1").MappedPattern(
            op="matmul_v2", name="trans_y", index=2
        )
        fused_blha_op.Attr("trans_ffn2").MappedPattern(
            op="matmul_v2", name="trans_y", index=3
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def llama_fuse_attention_layer_end():
    def pattern(
        hidden_state,
        norm_weight,
        qkv_weight,
        out_weight,
        ffn_norm_weight,
        ffn1_weight,
        ffn2_weight,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        input_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight,
            x=hidden_state,
        )
        qkv_proj = ir.PassDesc.OP.matmul_v2(X=input_norm.Output("out"), Y=qkv_weight)
        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv_proj.Output("Out"),
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            padding_offsets=padding_offsets,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            rope_emb=rope_emb,
        )
        out_proj = ir.PassDesc.OP.matmul_v2(X=blha.Output("fmha_out"), Y=out_weight)
        ffn_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=ffn_norm_weight, x=out_proj.Output("Out")
        )
        ffn1_proj = ir.PassDesc.OP.matmul_v2(X=ffn_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(
            x=ffn1_proj.Output("Out"),
        )
        ffn2_proj = ir.PassDesc.OP.matmul_v2(
            X=fused_bias_act.Output("out"), Y=ffn2_weight
        )

        fused_bias_res_ln_op = ir.PassDesc.OP.fused_bias_residual_layernorm
        fused_bias_res_ln_op._outputs.pop("mean")
        fused_bias_res_ln_op._outputs.pop("residual_out")
        fused_bias_res_ln_op._outputs.pop("variance")
        fused_bias_res_ln_out = fused_bias_res_ln_op(x=ffn2_proj.Output("Out"))
        result = ir.PassDesc.OP.rebuild_padding_v2(
            cum_offsets=cum_offsets,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            tmp_out=fused_bias_res_ln_out.Output("out"),
        )
        return result.Output("out")

    def replace(
        hidden_state,
        norm_weight,
        qkv_weight,
        out_weight,
        ffn_norm_weight,
        ffn1_weight,
        ffn2_weight,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        fused_blha_op = ir.PassDesc.OP.fused_blha_layer_op
        fused_blha_op(
            hidden=hidden_state,
            norm_weight=norm_weight,
            qkv_weight=qkv_weight,
            out_linear_weight=out_weight,
            ffn_norm_weight=ffn_norm_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            rope_emb=rope_emb,
            cache_k=key_cache,
            cache_v=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            block_tables=block_tables,
        )
        fused_blha_op.SetAttr("flag", 2)
        fused_blha_op.SetAttr("use_neox_style", False)
        fused_blha_op.Attr("max_seq_len").MappedPattern(
            op="block_multihead_attention", name="max_seq_len", index=0
        )
        fused_blha_op.Attr("block_size").MappedPattern(
            op="block_multihead_attention", name="block_size", index=0
        )
        fused_blha_op.Attr("epsilon").MappedPattern(
            op="rms_norm", name="epsilon", index=0
        )
        fused_blha_op.Attr("qkv_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        fused_blha_op.Attr("out_linear_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=1
        )
        fused_blha_op.Attr("ffn1_quant_scale").MappedPattern(
            op="block_multihead_attention", name="out_scale", index=0
        )
        fused_blha_op.Attr("ffn2_quant_scale").MappedPattern(
            op="fused_bias_act", name="quant_scale", index=0
        )
        fused_blha_op.Attr("trans_qkv").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        fused_blha_op.Attr("trans_out_linear").MappedPattern(
            op="matmul_v2", name="trans_y", index=1
        )
        fused_blha_op.Attr("trans_ffn1").MappedPattern(
            op="matmul_v2", name="trans_y", index=2
        )
        fused_blha_op.Attr("trans_ffn2").MappedPattern(
            op="matmul_v2", name="trans_y", index=3
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def llama_fuse_lm_head_with_slice():
    def pattern(x, norm_weight, linear_weight):
        norm = ir.PassDesc.OP.rms_norm(norm_weight=norm_weight, x=x)
        norm_slice = ir.PassDesc.OP.slice(Input=norm.Output("out"))
        matmul = ir.PassDesc.OP.matmul_v2(X=norm.Output("out"), Y=linear_weight)
        return matmul.Output("Out")

    def replace(x, norm_weight, linear_weight):
        lm_head = ir.PassDesc.OP.lm_head(
            x=x, norm_weight=norm_weight, linear_weight=linear_weight
        )
        lm_head.Attr("epsilon").MappedPattern(op="rms_norm", name="epsilon", index=0)
        lm_head.Attr("trans_weight").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        lm_head.SetAttr("rank", 0)
        lm_head.SetAttr("nranks", 1)
        lm_head.SetAttr("root", 0)
        lm_head.SetAttr("ring_id", 0)
        return lm_head

    return pattern, replace


@ir.RegisterPass
def llama_fuse_lm_head():
    def pattern(x, norm_weight, linear_weight):
        norm = ir.PassDesc.OP.rms_norm(norm_weight=norm_weight, x=x)
        matmul = ir.PassDesc.OP.matmul_v2(X=norm.Output("out"), Y=linear_weight)
        return matmul.Output("Out")

    def replace(x, norm_weight, linear_weight):
        lm_head = ir.PassDesc.OP.lm_head(
            x=x, norm_weight=norm_weight, linear_weight=linear_weight
        )
        lm_head.Attr("epsilon").MappedPattern(op="rms_norm", name="epsilon", index=0)
        lm_head.Attr("trans_weight").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        lm_head.SetAttr("rank", 0)
        lm_head.SetAttr("nranks", 1)
        lm_head.SetAttr("root", 0)
        lm_head.SetAttr("ring_id", 0)
        return lm_head

    return pattern, replace


@ir.RegisterPass
def llama_fuse_get_padding_offset():
    def pattern(input_ids, seq_lens_this_time):
        scale = ir.PassDesc.OP.scale(X=seq_lens_this_time)
        cumsum = ir.PassDesc.OP.cumsum(X=scale)
        reduce_sum = ir.PassDesc.OP.reduce_sum(X=seq_lens_this_time)
        results = ir.PassDesc.OP.get_padding_offset_v2(
            cum_offsets=cumsum,
            input_ids=input_ids,
            seq_len=seq_lens_this_time,
            token_num=reduce_sum,
        )
        return results.Output("x_remove_padding")

    def replace(input_ids, seq_lens_this_time):
        results = ir.PassDesc.OP.remove_padding(x=input_ids, seqlen=seq_lens_this_time)
        return results

    return pattern, replace


@ir.RegisterPass
def replace_top_p_sampling_by_top_k_v2():
    def pattern(probs, top_p):
        results = ir.PassDesc.OP.top_p_sampling(
            x=probs,
            ps=top_p,
        )
        return results.Output("ids")

    def replace(probs, top_p):
        top_k_v2_op = ir.PassDesc.OP.top_k_v2
        top_k_v2_op.SetAttr("axis", -1)
        top_k_v2_op.SetAttr("k", 1)
        top_k_v2_op.SetAttr("largest", True)
        top_k_v2_op(X=probs)
        return top_k_v2_op.Output("Indices")

    return pattern, replace


@ir.RegisterPass
def llama_fuse_attention_smooth_quant_layer_begin():
    def pattern(
        input_ids,
        embedding_weight,
        norm_weight,
        qkv_weight,
        qkv_out_scale,
        out_weight,
        out_shift,
        out_smooth,
        out_linear_out_scale,
        ffn_norm_weight,
        ffn1_weight,
        ffn1_out_scale,
        ffn2_weight,
        ffn2_shift,
        ffn2_smooth,
        ffn2_out_scale,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        embedding = ir.PassDesc.OP.lookup_table_v2(Ids=input_ids, W=embedding_weight)

        input_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight,
            x=embedding.Output("Out"),
        )

        qkv_proj = ir.PassDesc.OP.matmul_v2(X=input_norm.Output("out"), Y=qkv_weight)

        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv_proj.Output("Out"),
            qkv_out_scale=qkv_out_scale,
            out_shift=out_shift,
            out_smooth=out_smooth,
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            padding_offsets=padding_offsets,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            rope_emb=rope_emb,
        )

        out_proj = ir.PassDesc.OP.matmul_v2(X=blha.Output("fmha_out"), Y=out_weight)

        dequant_out_proj = ir.PassDesc.OP.dequant_int8(
            intput=out_proj, out_scale=out_linear_out_scale
        )

        ffn_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=ffn_norm_weight, x=dequant_out_proj
        )

        ffn1_proj = ir.PassDesc.OP.matmul_v2(X=ffn_norm.Output("out"), Y=ffn1_weight)

        fused_bias_act = ir.PassDesc.OP.fused_bias_act(
            x=ffn1_proj.Output("Out"),
            dequant_scales=ffn1_out_scale,
            shift=ffn2_shift,
            smooth=ffn2_smooth,
        )

        ffn2_proj = ir.PassDesc.OP.matmul_v2(
            X=fused_bias_act.Output("out"), Y=ffn2_weight
        )

        dequant_ffn2_proj = ir.PassDesc.OP.dequant_int8(
            intput=ffn2_proj, out_scale=ffn2_out_scale
        )

        return dequant_ffn2_proj

    def replace(
        input_ids,
        embedding_weight,
        norm_weight,
        qkv_weight,
        qkv_out_scale,
        out_weight,
        out_shift,
        out_smooth,
        out_linear_out_scale,
        ffn_norm_weight,
        ffn1_weight,
        ffn1_out_scale,
        ffn2_weight,
        ffn2_shift,
        ffn2_smooth,
        ffn2_out_scale,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        embedding = ir.PassDesc.OP.lookup_table_v2(Ids=input_ids, W=embedding_weight)

        fused_blha_op = ir.PassDesc.OP.fused_blha_layer_op
        fused_blha_op(
            hidden=embedding.Output("Out"),
            norm_weight=norm_weight,
            qkv_weight=qkv_weight,
            out_linear_weight=out_weight,
            ffn_norm_weight=ffn_norm_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            rope_emb=rope_emb,
            cache_k=key_cache,
            cache_v=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            block_tables=block_tables,
        )
        fused_blha_op._desc.set_input("qkv_deq_scale@OPTIONAL", [qkv_out_scale.name])
        fused_blha_op._desc.set_input("out_linear_shift@OPTIONAL", [out_shift.name])
        fused_blha_op._desc.set_input("out_linear_smooth@OPTIONAL", [out_smooth.name])
        fused_blha_op._desc.set_input(
            "out_linear_deq_scale@OPTIONAL", [out_linear_out_scale.name]
        )
        fused_blha_op._desc.set_input("ffn1_deq_scale@OPTIONAL", [ffn1_out_scale.name])
        fused_blha_op._desc.set_input("ffn2_shift@OPTIONAL", [ffn2_shift.name])
        fused_blha_op._desc.set_input("ffn2_smooth@OPTIONAL", [ffn2_smooth.name])
        fused_blha_op._desc.set_input("ffn2_deq_scale@OPTIONAL", [ffn2_out_scale.name])
        fused_blha_op.SetAttr("flag", 1)
        fused_blha_op.SetAttr("use_neox_style", False)
        fused_blha_op.Attr("max_seq_len").MappedPattern(
            op="block_multihead_attention", name="max_seq_len", index=0
        )
        fused_blha_op.Attr("block_size").MappedPattern(
            op="block_multihead_attention", name="block_size", index=0
        )
        fused_blha_op.Attr("epsilon").MappedPattern(
            op="rms_norm", name="epsilon", index=0
        )
        fused_blha_op.Attr("qkv_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        fused_blha_op.Attr("out_linear_quant_scale").MappedPattern(
            op="block_multihead_attention", name="out_scale", index=0
        )
        fused_blha_op.Attr("ffn1_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=1
        )
        fused_blha_op.Attr("ffn2_quant_scale").MappedPattern(
            op="fused_bias_act", name="quant_scale", index=0
        )
        fused_blha_op.Attr("trans_qkv").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        fused_blha_op.Attr("trans_out_linear").MappedPattern(
            op="matmul_v2", name="trans_y", index=1
        )
        fused_blha_op.Attr("trans_ffn1").MappedPattern(
            op="matmul_v2", name="trans_y", index=2
        )
        fused_blha_op.Attr("trans_ffn2").MappedPattern(
            op="matmul_v2", name="trans_y", index=3
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def llama_fuse_attention_smooth_quant_layer_end():
    def pattern(
        hidden_state,
        norm_weight,
        qkv_weight,
        qkv_out_scale,
        out_weight,
        out_shift,
        out_smooth,
        out_linear_out_scale,
        ffn_norm_weight,
        ffn1_weight,
        ffn1_out_scale,
        ffn2_weight,
        ffn2_shift,
        ffn2_smooth,
        ffn2_out_scale,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        input_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight,
            x=hidden_state,
        )

        qkv_proj = ir.PassDesc.OP.matmul_v2(X=input_norm.Output("out"), Y=qkv_weight)

        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv_proj.Output("Out"),
            qkv_out_scale=qkv_out_scale,
            out_shift=out_shift,
            out_smooth=out_smooth,
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            padding_offsets=padding_offsets,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            rope_emb=rope_emb,
        )

        out_proj = ir.PassDesc.OP.matmul_v2(X=blha.Output("fmha_out"), Y=out_weight)

        dequant_out_proj = ir.PassDesc.OP.dequant_int8(
            intput=out_proj, out_scale=out_linear_out_scale
        )

        ffn_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=ffn_norm_weight, x=dequant_out_proj
        )

        ffn1_proj = ir.PassDesc.OP.matmul_v2(X=ffn_norm.Output("out"), Y=ffn1_weight)

        fused_bias_act = ir.PassDesc.OP.fused_bias_act(
            x=ffn1_proj.Output("Out"),
            dequant_scales=ffn1_out_scale,
            shift=ffn2_shift,
            smooth=ffn2_smooth,
        )

        ffn2_proj = ir.PassDesc.OP.matmul_v2(
            X=fused_bias_act.Output("out"), Y=ffn2_weight
        )

        dequant_ffn2_proj = ir.PassDesc.OP.dequant_int8(
            intput=ffn2_proj, out_scale=ffn2_out_scale
        )

        fused_bias_res_ln_op = ir.PassDesc.OP.fused_bias_residual_layernorm
        fused_bias_res_ln_op._outputs.pop("mean")
        fused_bias_res_ln_op._outputs.pop("residual_out")
        fused_bias_res_ln_op._outputs.pop("variance")
        fused_bias_res_ln_out = fused_bias_res_ln_op(x=dequant_ffn2_proj)
        result = ir.PassDesc.OP.rebuild_padding_v2(
            cum_offsets=cum_offsets,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            tmp_out=fused_bias_res_ln_out.Output("out"),
        )
        return result.Output("out")

    def replace(
        hidden_state,
        norm_weight,
        qkv_weight,
        qkv_out_scale,
        out_weight,
        out_shift,
        out_smooth,
        out_linear_out_scale,
        ffn_norm_weight,
        ffn1_weight,
        ffn1_out_scale,
        ffn2_weight,
        ffn2_shift,
        ffn2_smooth,
        ffn2_out_scale,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        fused_blha_op = ir.PassDesc.OP.fused_blha_layer_op
        fused_blha_op(
            hidden=hidden_state,
            norm_weight=norm_weight,
            qkv_weight=qkv_weight,
            out_linear_weight=out_weight,
            ffn_norm_weight=ffn_norm_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            rope_emb=rope_emb,
            cache_k=key_cache,
            cache_v=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            block_tables=block_tables,
        )
        fused_blha_op._desc.set_input("qkv_deq_scale@OPTIONAL", [qkv_out_scale.name])
        fused_blha_op._desc.set_input("out_linear_shift@OPTIONAL", [out_shift.name])
        fused_blha_op._desc.set_input("out_linear_smooth@OPTIONAL", [out_smooth.name])
        fused_blha_op._desc.set_input(
            "out_linear_deq_scale@OPTIONAL", [out_linear_out_scale.name]
        )
        fused_blha_op._desc.set_input("ffn1_deq_scale@OPTIONAL", [ffn1_out_scale.name])
        fused_blha_op._desc.set_input("ffn2_shift@OPTIONAL", [ffn2_shift.name])
        fused_blha_op._desc.set_input("ffn2_smooth@OPTIONAL", [ffn2_smooth.name])
        fused_blha_op._desc.set_input("ffn2_deq_scale@OPTIONAL", [ffn2_out_scale.name])
        fused_blha_op.SetAttr("flag", 2)
        fused_blha_op.SetAttr("use_neox_style", False)
        fused_blha_op.Attr("max_seq_len").MappedPattern(
            op="block_multihead_attention", name="max_seq_len", index=0
        )
        fused_blha_op.Attr("block_size").MappedPattern(
            op="block_multihead_attention", name="block_size", index=0
        )
        fused_blha_op.Attr("epsilon").MappedPattern(
            op="rms_norm", name="epsilon", index=0
        )
        fused_blha_op.Attr("qkv_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        fused_blha_op.Attr("out_linear_quant_scale").MappedPattern(
            op="block_multihead_attention", name="out_scale", index=0
        )
        fused_blha_op.Attr("ffn1_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=1
        )
        fused_blha_op.Attr("ffn2_quant_scale").MappedPattern(
            op="fused_bias_act", name="quant_scale", index=0
        )
        fused_blha_op.Attr("trans_qkv").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        fused_blha_op.Attr("trans_out_linear").MappedPattern(
            op="matmul_v2", name="trans_y", index=1
        )
        fused_blha_op.Attr("trans_ffn1").MappedPattern(
            op="matmul_v2", name="trans_y", index=2
        )
        fused_blha_op.Attr("trans_ffn2").MappedPattern(
            op="matmul_v2", name="trans_y", index=3
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def llama_fuse_attention_smooth_quant_layer():
    def pattern(
        hidden_state,
        norm_weight,
        qkv_weight,
        qkv_out_scale,
        out_weight,
        out_shift,
        out_smooth,
        out_linear_out_scale,
        ffn_norm_weight,
        ffn1_weight,
        ffn1_out_scale,
        ffn2_weight,
        ffn2_shift,
        ffn2_smooth,
        ffn2_out_scale,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        input_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=norm_weight,
            x=hidden_state,
        )

        qkv_proj = ir.PassDesc.OP.matmul_v2(X=input_norm.Output("out"), Y=qkv_weight)

        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv_proj.Output("Out"),
            qkv_out_scale=qkv_out_scale,
            out_shift=out_shift,
            out_smooth=out_smooth,
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            padding_offsets=padding_offsets,
            cum_offsets=cum_offsets,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            rope_emb=rope_emb,
        )

        out_proj = ir.PassDesc.OP.matmul_v2(X=blha.Output("fmha_out"), Y=out_weight)

        dequant_out_proj = ir.PassDesc.OP.dequant_int8(
            intput=out_proj, out_scale=out_linear_out_scale
        )

        ffn_norm = ir.PassDesc.OP.rms_norm(
            norm_weight=ffn_norm_weight, x=dequant_out_proj
        )

        ffn1_proj = ir.PassDesc.OP.matmul_v2(X=ffn_norm.Output("out"), Y=ffn1_weight)

        fused_bias_act = ir.PassDesc.OP.fused_bias_act(
            x=ffn1_proj.Output("Out"),
            dequant_scales=ffn1_out_scale,
            shift=ffn2_shift,
            smooth=ffn2_smooth,
        )

        ffn2_proj = ir.PassDesc.OP.matmul_v2(
            X=fused_bias_act.Output("out"), Y=ffn2_weight
        )

        dequant_ffn2_proj = ir.PassDesc.OP.dequant_int8(
            intput=ffn2_proj, out_scale=ffn2_out_scale
        )

        return dequant_ffn2_proj

    def replace(
        hidden_state,
        norm_weight,
        qkv_weight,
        qkv_out_scale,
        out_weight,
        out_shift,
        out_smooth,
        out_linear_out_scale,
        ffn_norm_weight,
        ffn1_weight,
        ffn1_out_scale,
        ffn2_weight,
        ffn2_shift,
        ffn2_smooth,
        ffn2_out_scale,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        padding_offsets,
        cum_offsets,
        cu_seqlens_q,
        cu_seqlens_k,
        block_tables,
        rope_emb,
    ):
        fused_blha_op = ir.PassDesc.OP.fused_blha_layer_op
        fused_blha_op(
            hidden=hidden_state,
            norm_weight=norm_weight,
            qkv_weight=qkv_weight,
            out_linear_weight=out_weight,
            ffn_norm_weight=ffn_norm_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            rope_emb=rope_emb,
            cache_k=key_cache,
            cache_v=value_cache,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            block_tables=block_tables,
        )
        fused_blha_op._desc.set_input("qkv_deq_scale@OPTIONAL", [qkv_out_scale.name])
        fused_blha_op._desc.set_input("out_linear_shift@OPTIONAL", [out_shift.name])
        fused_blha_op._desc.set_input("out_linear_smooth@OPTIONAL", [out_smooth.name])
        fused_blha_op._desc.set_input(
            "out_linear_deq_scale@OPTIONAL", [out_linear_out_scale.name]
        )
        fused_blha_op._desc.set_input("ffn1_deq_scale@OPTIONAL", [ffn1_out_scale.name])
        fused_blha_op._desc.set_input("ffn2_shift@OPTIONAL", [ffn2_shift.name])
        fused_blha_op._desc.set_input("ffn2_smooth@OPTIONAL", [ffn2_smooth.name])
        fused_blha_op._desc.set_input("ffn2_deq_scale@OPTIONAL", [ffn2_out_scale.name])
        fused_blha_op.SetAttr("flag", 0)
        fused_blha_op.SetAttr("use_neox_style", False)
        fused_blha_op.Attr("max_seq_len").MappedPattern(
            op="block_multihead_attention", name="max_seq_len", index=0
        )
        fused_blha_op.Attr("block_size").MappedPattern(
            op="block_multihead_attention", name="block_size", index=0
        )
        fused_blha_op.Attr("epsilon").MappedPattern(
            op="rms_norm", name="epsilon", index=0
        )
        fused_blha_op.Attr("qkv_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=0
        )
        fused_blha_op.Attr("out_linear_quant_scale").MappedPattern(
            op="block_multihead_attention", name="out_scale", index=0
        )
        fused_blha_op.Attr("ffn1_quant_scale").MappedPattern(
            op="rms_norm", name="quant_scale", index=1
        )
        fused_blha_op.Attr("ffn2_quant_scale").MappedPattern(
            op="fused_bias_act", name="quant_scale", index=0
        )
        fused_blha_op.Attr("trans_qkv").MappedPattern(
            op="matmul_v2", name="trans_y", index=0
        )
        fused_blha_op.Attr("trans_out_linear").MappedPattern(
            op="matmul_v2", name="trans_y", index=1
        )
        fused_blha_op.Attr("trans_ffn1").MappedPattern(
            op="matmul_v2", name="trans_y", index=2
        )
        fused_blha_op.Attr("trans_ffn2").MappedPattern(
            op="matmul_v2", name="trans_y", index=3
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace
