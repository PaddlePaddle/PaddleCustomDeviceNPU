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
def chatglm_fuse_attention_layer():
    def pattern(
        hidden_state,
        norm_weight,
        qkv_weight,
        qkv_bias,
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
        qkv = ir.PassDesc.OP.elementwise_add(X=qkv_proj.Output("Out"), Y=qkv_bias)
        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv.Output("Out"),
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
        qkv_bias,
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
        fused_blha_op._desc.set_input("qkv_bias@OPTIONAL", [qkv_bias.name])
        fused_blha_op.SetAttr("flag", 0)
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
        fused_blha_op.Attr("use_neox_style").MappedPattern(
            op="block_multihead_attention", name="use_neox_style", index=0
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def chatglm_fuse_attention_layer_begin():
    def pattern(
        input_ids,
        embedding_weight,
        norm_weight,
        qkv_weight,
        qkv_bias,
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
        qkv = ir.PassDesc.OP.elementwise_add(X=qkv_proj.Output("Out"), Y=qkv_bias)
        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv.Output("Out"),
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
        qkv_bias,
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
        fused_blha_op._desc.set_input("qkv_bias@OPTIONAL", [qkv_bias.name])
        fused_blha_op.SetAttr("flag", 1)
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
        fused_blha_op.Attr("use_neox_style").MappedPattern(
            op="block_multihead_attention", name="use_neox_style", index=0
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace


@ir.RegisterPass
def chatglm_fuse_attention_layer_end():
    def pattern(
        hidden_state,
        norm_weight,
        qkv_weight,
        qkv_bias,
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
        qkv = ir.PassDesc.OP.elementwise_add(X=qkv_proj.Output("Out"), Y=qkv_bias)
        blha = ir.PassDesc.OP.block_multihead_attention(
            qkv=qkv.Output("Out"),
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
        qkv_bias,
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
        fused_blha_op._desc.set_input("qkv_bias@OPTIONAL", [qkv_bias.name])
        fused_blha_op.SetAttr("flag", 2)
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
        fused_blha_op.Attr("use_neox_style").MappedPattern(
            op="block_multihead_attention", name="use_neox_style", index=0
        )
        fused_blha_op.SetAttr("use_alibi", False)
        return fused_blha_op.Output("hidden_out")

    return pattern, replace
