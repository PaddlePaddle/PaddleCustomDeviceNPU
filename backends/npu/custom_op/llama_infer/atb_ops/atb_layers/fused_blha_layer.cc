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

#include "fused_blha_layer.h"  // NOLINT

#include <cmath>

#include "linear.h"             // NOLINT
#include "mixed_gate_up_act.h"  // NOLINT
#include "qkv_split.h"          // NOLINT

namespace atb_layers {
void CreateFusedBlhaLayer(const FusedBlhaLayerParam &param,
                          atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT_HIDDEN_STATES = TENSOR_ID++;

  uint64_t INPUT_NORM_WEIGHT = TENSOR_ID++;

  uint64_t INPUT_QKV_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_QKV_BIAS = param.has_qkv_bias ? TENSOR_ID++ : 0;
  uint64_t INPUT_QKV_DEQSCALE = param.use_matmul_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_QKV_DEQOFFSET = param.use_matmul_int8 ? TENSOR_ID++ : 0;

  uint64_t INPUT_OUT_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_OUT_SMOOTH =
      param.use_matmul_int8 && param.use_smooth_quant ? TENSOR_ID++ : 0;
  uint64_t INPUT_OUT_SHIFT =
      param.use_matmul_int8 && param.use_smooth_quant ? TENSOR_ID++ : 0;
  uint64_t INPUT_OUT_DEQSCALE = param.use_matmul_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_OUT_DEQOFFSET = param.use_matmul_int8 ? TENSOR_ID++ : 0;

  uint64_t INPUT_FFN_NORM_WEIGHT = TENSOR_ID++;

  uint64_t INPUT_FFN1_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_FFN1_DEQSCALE = param.use_matmul_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_FFN1_DEQOFFSET = param.use_matmul_int8 ? TENSOR_ID++ : 0;

  uint64_t INPUT_FFN2_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_FFN2_SMOOTH =
      param.use_matmul_int8 && param.use_smooth_quant ? TENSOR_ID++ : 0;
  uint64_t INPUT_FFN2_SHIFT =
      param.use_matmul_int8 && param.use_smooth_quant ? TENSOR_ID++ : 0;
  uint64_t INPUT_FFN2_DEQSCALE = param.use_matmul_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_FFN2_DEQOFFSET = param.use_matmul_int8 ? TENSOR_ID++ : 0;

  uint64_t INPUT_COS = param.use_alibi ? 0 : TENSOR_ID++;
  uint64_t INPUT_SIN = param.use_alibi ? 0 : TENSOR_ID++;
  uint64_t INPUT_MASK = param.is_prefill || param.use_alibi ? TENSOR_ID++ : 0;

  uint64_t INPUT_CACHE_K = TENSOR_ID++;
  uint64_t INPUT_CACHE_V = TENSOR_ID++;

  uint64_t INPUT_CACHE_K_QUANT_SCALE = param.cache_kv_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_CACHE_K_QUANT_OFFSET = param.cache_kv_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_CACHE_V_QUANT_SCALE = param.cache_kv_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_CACHE_V_QUANT_OFFSET = param.cache_kv_int8 ? TENSOR_ID++ : 0;
  uint64_t INPUT_CACHE_K_DEQUANT_SCALE =
      (param.cache_kv_int8 && !param.is_prefill) ? TENSOR_ID++ : 0;
  uint64_t INPUT_CACHE_V_DEQUANT_SCALE =
      (param.cache_kv_int8 && !param.is_prefill) ? TENSOR_ID++ : 0;
  uint64_t INPUT_SLOTS = TENSOR_ID++;
  uint64_t INPUT_BLOCK_TABLES = !param.is_prefill ? TENSOR_ID++ : 0;
  uint64_t INPUT_SEQLEN = TENSOR_ID++;
  uint64_t INPUT_BATCH_STATUS = !param.is_prefill ? TENSOR_ID++ : INPUT_SEQLEN;

  uint64_t OUTPUT = TENSOR_ID++;

  uint64_t INTERMEDIATE_NORM_OUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_QKV_OUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_Q = TENSOR_ID++;
  uint64_t INTERMEDIATE_K = TENSOR_ID++;
  uint64_t INTERMEDIATE_V = TENSOR_ID++;
  uint64_t INTERMEDIATE_EMB_Q = param.use_alibi ? INTERMEDIATE_Q : TENSOR_ID++;
  uint64_t INTERMEDIATE_EMB_K = param.use_alibi ? INTERMEDIATE_K : TENSOR_ID++;
  uint64_t INTERMEDIATE_ATTN_OUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_LINEAR_OUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_LINEAR_ADD_RES = TENSOR_ID++;
  uint64_t INTERMEDIATE_FFN_NORM_OUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_FFN1_OUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_FFN2_OUT = TENSOR_ID++;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "FusedBlhaLayerOp";
  opGraph.inTensorNum = INPUT_BATCH_STATUS - INPUT_HIDDEN_STATES + 1;
  opGraph.outTensorNum = 1;
  opGraph.internalTensorNum = INTERMEDIATE_FFN2_OUT - INTERMEDIATE_NORM_OUT + 1;
  if (param.use_alibi) {
    opGraph.nodes.resize(11);
  } else {
    opGraph.nodes.resize(12);
  }

  // rms_norm
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::RmsNormParam opParam;
    opParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    opParam.normParam.epsilon = param.epsilon;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INPUT_HIDDEN_STATES, INPUT_NORM_WEIGHT};
    opNode.outTensorIds = {INTERMEDIATE_NORM_OUT};
  }

  // qkv
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::LinearParam opParam;
    opParam.trans_weight = param.trans_qkv_weight;
    opParam.has_bias = param.has_qkv_bias;
    opParam.input_quant = param.use_matmul_int8;
    opParam.input_quant_scale = param.qkv_quant_scale;
    opParam.input_quant_offset = 0;
    opParam.input_smooth_quant = false;
    opParam.has_dequant_offset = param.use_matmul_int8;
    atb::CreateOperation(opParam, &opNode.operation);
    if (param.has_qkv_bias && param.use_matmul_int8) {
      opNode.inTensorIds = {INTERMEDIATE_NORM_OUT,
                            INPUT_QKV_WEIGHT,
                            INPUT_QKV_BIAS,
                            INPUT_QKV_DEQSCALE,
                            INPUT_QKV_DEQOFFSET};
    } else if (param.has_qkv_bias) {
      opNode.inTensorIds = {
          INTERMEDIATE_NORM_OUT, INPUT_QKV_WEIGHT, INPUT_QKV_BIAS};
    } else if (param.use_matmul_int8) {
      opNode.inTensorIds = {INTERMEDIATE_NORM_OUT,
                            INPUT_QKV_WEIGHT,
                            INPUT_QKV_DEQSCALE,
                            INPUT_QKV_DEQOFFSET};
    } else {
      opNode.inTensorIds = {INTERMEDIATE_NORM_OUT, INPUT_QKV_WEIGHT};
    }
    opNode.outTensorIds = {INTERMEDIATE_QKV_OUT};
  }

  // split q,k,v
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::QKVSplitParam opParam;
    opParam.head_num = param.head_num;
    opParam.kv_head_num = param.kv_head_num;
    opParam.head_dim = param.head_dim;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_QKV_OUT};
    opNode.outTensorIds = {INTERMEDIATE_Q, INTERMEDIATE_K, INTERMEDIATE_V};
  }

  // rope
  if (!param.use_alibi) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::RopeParam opParam;
    opParam.rotaryCoeff = param.rope_neox ? param.head_dim : 2;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {
        INTERMEDIATE_Q, INTERMEDIATE_K, INPUT_COS, INPUT_SIN, INPUT_SEQLEN};
    opNode.outTensorIds = {INTERMEDIATE_EMB_Q, INTERMEDIATE_EMB_K};
  }

  // write kv
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ReshapeAndCacheParam opParam;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_EMB_K,
                          INTERMEDIATE_V,
                          INPUT_CACHE_K,
                          INPUT_CACHE_V,
                          INPUT_SLOTS};
    opNode.outTensorIds = {INPUT_CACHE_K, INPUT_CACHE_V};  // write in place
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
    opNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = param.kv_head_num;
      newShape.dims[2] = param.head_dim;
    };
    opNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = param.kv_head_num;
      newShape.dims[2] = param.head_dim;
    };
    opNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 4;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[2];
      newShape.dims[2] = oldShape.dims[1];
      newShape.dims[3] = oldShape.dims[3];
    };
    opNode.inTensorReshapeFuncs[3] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 4;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[2];
      newShape.dims[2] = oldShape.dims[1];
      newShape.dims[3] = oldShape.dims[3];
    };
  }

  if (param.is_prefill) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::SelfAttentionParam opParam;
    opParam.headNum = param.head_num;
    opParam.kvHeadNum = param.kv_head_num;
    opParam.qkScale = 1.0f / sqrt(param.head_dim);
    opParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    opParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    if (param.use_alibi) {
      opParam.isTriuMask = 0;
      opParam.maskType =
          atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
    } else {
      opParam.isTriuMask = 1;
    }
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_EMB_Q,
                          INTERMEDIATE_EMB_K,
                          INTERMEDIATE_V,
                          INPUT_MASK,
                          INPUT_SEQLEN};
    opNode.outTensorIds = {INTERMEDIATE_ATTN_OUT};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
  } else {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::PagedAttentionParam opParam;
    opParam.headNum = param.head_num;
    opParam.qkScale = 1.0f / sqrt(param.head_dim);
    opParam.kvHeadNum = param.kv_head_num;
    if (param.use_alibi) {
      opParam.maskType =
          atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    } else {
      opParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }
    opParam.batchRunStatusEnable = true;

    atb::CreateOperation(opParam, &opNode.operation);

    if (param.use_alibi) {
      opNode.inTensorIds = {INTERMEDIATE_EMB_Q,
                            INPUT_CACHE_K,
                            INPUT_CACHE_V,
                            INPUT_BLOCK_TABLES,
                            INPUT_SEQLEN,
                            INPUT_MASK,
                            INPUT_BATCH_STATUS};
    } else {
      opNode.inTensorIds = {INTERMEDIATE_EMB_Q,
                            INPUT_CACHE_K,
                            INPUT_CACHE_V,
                            INPUT_BLOCK_TABLES,
                            INPUT_SEQLEN,
                            INPUT_BATCH_STATUS};
    }

    opNode.outTensorIds = {INTERMEDIATE_ATTN_OUT};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
    opNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = param.kv_head_num;
      newShape.dims[2] = param.head_dim;
    };
    opNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 4;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[2];
      newShape.dims[2] = oldShape.dims[1];
      newShape.dims[3] = oldShape.dims[3];
    };
    opNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 4;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[2];
      newShape.dims[2] = oldShape.dims[1];
      newShape.dims[3] = oldShape.dims[3];
    };
  }

  // outlinear
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::LinearParam opParam;
    opParam.trans_weight = param.trans_out_weight;
    opParam.has_bias = false;
    opParam.input_quant = param.use_matmul_int8;
    opParam.input_quant_scale = param.out_quant_scale;
    opParam.input_quant_offset = 0;
    opParam.input_smooth_quant =
        param.use_matmul_int8 && param.use_smooth_quant;
    opParam.has_dequant_offset = param.use_matmul_int8;
    atb::CreateOperation(opParam, &opNode.operation);

    if (param.use_smooth_quant) {
      opNode.inTensorIds = {INTERMEDIATE_ATTN_OUT,
                            INPUT_OUT_WEIGHT,
                            INPUT_OUT_DEQSCALE,
                            INPUT_OUT_DEQOFFSET,
                            INPUT_OUT_SMOOTH,
                            INPUT_OUT_SHIFT};
    } else if (param.use_matmul_int8) {
      opNode.inTensorIds = {INTERMEDIATE_ATTN_OUT,
                            INPUT_OUT_WEIGHT,
                            INPUT_OUT_DEQSCALE,
                            INPUT_OUT_DEQOFFSET};
    } else {
      opNode.inTensorIds = {INTERMEDIATE_ATTN_OUT, INPUT_OUT_WEIGHT};
    }
    opNode.outTensorIds = {INTERMEDIATE_LINEAR_OUT};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
    opNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      if (oldShape.dimNum == 3) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.head_num * param.head_dim;
      } else {
        newShape = oldShape;
      }
    };
  }

  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ElewiseParam opParam;
    opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_LINEAR_OUT, INPUT_HIDDEN_STATES};
    opNode.outTensorIds = {INTERMEDIATE_LINEAR_ADD_RES};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
  }

  // ffn norm
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::RmsNormParam opParam;
    opParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    opParam.normParam.epsilon = param.epsilon;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_LINEAR_ADD_RES, INPUT_FFN_NORM_WEIGHT};
    opNode.outTensorIds = {INTERMEDIATE_FFN_NORM_OUT};
  }

  // gate_up
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::MixedGateUpActParam opParam;
    opParam.trans_weight = param.trans_ffn1_weight;
    opParam.act = param.ffn_act;
    opParam.scale = param.scale;
    opParam.linear_quant = param.use_matmul_int8;
    opParam.input_quant = param.use_matmul_int8;
    opParam.input_quant_scale = param.ffn1_quant_scale;
    opParam.input_quant_offset = 0;
    opParam.input_smooth_quant = false;
    opParam.has_dequant_offset = param.use_matmul_int8;
    atb::CreateOperation(opParam, &opNode.operation);
    if (param.use_matmul_int8) {
      opNode.inTensorIds = {INTERMEDIATE_FFN_NORM_OUT,
                            INPUT_FFN1_WEIGHT,
                            INPUT_FFN1_DEQSCALE,
                            INPUT_FFN1_DEQOFFSET};
    } else {
      opNode.inTensorIds = {INTERMEDIATE_FFN_NORM_OUT, INPUT_FFN1_WEIGHT};
    }
    opNode.outTensorIds = {INTERMEDIATE_FFN1_OUT};
  }

  // down
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::LinearParam opParam;
    opParam.trans_weight = param.trans_ffn2_weight;
    opParam.has_bias = false;
    opParam.input_quant = param.use_matmul_int8;
    opParam.input_quant_scale = param.ffn2_quant_scale;
    opParam.input_quant_offset = 0;
    opParam.input_smooth_quant =
        param.use_matmul_int8 && param.use_smooth_quant;
    opParam.has_dequant_offset = param.use_matmul_int8;
    atb::CreateOperation(opParam, &opNode.operation);

    if (param.use_smooth_quant) {
      opNode.inTensorIds = {INTERMEDIATE_FFN1_OUT,
                            INPUT_FFN2_WEIGHT,
                            INPUT_FFN2_DEQSCALE,
                            INPUT_FFN2_DEQOFFSET,
                            INPUT_FFN2_SMOOTH,
                            INPUT_FFN2_SHIFT};
    } else if (param.use_matmul_int8) {
      opNode.inTensorIds = {INTERMEDIATE_FFN1_OUT,
                            INPUT_FFN2_WEIGHT,
                            INPUT_FFN2_DEQSCALE,
                            INPUT_FFN2_DEQOFFSET};
    } else {
      opNode.inTensorIds = {INTERMEDIATE_FFN1_OUT, INPUT_FFN2_WEIGHT};
    }
    opNode.outTensorIds = {INTERMEDIATE_FFN2_OUT};
  }

  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ElewiseParam opParam;
    opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_FFN2_OUT, INTERMEDIATE_LINEAR_ADD_RES};
    opNode.outTensorIds = {OUTPUT};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
  }

  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
          atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.resize(1);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
      };
  atb::CreateOperation(opGraph, operation);
}
}  // namespace atb_layers

#endif
