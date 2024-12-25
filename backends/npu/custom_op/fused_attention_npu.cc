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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

enum class DropOutStatus { DROPOUT_NORMAL = 0, DROPOUT_NONE, DROPOUT_ALL };

DropOutStatus get_dropout_status(double keep_prob) {
  if (keep_prob == 0) {
    return DropOutStatus::DROPOUT_ALL;
  }
  if (keep_prob == 1.) {
    return DropOutStatus::DROPOUT_NONE;
  }
  return DropOutStatus::DROPOUT_NORMAL;
}

std::vector<std::vector<int64_t>> fusedattentionInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape) {
  return {q_shape, k_shape, v_shape};
}

phi::DenseTensor paddletensor2densortensor(const paddle::Tensor& paddletensor) {
  return *(static_cast<const phi::DenseTensor*>(paddletensor.impl().get()));
}

const phi::CustomContext* getcontext(const paddle::Tensor& tensor) {
  return static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(tensor.place()));
}

std::pair<int64_t, int64_t> get_pair(
    const paddle::optional<paddle::Tensor>& fixed_seed_offset,
    const phi::CustomContext& dev_ctx,
    bool is_test) {
  int64_t seed = 0;
  int64_t offset = 0;
  if (is_test) {
    return std::make_pair(seed, offset);
  }
  // npu上的tensor不支持索引，先复制到cpu上再取值，这个后续优化？！！！耗时操作
  if (fixed_seed_offset) {
    auto fixed_seed_offset_ptr = fixed_seed_offset.get_ptr();
    auto fixed_seed_offset_tensor =
        paddletensor2densortensor(*fixed_seed_offset_ptr);
    phi::DenseTensor cpu_tensor;
    phi::CPUPlace cpu_place;
    phi::Copy(dev_ctx, fixed_seed_offset_tensor, cpu_place, true, &cpu_tensor);
    const int64_t* fixed_seed_offset_data = (cpu_tensor).data<int64_t>();
    seed = static_cast<int64_t>(fixed_seed_offset_data[0]);
    offset = static_cast<int64_t>(fixed_seed_offset_data[1]);
  } else {
    seed = static_cast<uint64_t>((dev_ctx).GetGenerator()->Random64());
    offset = static_cast<uint64_t>((dev_ctx).GetGenerator()->Random64());
  }
  return std::make_pair(seed, offset);
}

int64_t get_single(const paddle::optional<paddle::Tensor>& fixed_seed_offset,
                   const phi::CustomContext& dev_ctx) {
  int64_t seed = 0;
  // npu上的tensor不支持索引，先复制到cpu上再取值，这个后续优化？！！！耗时操作
  if (fixed_seed_offset) {
    auto fixed_seed_offset_ptr = fixed_seed_offset.get_ptr();
    auto fixed_seed_offset_tensor =
        paddletensor2densortensor(*fixed_seed_offset_ptr);
    phi::DenseTensor cpu_tensor;
    phi::CPUPlace cpu_place;
    phi::Copy(dev_ctx, fixed_seed_offset_tensor, cpu_place, true, &cpu_tensor);
    const int64_t* fixed_seed_offset_data = (cpu_tensor).data<int64_t>();
    seed = static_cast<int64_t>(fixed_seed_offset_data[0]);
  }
  return seed;
}

// query,key,value：（batch,seq_len,head_num,head_dim）
std::vector<paddle::Tensor> npu_flash_attention(
    const paddle::Tensor& query,
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::optional<paddle::Tensor>& fixed_seed_offset,
    const paddle::optional<paddle::Tensor>& attn_mask,
    float dropout = 0.0,
    bool casual = false,
    bool return_softmax = false,
    bool is_test = false,
    bool is_triangle_upper_mask = true) {
  auto dev_ctx = getcontext(query);
  // q,k,v [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = paddletensor2densortensor(query);
  auto key_tensor = paddletensor2densortensor(key);
  auto value_tensor = paddletensor2densortensor(value);

  auto query_tensor_dims = phi::vectorize(query_tensor.dims());
  auto key_tensor_dims = phi::vectorize(key_tensor.dims());

  auto query_dtype = query_tensor.dtype();
  auto key_dtype = key_tensor.dtype();
  auto value_dtype = value_tensor.dtype();

  PADDLE_ENFORCE_EQ(query_tensor_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim [batch_size, "
                        "seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(query_tensor_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim [batch_size, "
                        "seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(key_tensor.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim [batch_size, "
                        "seq_len, num_heads, head_dim]"));
  PD_CHECK(dropout >= 0 && dropout <= 1,
           "The dropout value must be in range of [0, 1], but got ",
           dropout);

  PD_CHECK(query_dtype == phi::DataType::FLOAT16 ||
               query_dtype == phi::DataType::BFLOAT16,
           "The query tensor dtype must be bfloat16 or float16 , but got ",
           query_dtype);
  PD_CHECK(key_dtype == phi::DataType::FLOAT16 ||
               key_dtype == phi::DataType::BFLOAT16,
           "The key tensor dtype must be bfloat16 or float16 , but got ",
           key_dtype);
  PD_CHECK(value_dtype == phi::DataType::FLOAT16 ||
               value_dtype == phi::DataType::BFLOAT16,
           "The value tensor dtype must be bfloat16 or float16 , but got ",
           value_dtype);

  const int64_t head_num = query_tensor_dims[2];
  const double scale = 1.0f / std::sqrt(query_tensor_dims[3]);

  void* realShiftOptional = nullptr;
  void* padding_mask = nullptr;
  void* prefixOptional = nullptr;
  int64_t pre_tockens = 65536;
  int64_t next_tockens = 65536;
  int64_t inner_precise = 0;
  int64_t sparseModeOptional = 0;

  // seed offset
  std::pair<int64_t, int64_t> seed_offset =
      get_pair(fixed_seed_offset, *dev_ctx, is_test);
  int64_t seed = seed_offset.first;
  int64_t offset = seed_offset.second;
  // seed_tensor
  std::shared_ptr<phi::DenseTensor> seed_tensor =
      std::make_shared<phi::DenseTensor>();
  seed_tensor->Resize({1});
  (*dev_ctx).Alloc(seed_tensor.get(), phi::DataType::INT64);
  // offset_tensor
  std::shared_ptr<phi::DenseTensor> offset_tensor =
      std::make_shared<phi::DenseTensor>();
  offset_tensor->Resize({1});
  (*dev_ctx).Alloc(offset_tensor.get(), phi::DataType::INT64);

  if (is_test) {
    VLOG(3) << "You are in the inference phase and do not need to drop it.";
  }

  phi::DenseTensor* attn_mask_tensor = nullptr;
  std::shared_ptr<phi::DenseTensor> attn_mask_tensor_null =
      std::make_shared<phi::DenseTensor>();
  // casual ，用户mask无效，用上三角矩阵
  if (casual) {
    VLOG(3) << "Forward flash attention with casual mask.attn_mask is invalid, "
               "and use upper triangular matrix mask";
    int64_t diagonal = 0;
    attn_mask_tensor_null->Resize({query_tensor_dims[1], query_tensor_dims[1]});
    (*dev_ctx).Alloc(attn_mask_tensor_null.get(), phi::DataType::BOOL);
    EXEC_NPU_CMD(aclnnInplaceOne, *dev_ctx, *attn_mask_tensor_null);
    EXEC_NPU_CMD(aclnnTril,
                 *dev_ctx,
                 *attn_mask_tensor_null,
                 diagonal,
                 *attn_mask_tensor_null);
    EXEC_NPU_CMD(aclnnBitwiseNot,
                 *dev_ctx,
                 *attn_mask_tensor_null,
                 *attn_mask_tensor_null);
    attn_mask_tensor = attn_mask_tensor_null.get();
    next_tockens = 0;
  } else if (attn_mask) {
    VLOG(3) << "Forward flash attention with user defined mask";
    // 用户指定mask
    auto attn_mask_ptr = *(attn_mask.get_ptr());
    attn_mask_tensor =
        static_cast<phi::DenseTensor*>(attn_mask_ptr.impl().get());
    auto mask_dtype = attn_mask_tensor->dtype();

    PD_CHECK(mask_dtype == phi::DataType::BOOL,
             "The mask tensor dtype must be bool , but got ",
             mask_dtype);
    // 标准上三角矩阵mask，next_tockens为0可以提升性能
    if (is_triangle_upper_mask) {
      next_tockens = 0;
    }
  } else {
    VLOG(3) << "Forward flash attention without mask";
    // 无mask
    attn_mask_tensor = attn_mask_tensor_null.get();
  }

  int64_t numels = 0;
  double keep_prob = 1.0;
  // dropmask
  std::shared_ptr<phi::DenseTensor> dropmask =
      std::make_shared<phi::DenseTensor>();
  numels = query_tensor_dims[0] * query_tensor_dims[2] * query_tensor_dims[1] *
           query_tensor_dims[1];  // [B,N,S,S]
  if (!is_test) {
    if (get_dropout_status(keep_prob) != DropOutStatus::DROPOUT_NONE) {
      keep_prob = 1 - dropout;
      // (B,N,S,S)
      std::vector<int64_t> length_shape = {numels};
      dropmask->Resize({numels / 8});
      (*dev_ctx).Alloc(dropmask.get(), phi::DataType::UINT8);
      if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        custom_kernel::FillNpuTensorWithConstant(
            seed_tensor.get(), *dev_ctx, seed);
        custom_kernel::FillNpuTensorWithConstant(
            offset_tensor.get(), *dev_ctx, offset);
        EXEC_NPU_CMD(aclnnDropoutGenMask,
                     *dev_ctx,
                     length_shape,
                     dropout,
                     seed,
                     offset,
                     *dropmask);
      } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        EXEC_NPU_CMD(aclnnInplaceZero, *dev_ctx, *dropmask);
      }
    }
  }
  auto dropmask_tensor = dropmask.get();

  // attention_score
  std::shared_ptr<phi::DenseTensor> attention_score =
      std::make_shared<phi::DenseTensor>();
  attention_score->Resize(query_tensor.dims());
  (*dev_ctx).Alloc(attention_score.get(), query_tensor.dtype());

  // softmax_max
  int64_t S0 = query_tensor_dims[1];
  int64_t S1 = key_tensor_dims[1];
  int64_t B = query_tensor_dims[0];
  std::shared_ptr<phi::DenseTensor> softmax_max =
      std::make_shared<phi::DenseTensor>();
  softmax_max->Resize({B, head_num, S0, 8});
  (*dev_ctx).Alloc(softmax_max.get(), phi::DataType::FLOAT32);

  // softmax_sum
  std::shared_ptr<phi::DenseTensor> softmax_sum =
      std::make_shared<phi::DenseTensor>();
  softmax_sum->Resize({B, head_num, S0, 8});
  (*dev_ctx).Alloc(softmax_sum.get(), phi::DataType::FLOAT32);

  // softmax_out,此处最好填空tensor，但是paddle目前传空tensor会报错！！！！
  std::shared_ptr<phi::DenseTensor> softmax_out =
      std::make_shared<phi::DenseTensor>();
  softmax_out->Resize({1});
  (*dev_ctx).Alloc(softmax_out.get(), query_tensor.dtype());

  // numel_tensor
  std::shared_ptr<phi::DenseTensor> numel_tensor =
      std::make_shared<phi::DenseTensor>();
  numel_tensor->Resize({1});
  (*dev_ctx).Alloc(numel_tensor.get(), phi::DataType::INT64);
  custom_kernel::FillNpuTensorWithConstant(
      numel_tensor.get(), *dev_ctx, numels);

  // BSND:(batch,seq_len,head_num,head_dim)
  char* input_layout_ptr = "BSND";
  EXEC_NPU_CMD(aclnnFlashAttentionScore,
               *dev_ctx,
               query_tensor,
               key_tensor,
               value_tensor,
               realShiftOptional,
               *dropmask_tensor,
               padding_mask,
               *attn_mask_tensor,
               prefixOptional,
               scale,
               keep_prob,
               pre_tockens,
               next_tockens,
               head_num,
               input_layout_ptr,
               inner_precise,
               sparseModeOptional,
               *softmax_max,
               *softmax_sum,
               *softmax_out,
               *attention_score);
  return {paddle::Tensor(attention_score),
          paddle::Tensor(softmax_max),
          paddle::Tensor(softmax_sum),
          paddle::Tensor(softmax_out),
          paddle::Tensor(seed_tensor),
          paddle::Tensor(offset_tensor),
          paddle::Tensor(numel_tensor)};
}

std::vector<paddle::Tensor> npu_flash_attention_grad(
    const paddle::Tensor& query,
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::Tensor& grad_out,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& softmax_max,
    const paddle::optional<paddle::Tensor>& softmax_sum,
    const paddle::optional<paddle::Tensor>& softmax_out,
    const paddle::optional<paddle::Tensor>& attention_score,
    const paddle::optional<paddle::Tensor>& seed_tensor,
    const paddle::optional<paddle::Tensor>& offset_tensor,
    const paddle::optional<paddle::Tensor>& numel_tensor,
    float dropout,
    bool casual,
    bool is_triangle_upper_mask) {
  // q,k,v [batch_size, seq_len, num_heads, head_dim]
  auto dev_ctx = getcontext(query);
  // q,k,v [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = paddletensor2densortensor(query);
  auto key_tensor = paddletensor2densortensor(key);
  auto value_tensor = paddletensor2densortensor(value);
  auto grad_tensor = paddletensor2densortensor(grad_out);

  auto query_tensor_dims = phi::vectorize(query_tensor.dims());
  auto key_tensor_dims = phi::vectorize(key_tensor.dims());

  PADDLE_ENFORCE_EQ(query_tensor_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim [batch_size, "
                        "seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(query_tensor_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim [batch_size, "
                        "seq_len, num_heads, head_dim]"));
  PADDLE_ENFORCE_EQ(key_tensor.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "flash_attn receive input with dim [batch_size, "
                        "seq_len, num_heads, head_dim]"));
  PD_CHECK(dropout >= 0 && dropout <= 1,
           "The dropout value must be in range of [0, 1], but got ",
           dropout);

  const int64_t head_num = query_tensor_dims[2];
  const double scale = 1.0f / std::sqrt(query_tensor_dims[3]);

  void* realShiftOptional = nullptr;
  void* prefixOptional = nullptr;
  void* drop_mask = nullptr;
  void* padding_mask = nullptr;

  int64_t pre_tockens = 65536;
  int64_t next_tockens = 65536;
  int64_t inner_precise = 0;
  int64_t sparseModeOptional = 0;
  int64_t seed = 0;
  int64_t offset = 0;
  int64_t numel = 0;

  numel = get_single(numel_tensor, *dev_ctx);
  // drop_mask
  double keep_prob = 1 - dropout;
  std::shared_ptr<phi::DenseTensor> dropmask =
      std::make_shared<phi::DenseTensor>();
  std::vector<int64_t> length_shape = {numel};

  if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
    dropmask->Resize({numel / 8});
    seed = get_single(seed_tensor, *dev_ctx);
    offset = get_single(offset_tensor, *dev_ctx);
    (*dev_ctx).Alloc(dropmask.get(), phi::DataType::UINT8);
    EXEC_NPU_CMD(aclnnDropoutGenMask,
                 *dev_ctx,
                 length_shape,
                 dropout,
                 seed,
                 offset,
                 *dropmask);
  } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
    dropmask->Resize({numel / 8});
    (*dev_ctx).Alloc(dropmask.get(), phi::DataType::UINT8);
    EXEC_NPU_CMD(aclnnInplaceZero, *dev_ctx, *dropmask);
  }
  auto dropmask_tensor = dropmask.get();

  // atten_mask
  phi::DenseTensor* attn_mask_tensor = nullptr;
  std::shared_ptr<phi::DenseTensor> attn_mask_tensor_null =
      std::make_shared<phi::DenseTensor>();
  // casual ，用户mask无效，用上三角矩阵
  if (casual) {
    VLOG(3) << "Backward flash attention with casual mask.attn_mask is "
               "invalid, and use upper triangular matrix mask";
    int64_t diagonal = 0;
    attn_mask_tensor_null->Resize({query_tensor_dims[1], query_tensor_dims[1]});
    (*dev_ctx).Alloc(attn_mask_tensor_null.get(), phi::DataType::BOOL);
    EXEC_NPU_CMD(aclnnInplaceOne, *dev_ctx, *attn_mask_tensor_null);
    EXEC_NPU_CMD(aclnnTril,
                 *dev_ctx,
                 *attn_mask_tensor_null,
                 diagonal,
                 *attn_mask_tensor_null);
    EXEC_NPU_CMD(aclnnBitwiseNot,
                 *dev_ctx,
                 *attn_mask_tensor_null,
                 *attn_mask_tensor_null);
    attn_mask_tensor = attn_mask_tensor_null.get();
    next_tockens = 0;
  } else if (attn_mask) {
    VLOG(3) << "Forward flash attention with user defined mask";
    // 用户指定mask
    auto attn_mask_ptr = *(attn_mask.get_ptr());
    attn_mask_tensor =
        static_cast<phi::DenseTensor*>(attn_mask_ptr.impl().get());
    if (is_triangle_upper_mask) {
      next_tockens = 0;
    }
  } else {
    VLOG(3) << "Forward flash attention without mask";
    // 无mask
    attn_mask_tensor = attn_mask_tensor_null.get();
  }

  // softmax_max
  const phi::DenseTensor* softmax_max_tensor_notnull = nullptr;
  std::shared_ptr<phi::DenseTensor> softmax_max_tensor_null =
      std::make_shared<phi::DenseTensor>();
  if (softmax_max) {
    auto softmax_max_ptr = *(softmax_max.get_ptr());
    softmax_max_tensor_notnull =
        static_cast<const phi::DenseTensor*>(softmax_max_ptr.impl().get());
  }
  auto softmax_max_tensor =
      softmax_max ? softmax_max_tensor_notnull : softmax_max_tensor_null.get();
  // softmax_sum
  const phi::DenseTensor* softmax_sum_tensor_notnull = nullptr;
  std::shared_ptr<phi::DenseTensor> softmax_sum_tensor_null =
      std::make_shared<phi::DenseTensor>();
  if (softmax_sum) {
    auto softmax_sum_ptr = *(softmax_sum.get_ptr());
    softmax_sum_tensor_notnull =
        static_cast<const phi::DenseTensor*>(softmax_sum_ptr.impl().get());
  }
  auto softmax_sum_tensor =
      softmax_sum ? softmax_sum_tensor_notnull : softmax_sum_tensor_null.get();
  // softmax_out
  const phi::DenseTensor* softmax_out_tensor_notnull = nullptr;
  std::shared_ptr<phi::DenseTensor> softmax_out_tensor_null =
      std::make_shared<phi::DenseTensor>();
  if (softmax_out) {
    auto softmax_sum_ptr = *(softmax_out.get_ptr());
    softmax_out_tensor_notnull =
        static_cast<const phi::DenseTensor*>(softmax_sum_ptr.impl().get());
  }
  auto softmax_out_tensor =
      softmax_out ? softmax_out_tensor_notnull : softmax_out_tensor_null.get();
  // attention_score
  const phi::DenseTensor* attention_score_out_tensor_notnull = nullptr;
  std::shared_ptr<phi::DenseTensor> attention_score_out_tensor_null =
      std::make_shared<phi::DenseTensor>();
  if (attention_score) {
    auto attention_score_out_ptr = *(attention_score.get_ptr());
    attention_score_out_tensor_notnull = static_cast<const phi::DenseTensor*>(
        attention_score_out_ptr.impl().get());
  }
  auto attention_score_out_tensor = attention_score
                                        ? attention_score_out_tensor_notnull
                                        : attention_score_out_tensor_null.get();

  std::shared_ptr<phi::DenseTensor> dq_out =
      std::make_shared<phi::DenseTensor>();
  dq_out->Resize(query_tensor.dims());
  (*dev_ctx).Alloc(dq_out.get(), query_tensor.dtype());

  std::shared_ptr<phi::DenseTensor> dk_out =
      std::make_shared<phi::DenseTensor>();
  dk_out->Resize(key_tensor.dims());
  (*dev_ctx).Alloc(dk_out.get(), key_tensor.dtype());

  std::shared_ptr<phi::DenseTensor> dv_out =
      std::make_shared<phi::DenseTensor>();
  dv_out->Resize(value_tensor.dims());
  (*dev_ctx).Alloc(dv_out.get(), value_tensor.dtype());
#if (CANN_VERSION_CODE >= 700000 && CANN_VERSION_CODE < 800000)
  // dpse_out在cann8.0上需要传一个shape为0的tensor
  std::shared_ptr<phi::DenseTensor> dpse_out =
      std::make_shared<phi::DenseTensor>();
  dpse_out->Resize({1});
  (*dev_ctx).Alloc(dpse_out.get(), query_tensor.dtype());
#else
  void* dpse_out_null = nullptr;
#endif
  char* input_layout_ptr = "BSND";
#if (CANN_VERSION_CODE >= 700000 && CANN_VERSION_CODE < 800000)
  EXEC_NPU_CMD(aclnnFlashAttentionScoreGrad,
               *dev_ctx,
               query_tensor,
               key_tensor,
               value_tensor,
               grad_tensor,
               realShiftOptional,
               *dropmask_tensor,
               padding_mask,
               *attn_mask_tensor,
               *softmax_max_tensor,
               *softmax_sum_tensor,
               *softmax_out_tensor,
               *attention_score_out_tensor,
               prefixOptional,
               scale,
               keep_prob,
               pre_tockens,
               next_tockens,
               head_num,
               input_layout_ptr,
               inner_precise,
               sparseModeOptional,
               *dq_out,
               *dk_out,
               *dv_out,
               *dpse_out);
#else
  EXEC_NPU_CMD(aclnnFlashAttentionScoreGrad,
               *dev_ctx,
               query_tensor,
               key_tensor,
               value_tensor,
               grad_tensor,
               realShiftOptional,
               *dropmask_tensor,
               padding_mask,
               *attn_mask_tensor,
               *softmax_max_tensor,
               *softmax_sum_tensor,
               *softmax_out_tensor,
               *attention_score_out_tensor,
               prefixOptional,
               scale,
               keep_prob,
               pre_tockens,
               next_tockens,
               head_num,
               input_layout_ptr,
               inner_precise,
               sparseModeOptional,
               *dq_out,
               *dk_out,
               *dv_out,
               dpse_out_null);
#endif
  return {
      paddle::Tensor(dq_out), paddle::Tensor(dk_out), paddle::Tensor(dv_out)};
}

PD_BUILD_OP(flash_attention_npu)
    .Inputs({"query",
             "key",
             "value",
             paddle::Optional("fixed_seed_offset"),
             paddle::Optional("attn_mask")})
    .Outputs({"attention_score",
              "softmax_max",
              "softmax_sum",
              "softmax_out",
              "seed",
              "offset",
              "numel"})
    .Attrs({"dropout: float",
            "casual:bool",
            "return_softmax:bool",
            "is_test:bool",
            "is_triangle_upper_mask:bool"})
    .SetKernelFn(PD_KERNEL(npu_flash_attention))
    .SetInferShapeFn(PD_INFER_SHAPE(
        fusedattentionInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_GRAD_OP(flash_attention_npu)
    .Inputs({"query",
             "key",
             "value",
             paddle::Grad("attention_score"),
             paddle::Optional("attn_mask"),
             "softmax_max",
             "softmax_sum",
             "softmax_out",
             "attention_score",
             "seed",
             "offset",
             "numel"})
    .Outputs({paddle::Grad("query"),
              paddle::Grad("key"),
              paddle::Grad("value")})
    .Attrs({"dropout: float", "casual:bool", "is_triangle_upper_mask:bool"})
    .SetKernelFn(PD_KERNEL(npu_flash_attention_grad))
    .SetInferShapeFn(PD_INFER_SHAPE(
        fusedattentionInferShape));  // neccessary if the op has muti_inputs
