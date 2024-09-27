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

#include <istream>
#include <sstream>
#include <vector>

#include "common/gcu_funcs.h"
#include "common/gcu_op_runner.h"
#include "kernels/funcs/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

#define DEFINE_MODE_DETECTOR(MODE_NAME, MODE_STR)       \
  inline bool is_##MODE_NAME(const std::string& mode) { \
    return mode == #MODE_STR;                           \
  }

DEFINE_MODE_DETECTOR(lstm, LSTM);
DEFINE_MODE_DETECTOR(gru, GRU);
DEFINE_MODE_DETECTOR(rnn_relu, RNN_RELU);
DEFINE_MODE_DETECTOR(rnn_tanh, RNN_TANH);

void ResetParameterVector(
    const std::vector<const phi::DenseTensor*>& raw_params_vec,
    int num_layers,
    bool is_bidirec,
    std::vector<std::vector<phi::DenseTensor>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    std::vector<phi::DenseTensor> tensor_list;
    tensor_list.reserve(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      phi::DenseTensor tensor_holder;
      tensor_list.emplace_back(tensor_holder);
    }
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      tensor_list[j] = *raw_params_vec[tensor_idx];
    }
    params_vec->emplace_back(tensor_list);
  }
}

template <typename Context>
void rnn_slice(const Context& dev_ctx,
               const phi::DenseTensor& input,
               std::vector<phi::DenseTensor>& out) {  // NOLINT
  // Warn : This function only slices the index 0 dimension.
  std::vector<int64_t> axes_t = {0};
  auto meta = phi::DenseTensorMeta(
      input.dtype(),
      phi::make_ddim({1, input.dims().at(1), input.dims().at(2)}));

  for (int i = 0; i < input.dims().at(0); ++i) {
    std::vector<int64_t> starts = {i};
    phi::DenseTensor output_tmp = TensorEmpty(dev_ctx, meta);
    custom_kernel::SliceBase(dev_ctx, input, axes_t, starts, &output_tmp);
    out.push_back(output_tmp);
  }
}

template <typename T, typename Context>
void SlicePreState(const Context& dev_ctx,
                   const phi::DenseTensor& pre_state,
                   int num_layers,
                   bool is_bidirec,
                   std::vector<std::vector<phi::DenseTensor>>* pre_state_vec) {
  auto stream = dev_ctx.stream();
  const int& direction_num = is_bidirec ? 2 : 1;

  std::vector<phi::DenseTensor> out;
  rnn_slice<Context>(dev_ctx, pre_state, out);

  int k = 0;
  for (int i = 0; i < num_layers; ++i) {
    std::vector<phi::DenseTensor> tensor_list;
    for (int j = 0; j < direction_num; ++j) {
      tensor_list.emplace_back(out[k]);
      ++k;
    }
    pre_state_vec->emplace_back(tensor_list);
  }
}

template <typename Context>
void rnn_concat(const Context& dev_ctx,
                std::vector<phi::DenseTensor> input,
                phi::DenseTensor& out,  // NOLINT
                int64_t dim) {
  auto out_tensor = CreateTopsatenTensor(out);
  std::vector<topsatenTensor> in_tensors;
  for (const auto& tensor : input) {
    in_tensors.emplace_back(CreateTopsatenTensor(tensor));
  }
  std::string abstract_info =
      custom_kernel::GetAbstractInfo("topsatenCat", out, input, dim);
  LAUNCH_TOPSATENOP_WITH_RAW_ATEN_DEF(
      topsatenCat, dev_ctx, abstract_info, out_tensor, in_tensors, dim);
}

template <typename T, typename Context>
void LSTMKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& init_h,
                const phi::DenseTensor& init_c,
                const phi::DenseTensor& wi,
                const phi::DenseTensor& wh,
                const phi::DenseTensor& bi,
                const phi::DenseTensor& bh,
                const std::vector<int>& SequenceLength,
                float dropout_prob,
                phi::DenseTensor* out,
                phi::DenseTensor* last_h,
                phi::DenseTensor* last_c,
                bool is_b) {
  std::vector<phi::DenseTensor> input_list;
  rnn_slice<Context>(dev_ctx, x, input_list);
  if (is_b) {
    std::reverse(input_list.begin(), input_list.end());
  }

  std::vector<phi::DenseTensor> last_h_list;
  std::vector<phi::DenseTensor> last_c_list;
  for (int32_t i = 0; i < x.dims().at(0); ++i) {
    phi::DenseTensor tmp_h, tmp_c;
    tmp_h.Resize(
        phi::make_ddim({1, last_h->dims().at(1), last_h->dims().at(2)}));
    tmp_c.Resize(
        phi::make_ddim({1, last_c->dims().at(1), last_c->dims().at(2)}));
    dev_ctx.template Alloc<T>(&tmp_h);
    dev_ctx.template Alloc<T>(&tmp_c);
    last_h_list.push_back(tmp_h);
    last_c_list.push_back(tmp_c);
  }

  std::vector<phi::DenseTensor> out_list;
  std::vector<phi::DenseTensor> init_list;

  for (int32_t i = 0; i < x.dims().at(0); ++i) {
    out_list.clear();
    init_list.clear();
    if (i > 0) {
      init_list.push_back(last_h_list[i - 1]);
      init_list.push_back(last_c_list[i - 1]);
      out_list.push_back(last_h_list[i]);
      out_list.push_back(last_c_list[i]);
    } else {
      init_list.push_back(init_h);
      init_list.push_back(init_c);
      out_list.push_back(last_h_list[i]);
      out_list.push_back(last_c_list[i]);
    }

    LAUNCH_TOPSATENOP(topsatenLstmCell,
                      dev_ctx,
                      out_list,
                      input_list[i],
                      init_list,
                      wi,
                      wh,
                      bi,
                      bh);
  }

  if (is_b) {
    std::reverse(last_h_list.begin(), last_h_list.end());
  }

  rnn_concat(dev_ctx, last_h_list, *out, 0);
  *last_h = last_h_list[last_h_list.size() - 1];
  *last_c = last_c_list[last_c_list.size() - 1];
}

template <typename T, typename Context>
void DropoutHelper(const Context& dev_ctx,
                   DenseTensor* x,
                   DenseTensor* y,
                   const DenseTensor* mask,
                   float dropout_prob) {
  if (dropout_prob == 1.0f) {
    auto meta = phi::DenseTensorMeta(x->dtype(), x->dims());
    *y = TensorZeros(dev_ctx, meta);
  } else {
    LAUNCH_TOPSATENOP(topsatenMul, dev_ctx, *y, *x, *mask);
    LAUNCH_TOPSATENOP(topsatenDiv,
                      dev_ctx,
                      *y,
                      *y,
                      phi::Scalar(static_cast<float>(1.0f - dropout_prob)));
  }
}

template <typename T, typename Context>
void DropoutGcuFunctionInplace(const Context& dev_ctx,
                               DenseTensor* x,
                               DenseTensor* y,
                               DenseTensor* mask,
                               const float& dropout_prob,
                               const int& seed_number,
                               bool is_test,
                               bool* is_has_reset) {
  if (is_test) {
    return;
  }
  phi::DenseTensor mask_temp;
  mask_temp.Resize(mask->dims());
  dev_ctx.template HostAlloc<T>(&mask_temp);
  size_t size = common::product(x->dims());
  auto* mask_data = mask_temp.data<T>();
  if (!(*is_has_reset)) {
    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      std::fill(mask_data, mask_data + size, static_cast<T>(0));
    } else {
      std::shared_ptr<std::mt19937_64> engine;
      if (seed_number) {
        engine = std::make_shared<std::mt19937_64>();
        engine->seed(seed_number);
      } else {
        engine = dev_ctx.GetGenerator()->GetCPUEngine();
      }
      std::uniform_real_distribution<float> dist(0, 1);
      for (size_t i = 0; i < size; ++i) {
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
        } else {
          mask_data[i] = 1;
        }
      }
    }
    *is_has_reset = true;
  }
  DropoutHelper<T, Context>(dev_ctx, x, y, &mask_temp, dropout_prob);
}

template <typename T, typename Context>
void RnnKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const std::vector<const phi::DenseTensor*>& pre_state,
               const std::vector<const phi::DenseTensor*>& weight_list,
               const paddle::optional<phi::DenseTensor>& sequence_length,
               float dropout_prob,
               bool is_bidirec,
               int input_size,
               int hidden_size,
               int num_layers,
               const std::string& mode,
               int seed,
               bool is_test,
               phi::DenseTensor* out,
               phi::DenseTensor* dropout_state,
               std::vector<phi::DenseTensor*> state,
               phi::DenseTensor* reserve) {
  auto init_h = pre_state[0];
  auto init_c = pre_state[1];

  int direction_num = is_bidirec ? 2 : 1;
  const auto& init_h_dims = init_h->dims();
  PADDLE_ENFORCE_EQ(init_h_dims[0],
                    num_layers * direction_num,
                    phi::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of init hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers,
                        init_h_dims[0]));
  if (is_lstm(mode)) {
    const auto& init_c_dims = init_c->dims();  // NOLINT
    PADDLE_ENFORCE_EQ(init_c_dims[0],
                      num_layers * direction_num,
                      phi::errors::InvalidArgument(
                          "The num_layers of in RNN layer must be the same as "
                          "first dim of cell state hidden, but received"
                          " num_layers:%d, dim:%d",
                          num_layers,
                          init_h_dims[0]));
  }

  {  // dropout_state data
    if (dropout_state) {
      if (dropout_state->numel() != out->numel()) dropout_state->clear();
    }

    dropout_state->Resize(out->dims());
    dev_ctx.template Alloc<uint8_t>(dropout_state);

    auto shape = phi::vectorize(dropout_state->meta().dims);
    LAUNCH_TOPSATENOP(topsatenOnes,
                      dev_ctx,
                      *dropout_state,
                      shape,
                      dropout_state->meta().dtype);
  }

  auto last_h = state[0];
  auto last_c = state[1];
  dev_ctx.template Alloc<T>(last_h);
  dev_ctx.template Alloc<T>(last_c);

  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    // reset parameter, init_h and init_c
    std::vector<std::vector<phi::DenseTensor>> parameter_lists;
    parameter_lists.reserve(num_layers);
    custom_kernel::ResetParameterVector(
        weight_list, num_layers, is_bidirec, &parameter_lists);

    std::vector<std::vector<phi::DenseTensor>> init_h_list, init_c_list;
    init_h_list.reserve(num_layers);
    init_c_list.reserve(num_layers);
    custom_kernel::SlicePreState<T, Context>(
        dev_ctx, *init_h, num_layers, is_bidirec, &init_h_list);
    custom_kernel::SlicePreState<T, Context>(
        dev_ctx, *init_c, num_layers, is_bidirec, &init_c_list);

    if (is_lstm(mode)) {
      std::vector<phi::DenseTensor> out_vec, last_h_vec, last_c_vec;
      phi::DenseTensor input = x;
      bool has_dropout_reset = false;
      for (int i = 0; i < num_layers; ++i) {
        int32_t seq_length = x.dims().at(0);
        int32_t batch_size = x.dims().at(1);
        int32_t hidden_size = init_h->dims().at(2);

        phi::DenseTensor out_tmp_f, last_h_tmp_f, last_c_tmp_f;
        out_tmp_f.Resize(phi::make_ddim({seq_length, batch_size, hidden_size}));
        last_h_tmp_f.Resize(phi::make_ddim({1, batch_size, hidden_size}));
        last_c_tmp_f.Resize(phi::make_ddim({1, batch_size, hidden_size}));
        dev_ctx.template Alloc<T>(&out_tmp_f);
        dev_ctx.template Alloc<T>(&last_h_tmp_f);
        dev_ctx.template Alloc<T>(&last_c_tmp_f);

        if (direction_num == 1) {
          out_vec.clear();
          LSTMKernel<T, Context>(dev_ctx,
                                 input,
                                 init_h_list[i][0],
                                 init_c_list[i][0],
                                 parameter_lists[i][0],
                                 parameter_lists[i][1],
                                 parameter_lists[i][2],
                                 parameter_lists[i][3],
                                 {},
                                 dropout_prob,
                                 &out_tmp_f,
                                 &last_h_tmp_f,
                                 &last_c_tmp_f,
                                 false);

          input = out_tmp_f;
          out_vec.push_back(out_tmp_f);
          last_h_vec.push_back(last_h_tmp_f);
          last_c_vec.push_back(last_c_tmp_f);
        } else {
          out_vec.clear();
          phi::DenseTensor out_tmp_b, last_h_tmp_b, last_c_tmp_b;
          out_tmp_b.Resize(
              phi::make_ddim({seq_length, batch_size, hidden_size}));
          last_h_tmp_b.Resize(phi::make_ddim({1, batch_size, hidden_size}));
          last_c_tmp_b.Resize(phi::make_ddim({1, batch_size, hidden_size}));
          dev_ctx.template Alloc<T>(&out_tmp_b);
          dev_ctx.template Alloc<T>(&last_h_tmp_b);
          dev_ctx.template Alloc<T>(&last_c_tmp_b);

          LSTMKernel<T, Context>(dev_ctx,
                                 input,
                                 init_h_list[i][0],
                                 init_c_list[i][0],
                                 parameter_lists[i][0],
                                 parameter_lists[i][1],
                                 parameter_lists[i][2],
                                 parameter_lists[i][3],
                                 {},
                                 dropout_prob,
                                 &out_tmp_f,
                                 &last_h_tmp_f,
                                 &last_c_tmp_f,
                                 false);

          last_h_vec.push_back(last_h_tmp_f);
          last_c_vec.push_back(last_c_tmp_f);

          LSTMKernel<T, Context>(dev_ctx,
                                 input,
                                 init_h_list[i][1],
                                 init_c_list[i][1],
                                 parameter_lists[i][4],
                                 parameter_lists[i][5],
                                 parameter_lists[i][6],
                                 parameter_lists[i][7],
                                 {},
                                 dropout_prob,
                                 &out_tmp_b,
                                 &last_h_tmp_b,
                                 &last_c_tmp_b,
                                 true);

          last_h_vec.push_back(last_h_tmp_b);
          last_c_vec.push_back(last_c_tmp_b);

          std::vector<phi::DenseTensor> new_output{out_tmp_f, out_tmp_b};
          phi::DenseTensor concat_out;
          concat_out.Resize(out->dims());
          dev_ctx.template Alloc<T>(&concat_out);
          rnn_concat(dev_ctx, new_output, concat_out, 2);
          input = concat_out;
          out_vec.push_back(concat_out);
        }

        if (i + 1 < num_layers && dropout_prob != 0) {
          DropoutGcuFunctionInplace<T, Context>(dev_ctx,
                                                &out_vec[0],
                                                &input,
                                                dropout_state,
                                                dropout_prob,
                                                seed,
                                                is_test,
                                                &has_dropout_reset);
        }
      }
      rnn_concat(dev_ctx, last_h_vec, *last_h, 0);
      rnn_concat(dev_ctx, last_c_vec, *last_c, 0);
      *out = std::move(out_vec[0]);
    } else if (is_rnn_relu(mode)) {
      // relu model is not supported
      PADDLE_THROW(phi::errors::NotFound("relu model is not supported."));
    } else if (is_rnn_tanh(mode)) {
      // tanh model is not supported
      PADDLE_THROW(phi::errors::NotFound("tanh model is not supported."));
    } else if (is_gru(mode)) {
      // gru model is not supported
      PADDLE_THROW(phi::errors::NotFound("gru model is not supported."));
    } else {
      // other model is not supported
      PADDLE_THROW(phi::errors::NotFound(" other model is not supported."));
    }
  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(rnn,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RnnKernel,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
