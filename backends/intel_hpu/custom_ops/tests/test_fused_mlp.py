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

import argparse

import paddle
import paddlenlp_ops
import paddle.profiler as profiler

paddle.device.set_device("intel_hpu")

paddle.seed(20241214)


def init_data(
    batch_size=8,
    seqence_len=128,
    hidden_size=256,
    intermediate_size=512,
    dtype="bfloat16",
):
    with paddle.no_grad():
        x = paddle.rand(
            [batch_size, seqence_len, hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)

        gate_weight = paddle.normal(
            mean=0.0, std=0.02, shape=[hidden_size, intermediate_size]
        ).astype(dtype)
        up_weight = paddle.normal(
            mean=1.0, std=0.05, shape=[hidden_size, intermediate_size]
        ).astype(dtype)
        down_weight = paddle.normal(
            mean=0.5, std=0.12, shape=[intermediate_size, hidden_size]
        ).astype(dtype)
        proj_weight = paddle.concat([gate_weight, up_weight], axis=1)

    return x, gate_weight, up_weight, down_weight, proj_weight


def ref_mlp(
    x,
    gate_weight,
    up_weight,
    down_weight,
):
    def swiglu_naive(x, up=None):
        if up is not None:
            gate = x
        else:
            gate, up = paddle.chunk(x, chunks=2, axis=-1)
        silu = gate / (paddle.exp(-gate) + 1)
        return silu * up

    gate = paddle.matmul(x, gate_weight)
    up = paddle.matmul(x, up_weight)
    swiglu = swiglu_naive(x=gate, up=up)
    res = paddle.matmul(swiglu, down_weight)

    return res.numpy()


class refMlpOP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.x, self.gate_weight, self.up_weight, self.down_weight, _ = init_data()

    def forward(self):
        mlp_out_ref = ref_mlp(
            self.x,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
        )
        return mlp_out_ref


class fusedMlpOP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.x, self.gate_weight, self.up_weight, self.down_weight, _ = init_data()

    def forward(self):
        fused_mlp_out = paddlenlp_ops.fused_mlp(
            self.x,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
        )
        return fused_mlp_out


class fusedGateUpMlpOP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.x, _, _, self.down_weight, self.proj_weight = init_data()

    def forward(self):
        fused_gateup_mlp_out = paddlenlp_ops.fused_mlp(
            self.x,
            self.proj_weight,
            None,
            self.down_weight,
        )
        return fused_gateup_mlp_out


def run_profile(my_profile_func):
    prof = profiler.Profiler(
        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE],
        on_trace_ready=profiler.export_chrome_tracing("./profile"),
    )
    prof.start()
    for iter in range(20):
        with paddle.no_grad():
            mlp_out = my_profile_func()
    prof.stop()


def run_accuracy_check():
    ref_mlp = refMlpOP()
    fused_mlp = fusedMlpOP()
    fused_gate_up_mlp = fusedGateUpMlpOP()

    golden_res = ref_mlp()
    fused_res = fused_mlp()
    fused_gate_up_res = fused_gate_up_mlp()

    print((fused_res == golden_res).all())
    print((fused_gate_up_res == golden_res).all())


def main():
    parser = argparse.ArgumentParser(description="Run profile or accuracy check")
    parser.add_argument("--profile", action="store_true", help="Run profile")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy check")
    args = parser.parse_args()
    if args.profile:
        run_profile(fusedGateUpMlpOP())
        run_profile(fusedMlpOP())
        run_profile(refMlpOP())
    else:
        run_accuracy_check()


if __name__ == "__main__":
    main()
