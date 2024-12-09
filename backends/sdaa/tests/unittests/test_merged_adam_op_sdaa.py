# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import unittest
import paddle
import numpy as np
from paddle import _C_ops, _legacy_C_ops
from paddle.base.framework import in_dygraph_mode


def run_adam_op(
    params,
    grads,
    lrs,
    moment1s,
    moment2s,
    beta1_pows,
    beta2_pows,
    master_params,
    epsilon,
    beta1,
    beta2,
    place,
    multi_precision=False,
    use_merged=False,
):
    assert len(params) == len(grads)
    assert len(params) == len(lrs)
    assert len(params) == len(moment1s)
    assert len(params) == len(moment2s)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(master_params)
    paddle.disable_static()
    paddle.set_device("sdaa")

    param_vars = [paddle.base.dygraph.to_variable(p) for p in params]
    grad_vars = [paddle.base.dygraph.to_variable(g) for g in grads]
    lr_vars = [paddle.base.dygraph.to_variable(l) for l in lrs]
    moment1_vars = [paddle.base.dygraph.to_variable(m) for m in moment1s]
    moment2_vars = [paddle.base.dygraph.to_variable(m) for m in moment2s]
    beta1_pow_vars = [paddle.base.dygraph.to_variable(b) for b in beta1_pows]
    beta2_pow_vars = [paddle.base.dygraph.to_variable(b) for b in beta2_pows]
    master_param_vars = [paddle.base.dygraph.to_variable(m_p) for m_p in master_params]

    if not use_merged:
        paddle.set_device("cpu")
        for i in range(len(param_vars)):
            _, _, _, _, _, *_ = _legacy_C_ops.adam(
                param_vars[i],
                grad_vars[i],
                lr_vars[i],
                moment1_vars[i],
                moment2_vars[i],
                beta1_pow_vars[i],
                beta2_pow_vars[i],
                master_param_vars[i],
                param_vars[i],
                moment1_vars[i],
                moment2_vars[i],
                beta1_pow_vars[i],
                beta2_pow_vars[i],
                master_param_vars[i],
                "epsilon",
                epsilon,
                "beta1",
                beta1,
                "beta2",
                beta2,
                "multi_precision",
                multi_precision,
            )
        paddle.set_device("sdaa")
    else:
        if in_dygraph_mode():
            _, _, _, _, _, _ = _C_ops.merged_adam_(
                param_vars,
                grad_vars,
                lr_vars,
                moment1_vars,
                moment2_vars,
                beta1_pow_vars,
                beta2_pow_vars,
                master_param_vars,
                beta1,
                beta2,
                epsilon,
                multi_precision,
                False,
            )
        else:
            _, _, _, _, _, _ = _legacy_C_ops.merged_adam(
                param_vars,
                grad_vars,
                lr_vars,
                moment1_vars,
                moment2_vars,
                beta1_pow_vars,
                beta2_pow_vars,
                master_param_vars,
                param_vars,
                moment1_vars,
                moment2_vars,
                beta1_pow_vars,
                beta2_pow_vars,
                master_param_vars,
                "epsilon",
                epsilon,
                "beta1",
                beta1,
                "beta2",
                beta2,
                "multi_precision",
                multi_precision,
            )

    outputs = {
        "ParamOut": param_vars,
        "Moment1Out": moment1_vars,
        "Moment2Out": moment2_vars,
        "Beta1PowOut": beta1_pow_vars,
        "Beta2PowOut": beta2_pow_vars,
        "MasterParamOut": master_param_vars,
    }

    return outputs


class TestMergedAdam(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
        self.seed = 10
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True

    def gen_rand_data(self, shapes, dtype):
        return [np.random.random(s).astype(dtype) for s in shapes]

    def prepare_data(self, shapes, multi_precision, seed, place):
        np.random.seed(seed)
        mp_dtype = np.float32
        dtype = np.float32
        params = self.gen_rand_data(shapes, dtype)
        grads = self.gen_rand_data(shapes, dtype)
        lrs = self.gen_rand_data([[1], [1], [1], [1]], mp_dtype)
        moment1s = self.gen_rand_data(shapes, mp_dtype)
        moment2s = self.gen_rand_data(shapes, mp_dtype)
        beta1_pows = self.gen_rand_data([[1], [1], [1], [1]], mp_dtype)
        beta2_pows = self.gen_rand_data([[1], [1], [1], [1]], mp_dtype)
        master_params = [p.astype(mp_dtype) for p in params]
        return (
            params,
            grads,
            lrs,
            moment1s,
            moment2s,
            beta1_pows,
            beta2_pows,
            master_params,
        )

    def check_with_place(self, place, multi_precision):
        (
            params,
            grads,
            lrs,
            moment1s,
            moment2s,
            beta1_pows,
            beta2_pows,
            master_params,
        ) = self.prepare_data(self.shapes, multi_precision, self.seed, place)

        def run_op(use_merged):
            return run_adam_op(
                params=params,
                grads=grads,
                lrs=lrs,
                moment1s=moment1s,
                moment2s=moment2s,
                beta1_pows=beta1_pows,
                beta2_pows=beta2_pows,
                master_params=master_params,
                epsilon=0.9,
                beta1=0.9,
                beta2=0.99,
                place=place,
                multi_precision=multi_precision,
                use_merged=use_merged,
            )

        outs1 = run_op(True)
        outs2 = run_op(False)
        self.assertEqual(len(outs1), len(outs2))

        for key in outs1.keys():
            value1 = outs1[key]
            value2 = outs2[key]
            for i in range(len(value1)):
                np.testing.assert_allclose(value1[i], value2[i], rtol=1e-05, atol=1e-07)

    def test_main(self):
        # TODO(wangjr): merged_adam only support float kernel,
        # multi_precision is set to false in float
        for multi_precision in [False]:
            self.check_with_place(self.place, multi_precision)


if __name__ == "__main__":
    unittest.main()
