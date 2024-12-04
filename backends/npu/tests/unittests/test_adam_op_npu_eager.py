#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from npu_utils import check_soc_version

SEED = 2021


def adam_step(inputs, attributes):
    """
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    """
    param = convert_uint16_to_float(inputs["Param"])
    grad = convert_uint16_to_float(inputs["Grad"])
    moment1 = convert_uint16_to_float(inputs["Moment1"])
    moment2 = convert_uint16_to_float(inputs["Moment2"])
    lr = convert_uint16_to_float(inputs["LearningRate"])
    beta1_pow = convert_uint16_to_float(inputs["Beta1Pow"])
    beta2_pow = convert_uint16_to_float(inputs["Beta2Pow"])

    epsilon = attributes["epsilon"]

    if "beta1" in attributes:
        beta1 = attributes["beta1"]
    else:
        beta1 = convert_uint16_to_float(inputs["Beta1Tensor"][0])
    if "beta2" in attributes:
        beta2 = attributes["beta2"]
    else:
        beta2 = convert_uint16_to_float(inputs["Beta2Tensor"][0])

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))

    moment2_max_out = np.empty_like(moment2_out)
    return param_out, moment1_out, moment2_out, moment2_max_out


class TestAdam(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        grad = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        moment1 = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        # The second moment is positive
        moment2 = convert_float_to_uint16(
            np.random.random((102, 105)).astype(self.dtype)
        )
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = convert_float_to_uint16(np.zeros((102, 105)).astype(self.dtype))

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "Moment2Max": moment2_max,
            "LearningRate": convert_float_to_uint16(
                np.array([learning_rate]).astype(self.dtype)
            ),
            "Beta1Pow": convert_float_to_uint16(
                np.array([beta1_pow]).astype(self.dtype)
            ),
            "Beta2Pow": convert_float_to_uint16(
                np.array([beta2_pow]).astype(self.dtype)
            ),
        }

        self.attrs = {
            "epsilon": epsilon,
            "beta1": beta1,
            "beta2": beta2,
            "amsgrad": False,
        }

        param_out, moment1_out, moment2_out, moment2_max_out = adam_step(
            self.inputs, self.attrs
        )

        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "Moment2MaxOut": moment2_max_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([beta1_pow]).astype("float32") * beta1,
            "Beta2PowOut": np.array([beta2_pow]).astype("float32") * beta2,
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=4e-3
        )


class TestAdamWithEpsilonTensor(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        grad = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        moment1 = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        # The second moment is positive
        moment2 = convert_float_to_uint16(
            np.random.random((102, 105)).astype(self.dtype)
        )
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = convert_float_to_uint16(np.zeros((102, 105)).astype(self.dtype))

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "Moment2Max": moment2_max,
            "LearningRate": convert_float_to_uint16(
                np.array([learning_rate]).astype(self.dtype)
            ),
            "Beta1Pow": convert_float_to_uint16(
                np.array([beta1_pow]).astype(self.dtype)
            ),
            "Beta2Pow": convert_float_to_uint16(
                np.array([beta2_pow]).astype(self.dtype)
            ),
            "Beta1Tensor": convert_float_to_uint16(
                np.array([beta1]).astype(self.dtype)
            ),
            "Beta2Tensor": convert_float_to_uint16(
                np.array([beta2]).astype(self.dtype)
            ),
            "EpsilonTensor": convert_float_to_uint16(
                np.array([epsilon]).astype(self.dtype)
            ),
        }

        self.attrs = {"epsilon": epsilon, "amsgrad": False}

        param_out, moment1_out, moment2_out, moment2_max_out = adam_step(
            self.inputs, self.attrs
        )

        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "Moment2MaxOut": moment2_max_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([beta1_pow]).astype("float32") * beta1,
            "Beta2PowOut": np.array([beta2_pow]).astype("float32") * beta2,
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=4e-3
        )


class TestAdamOpWithSkipUpdate(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        grad = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        moment1 = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        # The second moment is positive
        moment2 = convert_float_to_uint16(
            np.random.random((102, 105)).astype(self.dtype)
        )
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = convert_float_to_uint16(np.zeros((102, 105)).astype(self.dtype))

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "Moment2Max": moment2_max,
            "LearningRate": convert_float_to_uint16(
                np.array([learning_rate]).astype(self.dtype)
            ),
            "Beta1Pow": convert_float_to_uint16(
                np.array([beta1_pow]).astype(self.dtype)
            ),
            "Beta2Pow": convert_float_to_uint16(
                np.array([beta2_pow]).astype(self.dtype)
            ),
            "Beta1Tensor": convert_float_to_uint16(
                np.array([beta1]).astype(self.dtype)
            ),
            "Beta2Tensor": convert_float_to_uint16(
                np.array([beta2]).astype(self.dtype)
            ),
            "EpsilonTensor": convert_float_to_uint16(
                np.array([epsilon]).astype(self.dtype)
            ),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        self.attrs = {"epsilon": epsilon, "amsgrad": False}

        self.outputs = {
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "Moment2MaxOut": moment2_max,
            "ParamOut": param,
            "Beta1PowOut": self.inputs["Beta1Pow"],
            "Beta2PowOut": self.inputs["Beta2Pow"],
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=4e-3
        )


class TestAdamOpWithGlobalBetaPow(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        grad = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        moment1 = convert_float_to_uint16(
            np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        )
        # The second moment is positive
        moment2 = convert_float_to_uint16(
            np.random.random((102, 105)).astype(self.dtype)
        )
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = convert_float_to_uint16(np.zeros((102, 105)).astype(self.dtype))

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "Moment2Max": moment2_max,
            "LearningRate": convert_float_to_uint16(
                np.array([learning_rate]).astype(self.dtype)
            ),
            "Beta1Pow": convert_float_to_uint16(
                np.array([beta1_pow]).astype(self.dtype)
            ),
            "Beta2Pow": convert_float_to_uint16(
                np.array([beta2_pow]).astype(self.dtype)
            ),
            "Beta1Tensor": convert_float_to_uint16(
                np.array([beta1]).astype(self.dtype)
            ),
            "Beta2Tensor": convert_float_to_uint16(
                np.array([beta2]).astype(self.dtype)
            ),
            "EpsilonTensor": convert_float_to_uint16(
                np.array([epsilon]).astype(self.dtype)
            ),
        }

        attributes = {"epsilon": epsilon, "amsgrad": False}

        param_out, moment1_out, moment2_out, moment2_max_out = adam_step(
            self.inputs, attributes
        )

        self.attrs = {"use_global_beta_pow": True}

        # use_global_beta_pow=True, Beta1PowOut and Beta2PowOut are empty.
        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "Moment2MaxOut": moment2_max_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([]),
            "Beta2PowOut": np.array([]),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=4e-3
        )


if __name__ == "__main__":
    unittest.main()
