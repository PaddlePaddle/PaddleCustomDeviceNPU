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
from tests.op_test import OpTest
from npu_utils import check_run_big_shape_test

paddle.enable_static()
SEED = 2021


def adam_step(inputs, attributes):
    """
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    """
    param = inputs["Param"]
    grad = inputs["Grad"]
    moment1 = inputs["Moment1"]
    moment2 = inputs["Moment2"]
    lr = inputs["LearningRate"]
    beta1_pow = inputs["Beta1Pow"]
    beta2_pow = inputs["Beta2Pow"]

    epsilon = attributes["epsilon"]

    if "beta1" in attributes:
        beta1 = attributes["beta1"]
    else:
        beta1 = inputs["Beta1Tensor"][0]
    if "beta2" in attributes:
        beta2 = attributes["beta2"]
    else:
        beta2 = inputs["Beta2Tensor"][0]

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
        param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype(self.dtype)
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = np.zeros((102, 105)).astype(self.dtype)

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
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
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

    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5
        )


@check_run_big_shape_test()
class TestAdamRank1(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_shape()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        grad = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random(self.shape).astype(self.dtype)
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = np.zeros(self.shape).astype(self.dtype)

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
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
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

    def init_shape(self):
        self.shape = (4000, 8192)

    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5
        )


@check_run_big_shape_test()
class TestAdamRank2(TestAdamRank1):
    def init_shape(self):
        self.shape = (8192, 1280)


@check_run_big_shape_test()
class TestAdamRank3(TestAdamRank1):
    def init_shape(self):
        self.shape = (1024, 8192)


@check_run_big_shape_test()
class TestAdamRank4(TestAdamRank1):
    def init_shape(self):
        self.shape = (8192, 7168)


@check_run_big_shape_test()
class TestAdamRank5(TestAdamRank1):
    def init_shape(self):
        self.shape = (3584, 8192)


@check_run_big_shape_test()
class TestAdamRank6(TestAdamRank1):
    def init_shape(self):
        self.shape = (8192, 4000)


class TestAdamWithEpsilonTensor(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype(self.dtype)
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = np.zeros((102, 105)).astype(self.dtype)

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
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
            "Beta1Tensor": np.array([beta1]).astype(self.dtype),
            "Beta2Tensor": np.array([beta2]).astype(self.dtype),
            "EpsilonTensor": np.array([epsilon]).astype(self.dtype),
        }

        self.attrs = {"epsilon": epsilon}

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

    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5
        )


@unittest.skip(reason="disable_ut in Paddle CI")
class TestAdamOpWithSkipUpdate(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype(self.dtype)
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = np.zeros((102, 105)).astype(self.dtype)

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
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
            "Beta1Tensor": np.array([beta1]).astype(self.dtype),
            "Beta2Tensor": np.array([beta2]).astype(self.dtype),
            "EpsilonTensor": np.array([epsilon]).astype(self.dtype),
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

    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5
        )


class TestAdamOpWithGlobalBetaPow(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype(self.dtype)
        # npu not support amsgrad, `moment2_max` is useless
        moment2_max = np.zeros((102, 105)).astype(self.dtype)

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
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
            "Beta1Tensor": np.array([beta1]).astype(self.dtype),
            "Beta2Tensor": np.array([beta2]).astype(self.dtype),
            "EpsilonTensor": np.array([epsilon]).astype(self.dtype),
        }

        attributes = {"epsilon": epsilon, "amsgrad": False}

        param_out, moment1_out, moment2_out, moment2_max_out = adam_step(
            self.inputs, attributes
        )

        self.attrs = {"use_global_beta_pow": True, "amsgrad": False}

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

    def test_check_output(self):
        self.check_output_with_place(
            no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5
        )


def create_test_fp64_class(parent):
    class TestAdamOpFp64Case(parent):
        def init_dtype(self):
            self.dtype = np.float64

        def test_check_output(self):
            self.check_output_with_place(
                no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5, rtol=1e-4
            )

    cls_name = "{0}_{1}".format(parent.__name__, "Fp64")
    TestAdamOpFp64Case.__name__ = cls_name
    globals()[cls_name] = TestAdamOpFp64Case


create_test_fp64_class(TestAdam)
create_test_fp64_class(TestAdamWithEpsilonTensor)
create_test_fp64_class(TestAdamOpWithSkipUpdate)
create_test_fp64_class(TestAdamOpWithGlobalBetaPow)


def create_test_fp16_class(parent):
    class TestAdamOpFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output_with_place(
                no_check_set=["Moment2MaxOut"], place=self.place, atol=1e-5, rtol=1e-4
            )

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestAdamOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestAdamOpFp16Case


create_test_fp16_class(TestAdam)
create_test_fp16_class(TestAdamWithEpsilonTensor)
create_test_fp16_class(TestAdamOpWithSkipUpdate)
create_test_fp16_class(TestAdamOpWithGlobalBetaPow)


class TestNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype("float32")
        b_np = np.random.random(size=(32, 32)).astype("float32")
        label_np = np.random.randint(2, size=(32, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype="float32")
            b = paddle.static.data(name="b", shape=[32, 32], dtype="float32")
            label = paddle.static.data(name="label", shape=[32, 1], dtype="int64")

            sum = paddle.add(a, b)
            z = paddle.pow(sum, 2.0)

            fc_1 = paddle.static.nn.fc(x=z, size=128)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            adam = paddle.optimizer.Adam(learning_rate=0.01)
            adam.minimize(loss)

        if run_npu:
            place = paddle.CustomPlace("npu", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "label": label_np},
                fetch_list=[prediction, loss],
            )
            if epoch % 10 == 0:
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)
        self.assertTrue(np.allclose(npu_pred, cpu_pred, atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(npu_loss, cpu_loss, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
