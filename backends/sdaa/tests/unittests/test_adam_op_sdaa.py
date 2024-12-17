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

import numpy as np
import unittest
from op_test import OpTest
import paddle
import paddle.base as base

paddle.enable_static()
SEED = 2022


def adam_wrapper(
    param,
    grad,
    LearningRate,
    moment1,
    moment2,
    beta1_pow,
    beta2_pow,
    master_weight=None,
    find_inf=None,
    beta1=0.78,
    beta2=0.836,
    epsilon=1e-4,
    lazy_mode=False,
):
    _, _, _, _, _, *_ = paddle._C_ops.adam_(
        param,
        grad,
        LearningRate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        find_inf,
        beta1,
        beta2,
        epsilon,
        lazy_mode,
        1000,
        False,
        False,
    )


def adamw_wrapper(
    param,
    grad,
    lr,
    moment1,
    moment2,
    beta1_pow,
    beta2_pow,
    master_weight=None,
    found_inf=None,
    beta1=0.78,
    beta2=0.836,
    epsilon=1e-4,
    lr_ratio=1.0,
    weight_decay=0.01,
    with_decay=True,
    lazy_mode=False,
):
    _, _, _, _, _, _ = paddle._C_ops.adamw_(
        param,
        grad,
        lr,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        found_inf,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        weight_decay,
        with_decay,
        lazy_mode,
        1000,
        False,
        False,
    )


def adam_step(inputs, attributes):
    param = inputs["Param"]
    grad = inputs["Grad"]
    moment1 = inputs["Moment1"]
    moment2 = inputs["Moment2"]
    lr = inputs["LearningRate"]
    beta1_pow = inputs["Beta1Pow"]
    beta2_pow = inputs["Beta2Pow"]

    epsilon = attributes["epsilon"]

    if "lr_ratio" in attributes:
        lr = lr * attributes["lr_ratio"]

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
    denom = (np.sqrt(moment2_out) / np.sqrt(1.0 - beta2_pow)) + epsilon
    param_out = param + ((moment1_out / denom) * (-(lr / (1.0 - beta1_pow))))
    return param_out, moment1_out, moment2_out


def adamw_step(inputs, attributes):
    param = inputs["Param"]
    grad = inputs["Grad"]
    moment1 = inputs["Moment1"]
    moment2 = inputs["Moment2"]
    lr = inputs["LearningRate"]
    beta1_pow = inputs["Beta1Pow"]
    beta2_pow = inputs["Beta2Pow"]

    epsilon = attributes["epsilon"]

    if "lr_ratio" in attributes:
        lr = lr * attributes["lr_ratio"]

    if attributes["with_decay"]:
        coeff = attributes["coeff"]
        decay = 1.0 - lr * coeff
        param2 = param * decay
        param = param2.copy()

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
    denom = (np.sqrt(moment2_out) / np.sqrt(1.0 - beta2_pow)) + epsilon
    param_out = param + ((moment1_out / denom) * (-(lr / (1.0 - beta1_pow))))
    return param_out, moment1_out, moment2_out


class TestAdamW(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "adamw"
        self.python_api = adamw_wrapper
        self.python_out_sig = ["Out"]
        param = np.random.uniform(-1, 1, (105, 102)).astype("float32")
        grad = np.random.uniform(-1, 1, (105, 102)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (105, 102)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((105, 102)).astype("float32")

        learning_rate = 0.5
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
            "LearningRate": np.array([learning_rate]).astype("float32"),
            "Beta1Pow": np.array([beta1_pow]).astype("float32"),
            "Beta2Pow": np.array([beta2_pow]).astype("float32"),
        }

        self.attrs = {
            "epsilon": epsilon,
            "beta1": beta1,
            "beta2": beta2,
            "coeff": 0.9,
            "with_decay": True,
        }

        param_out, moment1_out, moment2_out = adamw_step(self.inputs, self.attrs)

        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([beta1_pow]).astype("float32") * beta1,
            "Beta2PowOut": np.array([beta2_pow]).astype("float32") * beta2,
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestAdamOpWithSkipUpdate(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "adamw"
        self.python_api = adamw_wrapper
        self.python_out_sig = ["Out"]
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

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
            "LearningRate": np.array([learning_rate]).astype("float32"),
            "Beta1Pow": np.array([beta1_pow]).astype("float32"),
            "Beta2Pow": np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        self.attrs = {"epsilon": epsilon, "coeff": 0.02, "with_decay": True}

        self.outputs = {
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "ParamOut": param,
            "Beta1PowOut": self.inputs["Beta1Pow"],
            "Beta2PowOut": self.inputs["Beta2Pow"],
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestAdamOpWithoutDecay(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "adamw"
        self.python_api = adamw_wrapper
        self.python_out_sig = ["Out"]
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

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
            "LearningRate": np.array([learning_rate]).astype("float32"),
            "Beta1Pow": np.array([beta1_pow]).astype("float32"),
            "Beta2Pow": np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        self.attrs = {"epsilon": epsilon, "coeff": 0.02, "with_decay": False}

        self.outputs = {
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "ParamOut": param,
            "Beta1PowOut": self.inputs["Beta1Pow"],
            "Beta2PowOut": self.inputs["Beta2Pow"],
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestadamwNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        paddle.enable_static()
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
            adam = paddle.optimizer.AdamW(learning_rate=0.01, weight_decay=0.02)
            adam.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
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
        paddle.disable_static()
        return pred_res, loss_res

    def test_sdaa(self):
        sdaa_pred, sdaa_loss = self._test(True)
        cpu_pred, cpu_loss = self._test(False)
        self.assertTrue(np.allclose(sdaa_pred, cpu_pred, rtol=1e-3))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, rtol=1e-3))


class TestAdamOpWithSkipUpdate(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ["Out"]
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

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
            "LearningRate": np.array([learning_rate]).astype("float32"),
            "Beta1Pow": np.array([beta1_pow]).astype("float32"),
            "Beta2Pow": np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        self.attrs = {"epsilon": epsilon}

        self.outputs = {
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "ParamOut": param,
            "Beta1PowOut": self.inputs["Beta1Pow"],
            "Beta2PowOut": self.inputs["Beta2Pow"],
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestAdamOpWithGlobalBetaPow(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ["Out"]
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

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
            "LearningRate": np.array([learning_rate]).astype("float32"),
            "Beta1Pow": np.array([beta1_pow]).astype("float32"),
            "Beta2Pow": np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
        }

        attributes = {"epsilon": epsilon}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, attributes)

        self.attrs = {"use_global_beta_pow": True}

        # use_global_beta_pow=True, Beta1PowOut and Beta2PowOut are empty.
        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([]),
            "Beta2PowOut": np.array([]),
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


@unittest.skip("AMP is only supported on dygraph")
class TestAdamOpWithGlobalBetaPow_fp16(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "adam"
        self.python_api = adam_wrapper
        self.python_out_sig = ["Out"]
        param = np.random.uniform(-1, 1, (102, 105)).astype("float16")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float16")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float16")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float16")

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
            "LearningRate": np.array([learning_rate]).astype("float32"),
            "Beta1Pow": np.array([beta1_pow]).astype("float32"),
            "Beta2Pow": np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
            "EpsilonTensor": np.array([epsilon]).astype("float32"),
        }

        attributes = {"epsilon": epsilon}

        param_out, moment1_out, moment2_out = adam_step(self.inputs, attributes)

        self.attrs = {"use_global_beta_pow": True}

        # use_global_beta_pow=True, Beta1PowOut and Beta2PowOut are empty.
        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([]),
            "Beta2PowOut": np.array([]),
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestadamNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        paddle.enable_static()
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

            fc_1 = paddle.static.nn.fc(x=z, size=256)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

            adam = paddle.optimizer.Adam(
                learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, grad_clip=clip
            )
            adam.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
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
        paddle.disable_static()
        return pred_res, loss_res

    def test_sdaa(self):
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred, rtol=1e-2))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, rtol=1e-2))


class TestNetWithEpsilonTensor(unittest.TestCase):
    def _test(self, place, use_tensor=True):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 2021
        paddle.seed(SEED)
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 2)).astype("float32")
        b_np = np.random.random(size=(2, 2)).astype("float32")
        label_np = np.random.randint(2, size=(2, 1)).astype("int64")
        weight_attr1 = paddle.ParamAttr(
            name="weight1",
            initializer=paddle.nn.initializer.Constant(value=1.0),
            trainable=True,
        )
        weight_attr2 = paddle.ParamAttr(
            name="weight2",
            initializer=paddle.nn.initializer.Constant(value=2.0),
            trainable=True,
        )
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

        with paddle.static.program_guard(main_prog, startup_prog):
            with paddle.utils.unique_name.guard():

                a = paddle.static.data(name="a", shape=[2, 2], dtype="float32")
                b = paddle.static.data(name="b", shape=[2, 2], dtype="float32")
                label = paddle.static.data(name="label", shape=[2, 1], dtype="int64")

                sum = paddle.add(a, b)
                z = paddle.pow(sum, 2.0)

                fc_1 = paddle.static.nn.fc(x=z, size=2, weight_attr=weight_attr1)
                prediction = paddle.static.nn.fc(
                    x=fc_1, size=2, weight_attr=weight_attr2, activation="softmax"
                )

                cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
                loss = paddle.mean(cost)
                beta1_init = 0.9
                beta2_init = 0.999
                epsilon_init = 1e-8
                if use_tensor:
                    beta1 = paddle.static.create_global_var(
                        shape=[1],
                        value=float(beta1_init),
                        dtype="float32",
                        persistable=True,
                        name="beta1",
                    )
                    beta2 = paddle.static.create_global_var(
                        shape=[1],
                        value=float(beta2_init),
                        dtype="float32",
                        persistable=True,
                        name="beta2",
                    )
                    epsilon = paddle.static.create_global_var(
                        shape=[1],
                        value=float(epsilon_init),
                        dtype="float32",
                        persistable=True,
                        name="epsilon",
                    )

                    adam = paddle.optimizer.Adam(
                        learning_rate=0.01,
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=epsilon,
                        grad_clip=clip,
                    )
                else:
                    adam = paddle.optimizer.Adam(
                        learning_rate=0.01,
                        beta1=beta1_init,
                        beta2=beta2_init,
                        epsilon=epsilon_init,
                        grad_clip=clip,
                    )

                adam.minimize(loss)

        scope = base.Scope()
        with base.scope_guard(scope):
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            print("Start run on {}".format(place))
            print("use_tensor " + str(use_tensor))
            for epoch in range(10):
                pred_res, loss_res = exe.run(
                    main_prog,
                    feed={"a": a_np, "b": b_np, "label": label_np},
                    fetch_list=[prediction, loss],
                )
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )
            paddle.disable_static()
            return pred_res, loss_res

    def _test_with_place(self, place):
        preds = []
        losses = []

        for use_tensor in [True, False]:
            pred, loss = self._test(place, use_tensor)
            preds.append(pred)
            losses.append(loss)
        for pred in preds:
            self.assertTrue(np.allclose(pred, preds[0]))
        for loss in losses:
            self.assertTrue(np.allclose(loss, losses[0]))

    def test_adam_api(self):
        # NOTE(zhiqiu): cpu and gpu has different seed, so should compare separatly.
        self._test_with_place(paddle.CustomPlace("sdaa", 0))


if __name__ == "__main__":
    unittest.main()
