#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
from scipy import special
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


def np_gelu(x, approximate=False):
    if approximate:
        y = (
            0.5
            * x
            * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        )
    else:
        y = 0.5 * x * (1 + special.erf(x / np.sqrt(2)))
    return y


class TestGelu(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gelu"
        self.python_api = paddle.nn.functional.gelu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np_gelu(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-8)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.005,
        )


class TestGeluApproximate(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gelu"
        self.python_api = paddle.nn.functional.gelu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np_gelu(x, True)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {"approximate": True}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-8)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.005,
        )


class TestGeluFp16(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gelu"
        self.python_api = paddle.nn.functional.gelu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np_gelu(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestGeluFp16Approximate(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "gelu"
        self.python_api = paddle.nn.functional.gelu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np_gelu(x, True)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {"approximate": True}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        # The atol value is modified because the precision of the
        # DNN library is insufficient
        self.check_output_with_place(self.place, atol=1e-2)


class TestGeluNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
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

            c = paddle.multiply(a, b)

            fc_1 = paddle.static.nn.fc(x=c, size=128)
            fc_1_gelu = paddle.nn.functional.gelu(fc_1)
            prediction = paddle.static.nn.fc(x=fc_1_gelu, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.Momentum(learning_rate=0.01)
            sgd.minimize(loss)

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

        return pred_res, loss_res

    def test_sdaa(self):
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred, atol=1e-3))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
