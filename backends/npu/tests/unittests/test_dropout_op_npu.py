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

import unittest

import unittest

import numpy as np
import paddle.base as base
from tests.op_test import OpTest, skip_check_grad_ci

import paddle
from tests.op_test import OpTest, skip_check_grad_ci

paddle.enable_static()

SEED = 2021
EPOCH = 100


def convert_to_npu_mask(mask):
    mask = mask.reshape([-1])
    n = int((mask.size + 127) // 128 * 128 / 8)
    npu_mask = np.ones([n], dtype=np.uint8) * 255
    for i in range(mask.size):
        if not mask[i]:
            npu_mask[i // 8] = npu_mask[i // 8] & (~(1 << (7 - i % 8)))
    return npu_mask


class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((32, 64)).astype("uint8")),
        }

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestDropoutOpModeDown(TestDropoutOp):
    # change input shape
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 62)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "downgrade_in_infer",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((32, 62)).astype("uint8")),
        }


class TestDropoutOpInput1d(TestDropoutOp):
    # change input shape
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((3, 62)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((3, 62)).astype("uint8")),
        }


class TestDropoutOpInput1d_1(TestDropoutOp):
    # the input is 1-D
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((2000)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((2000)).astype("uint8")),
        }


class TestDropoutModeDownOp1(TestDropoutOp):
    # the dropout_prob is 1.0
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 1.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "downgrade_in_infer",
        }
        self.outputs = {
            "Out": np.zeros((32, 64)).astype("float32"),
            "Mask": convert_to_npu_mask(np.zeros((32, 64)).astype("uint8")),
        }


class TestDropoutModeDownOp1Fp16(TestDropoutOp):
    # the dropout_prob is 0.2
    def init_dtype(self):
        self.dtype = np.float16

    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.2,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "downgrade_in_infer",
        }
        self.outputs = {
            "Out": np.zeros((32, 64)).astype("float16"),
            "Mask": convert_to_npu_mask(np.zeros((32, 64)).astype("uint8")),
        }


class TestDropoutOp2(TestDropoutOp):
    # the dropout_prob is 1.0
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 1.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": np.zeros((32, 64)).astype("float32"),
            "Mask": convert_to_npu_mask(np.zeros((32, 64)).astype("uint8")),
        }


class TestDropoutOp3(TestDropoutOp):
    # the input dim is 3
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64, 2)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.0,
            "fix_seed": True,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((32, 64, 2)).astype("uint8")),
        }


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInference(OpTest):
    # is_test = True
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.35,
            "fix_seed": True,
            "is_test": True,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {"Out": self.inputs["X"]}

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInference2(TestDropoutOpInference):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64, 3)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.75,
            "is_test": True,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {"Out": self.inputs["X"]}


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOpInferenceDownscale(TestDropoutOpInference):
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {"X": np.random.random((32, 64, 3)).astype(self.dtype)}
        self.attrs = {
            "dropout_prob": 0.75,
            "is_test": True,
            "dropout_implementation": "downgrade_in_infer",
        }
        self.outputs = {"Out": self.inputs["X"] * (1.0 - self.attrs["dropout_prob"])}


class TestDropoutOpWithSeed(TestDropoutOp):
    # the seed is a Tensor
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {
            "X": np.random.random((32, 64)).astype(self.dtype),
            "Seed": np.asarray([125], dtype="int32"),
        }
        self.attrs = {
            "dropout_prob": 0.0,
            "is_test": False,
            "dropout_implementation": "upscale_in_train",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((32, 64)).astype("uint8")),
        }


class TestDropoutOpModeDownWithSeed(TestDropoutOp):
    # the seed is a Tensor
    def setUp(self):
        self.op_type = "dropout"
        self.set_npu()
        self.init_dtype()
        self.inputs = {
            "X": np.random.random((32, 64)).astype(self.dtype),
            "Seed": np.asarray([125], dtype="int32"),
        }
        self.attrs = {
            "dropout_prob": 0.0,
            "is_test": False,
            "dropout_implementation": "downgrade_in_infer",
        }
        self.outputs = {
            "Out": self.inputs["X"],
            "Mask": convert_to_npu_mask(np.ones((32, 64)).astype("uint8")),
        }


class TestDropoutOpFp16(TestDropoutOp):
    # float16
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("npu", 0)


class TestDropoutOpFp64(TestDropoutOp):
    # float64
    def init_dtype(self):
        self.dtype = np.double

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("npu", 0)


_list = [
    paddle.bitwise_and,
    paddle.bitwise_or,
    paddle.bitwise_xor,
]


class TestDropoutAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [base.CPUPlace(), paddle.CustomPlace("npu", 0)]

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name="input", shape=[40, 40], dtype="float32")
            res1 = paddle.nn.functional.dropout(
                x=input, p=0.0, training=False, mode="upscale_in_train"
            )
            res2 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=True, mode="upscale_in_train"
            )
            res3 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=False, mode="upscale_in_train"
            )
            res4 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=[0, 1], training=True, mode="upscale_in_train"
            )
            res5 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=[0, 1], training=False, mode="upscale_in_train"
            )
            res6 = paddle.nn.functional.dropout(
                x=input, p=1.0, training=True, mode="upscale_in_train"
            )
            res7 = paddle.nn.functional.dropout(
                x=input, p=0.0, training=True, mode="upscale_in_train"
            )
            res8 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=(0, 1), training=False, mode="upscale_in_train"
            )
            res9 = paddle.nn.functional.dropout(
                x=input, p=0.0, training=False, mode="downgrade_in_infer"
            )
            res10 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=True, mode="downgrade_in_infer"
            )
            res11 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=False, mode="downgrade_in_infer"
            )
            res12 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=[0, 1], training=True, mode="downgrade_in_infer"
            )
            res13 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=[0, 1], training=False, mode="downgrade_in_infer"
            )
            res14 = paddle.nn.functional.dropout(
                x=input, p=1.0, training=True, mode="downscale_in_infer"
            )
            res15 = paddle.nn.functional.dropout(
                x=input, p=0.0, training=True, mode="downgrade_in_infer"
            )
            res16 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=(0, 1), training=False, mode="downgrade_in_infer"
            )
            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np2 = np.zeros_like(in_np)

            exe = base.Executor(place)
            res_list = [
                res1,
                res2,
                res3,
                res4,
                res5,
                res7,
                res8,
                res9,
                res10,
                res11,
                res12,
                res13,
                res15,
                res16,
            ]
            for res in res_list:
                fetches = exe.run(
                    base.default_main_program(),
                    feed={"input": in_np},
                    fetch_list=[res],
                )
                self.assertTrue(np.allclose(fetches[0], res_np))
            fetches2 = exe.run(
                base.default_main_program(), feed={"input": in_np}, fetch_list=[res6]
            )
            self.assertTrue(np.allclose(fetches2[0], res_np2))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)


if __name__ == "__main__":
    unittest.main()
