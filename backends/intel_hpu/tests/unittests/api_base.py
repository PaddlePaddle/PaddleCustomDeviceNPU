# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import unittest


class TestAPIBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.seed(2066)
        np.random.seed(2099)
        paddle.disable_static()

    def generate_bool_data(self, shape):
        return np.random.randint(2, size=shape) == 1

    def generate_float_data(self, shape, dtype):
        return np.random.random(shape).astype(dtype)

    def generate_integer_data(self, shape, dtype, lownum=0, highnum=256):
        return np.random.randint(lownum, high=highnum, size=shape, dtype=dtype)

    def generate_data(self, shape, dtype):
        if dtype == np.float32 or dtype == np.float16 or dtype == np.float64:
            data = self.generate_float_data(shape, dtype)
        elif dtype == bool:
            data = self.generate_bool_data(shape)
        elif dtype == np.int8:
            data = self.generate_integer_data(shape, dtype, -128, 127)
        elif dtype == np.int16:
            data = self.generate_integer_data(shape, dtype, -32768, 32767)
        elif dtype == np.int32:
            data = self.generate_integer_data(shape, dtype, -2147483648, 2147483647)
        elif dtype == np.int64:
            data = self.generate_integer_data(shape, dtype, -2147483648, 2147483647)
        elif dtype == np.uint8:
            data = self.generate_integer_data(shape, dtype, 0, 256)
        elif dtype == np.uint16:
            data = self.generate_integer_data(shape, dtype, 0, 65535)
        elif dtype == np.uint32:
            data = self.generate_integer_data(shape, dtype, 0, 4294967295)
        elif dtype == np.uint64:
            data = self.generate_integer_data(shape, dtype, 0, 4294967295)
        else:
            raise Exception("Unsupportted datatype!")
        return data

    def calc_result(self, calculator, device):
        paddle.set_device(device)
        return calculator()

    def check_value(self, hpu_val, cpu_val, rtol, atol, label="output"):
        self.assertTrue(
            np.allclose(hpu_val, cpu_val, rtol, atol),
            f"Compare hpu {label} with cpu failed.\n  hpu_{label}:{hpu_val}\n  cpu_{label}:{cpu_val}",
        )

    def check_multi_values(self, hpu_vals, cpu_vals, rtol, atol, label="output"):
        self.assertTrue(
            len(hpu_vals) == len(cpu_vals),
            f"Compare hpu {label} with hpu failed.\n  hpu_{label}s:{hpu_vals}\n  cpu_{label}s:{cpu_vals}",
        )

        for hpu_val, cpu_val in zip(hpu_vals, cpu_vals):
            self.assertTrue(
                np.allclose(hpu_val, cpu_val, rtol, atol),
                f"Compare hpu {label} with cpu failed.\n  hpu_{label}s:{hpu_vals}\n  cpu_{label}s:{cpu_vals}",
            )

    def check_output_hpu_with_cpu(self, forward, rtol=1e-5, atol=1e-5):
        self.prepare_data()
        hpu_out = self.calc_result(forward, "intel_hpu")
        cpu_out = self.calc_result(forward, "cpu")
        # print(f"hpu_out:{hpu_out}")
        # print(f"cpu_out:{cpu_out}")
        if isinstance(hpu_out, (list, tuple)):
            self.check_multi_values(hpu_out, cpu_out, rtol, atol)
        else:
            self.check_value(hpu_out, cpu_out, rtol, atol)

    def check_output_hpu_with_customized(
        self, forward, customized, rtol=1e-5, atol=1e-5
    ):
        self.prepare_data()
        hpu_out = self.calc_result(forward, "intel_hpu")
        cpu_out = self.calc_result(customized, "cpu")
        # print(f"hpu_out:{hpu_out}")
        # print(f"cpu_out:{cpu_out}")
        if isinstance(hpu_out, (list, tuple)):
            self.check_multi_values(hpu_out, cpu_out, rtol, atol)
        else:
            self.check_value(hpu_out, cpu_out, rtol, atol)

    def check_grad_hpu_with_cpu(self, backward, rtol=1e-5, atol=1e-5):
        self.prepare_data()
        hpu_grad = self.calc_result(backward, "intel_hpu")
        cpu_grad = self.calc_result(backward, "cpu")
        # print(f"hpu_grad:{hpu_grad}")
        # print(f"cpu_grad:{cpu_grad}")
        if isinstance(hpu_grad, (list, tuple)):
            self.check_multi_values(hpu_grad, cpu_grad, rtol, atol, "grad")
        else:
            self.check_value(hpu_grad, cpu_grad, rtol, atol, "grad")

    def check_grad_hpu_with_customized(
        self, backward, customized, rtol=1e-5, atol=1e-5
    ):
        self.prepare_data()
        hpu_grad = self.calc_result(backward, "intel_hpu")
        cpu_grad = self.calc_result(customized, "cpu")
        # print(f"hpu_grad:{hpu_grad}")
        # print(f"cpu_grad:{cpu_grad}")
        if isinstance(hpu_grad, (list, tuple)):
            self.check_multi_values(hpu_grad, cpu_grad, rtol, atol, "grad")
        else:
            self.check_value(hpu_grad, cpu_grad, rtol, atol, "grad")
