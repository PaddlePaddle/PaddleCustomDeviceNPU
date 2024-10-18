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

import os
import numpy as np
import unittest
import paddle

paddle.enable_static()


@paddle.incubate.passes.ir.RegisterPass
def generate_add_n():
    def pattern(x, y, z):
        return paddle.add(paddle.add(x, y), z)

    def replace(x, y, z):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        res = paddle.incubate.passes.ir.PassDesc.OP.my_add_n(X=x, Y=y, Z=z)
        res.SetAttr("axis", 0)
        return res

    return pattern, replace


@paddle.jit.to_static(
    input_spec=[
        paddle.static.InputSpec([None, 32], "float32", "x"),
        paddle.static.InputSpec([None, 32], "float32", "y"),
        paddle.static.InputSpec([None, 32], "float32", "z"),
    ]
)
def func(x, y, z):
    return x + y + z


MODEL_FILE = "./saved_model"


class TestCustomPass(unittest.TestCase):
    def setUp(self):
        print("TestCustomPass setUp:", os.getenv("CUSTOM_DEVICE_ROOT"))
        for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
            if lib.endswith(".so"):
                paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                    lib
                )
        with paddle.pir_utils.OldIrGuard():
            paddle.jit.save(func, MODEL_FILE)

    def test_my_add_n(self):
        config = paddle.inference.Config()
        config.set_prog_file(MODEL_FILE + ".pdmodel")
        config.enable_memory_optim()
        config.enable_custom_device("gcu")
        config.set_optim_cache_dir("./optim_cache")
        pass_builder = config.pass_builder()
        pass_builder.append_pass("generate_add_n")
        print(pass_builder.all_passes())
        pass_builder.turn_on_debug()
        predictor = paddle.inference.create_predictor(config)

        np_inputs = [
            np.random.randn(2, 32).astype("float32"),
            np.random.randn(2, 32).astype("float32"),
            np.random.randn(2, 32).astype("float32"),
        ]
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(np_inputs[i])

        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        np.testing.assert_allclose(results[0], np.sum(np_inputs, axis=0), rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
