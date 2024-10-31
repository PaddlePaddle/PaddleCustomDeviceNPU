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
import paddle_custom_device.gcu.passes as passes

paddle.enable_static()

MODEL_FILE = "./model/conv_bn"


class TestCustomPass(unittest.TestCase):
    def setUp(self):
        print("TestCustomPass setUp:", os.getenv("CUSTOM_DEVICE_ROOT"))
        passes.setUp()

    def test_gcu_conv_bn_relu(self):
        config = paddle.inference.Config(
            MODEL_FILE + ".pdmodel", MODEL_FILE + ".pdiparams"
        )
        # config = Config(args.model_file, args.params_file)
        # config.set_prog_file(MODEL_FILE + ".pdmodel")
        config.enable_memory_optim()
        config.enable_custom_device("gcu")
        config.enable_new_ir(False)
        config.enable_new_executor(False)
        kPirGcuPasses = []
        config.enable_custom_passes(kPirGcuPasses, True)
        config.set_optim_cache_dir("./optim_cache")
        pass_builder = config.pass_builder()
        # pass_builder.append_pass("gcu_fuse_conv_bn")
        # logging.info(pass_builder.all_passes())
        pass_builder.turn_on_debug()
        predictor = paddle.inference.create_predictor(config)

        np_inputs = [
            np.random.randn(4, 3, 224, 224).astype("float32"),
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


if __name__ == "__main__":
    unittest.main()
