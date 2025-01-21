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

import os
import numpy as np
import unittest
import paddle
from paddle_custom_device.gcu import passes as gcu_passes

paddle.enable_static()

MODEL_DIR = "./model_graph"
MODEL_FILE = "test_graph"


class TestCustomEngine(unittest.TestCase):
    def setUp(self):
        print("TestCustomEngine setUp:", os.getenv("CUSTOM_DEVICE_ROOT"))
        gcu_passes.setUp()
        paddle.seed(2066)
        np.random.seed(2099)

    def test_custom_engine(self):
        config = paddle.inference.Config()
        config.set_model(MODEL_DIR, MODEL_FILE)
        config.enable_memory_optim()
        config.enable_custom_device("gcu")

        config.enable_new_ir(True)
        config.enable_new_executor(True)

        kPirGcuPasses = [
            "common_subexpression_elimination_pass",
            "constant_folding_pass",
            "gcu_op_marker_pass",
            "gcu_sub_graph_extract_pass",
            "gcu_replace_with_engine_op_pass",
        ]
        config.enable_custom_passes(kPirGcuPasses, True)

        predictor = paddle.inference.create_predictor(config)

        all_results = []
        for i in range(10):
            print("Run TestCustomEngine {}".format(i), flush=True)
            np_inputs = [
                np.random.random([16, 784]).astype("float32"),  # image
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
            # np.testing.assert_allclose(results[0], np.sum(np_inputs, axis=0), rtol=1e-2)
            all_results.append(results)
        # print(all_results)


if __name__ == "__main__":
    unittest.main()
