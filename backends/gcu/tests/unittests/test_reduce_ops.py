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

import paddle
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase


# The table retains its original format for better comparison of parameter settings.
# fmt: off
REDUCE_X_CASE = [
    # any
    {"reduce_api": paddle.any, "x_shape": [2, 6], "x_dtype": bool, "axis": None, "keepdim": False},
    {"reduce_api": paddle.any, "x_shape": [2, 6], "x_dtype": bool, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.any, "x_shape": [2, 6], "x_dtype": bool, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.any, "x_shape": [2, 6], "x_dtype": bool, "axis": None, "keepdim": True},
    {"reduce_api": paddle.any, "x_shape": [2, 6], "x_dtype": bool, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.any, "x_shape": [2, 6], "x_dtype": bool, "axis": -1, "keepdim": True},

    # mean
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.mean, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

    # sum
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.sum, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

    # max
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.max, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

    # min
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.min, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

    # prod
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.prod, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

    # amax
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.amax, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

    # amin
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float32, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float32, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float32, "axis": [1, 2], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.int32, "axis": [1, 2], "keepdim": False},

    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float16, "axis": 0, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 6], "x_dtype": np.float16, "axis": -1, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": False},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": None, "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 1], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [0, 2], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [-2, -1], "keepdim": True},
    {"reduce_api": paddle.amin, "x_shape": [2, 3, 4], "x_dtype": np.float16, "axis": [1, 2], "keepdim": True},

]
# fmt: on


@ddt
class TestReduceCommon(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.reduce_api = paddle.any
        self.x_shape = [2, 6]
        self.x_dtype = np.float32
        self.axis = None
        self.keepdim = False

    def prepare_data(self):
        if self.x_shape != np.int32:
            self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        else:
            self.data_x = self.generate_integer_data(
                self.x_shape, self.x_dtype, -60, 60
            )

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        if self.reduce_api in [paddle.sum]:
            return self.reduce_api(
                x, axis=self.axis, dtype=self.x_dtype, keepdim=self.keepdim
            )
        else:
            return self.reduce_api(x, axis=self.axis, keepdim=self.keepdim)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            origin_dtype = self.x_dtype
            self.x_dtype = np.float32
            out = self.calc_result(self.forward, "cpu")
            out = out.astype("float16")
            self.x_dtype = origin_dtype
        return out

    @data(*REDUCE_X_CASE)
    @unpack
    def test_check_output(self, reduce_api, x_shape, x_dtype, axis, keepdim):
        self.reduce_api = reduce_api
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.axis = axis
        self.keepdim = keepdim
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
