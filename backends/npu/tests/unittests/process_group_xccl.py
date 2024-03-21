# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
import paddle_custom_device
from npu_utils import check_soc_version_and_dtype


def init_process_group(strategy=None):
    nranks = paddle.distributed.ParallelEnv().nranks
    rank = dist.ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    pg_group = dist.init_parallel_env()

    return pg_group.process_group, pg_group


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)

    @check_soc_version_and_dtype
    def test_create_process_group_xccl(self):
        device_id = paddle.distributed.ParallelEnv().dev_id
        paddle.set_device("npu:%d" % device_id)

        assert paddle.distributed.is_available()

        pg, pg_group = init_process_group()
        print("rank:", pg.rank(), "size:", pg.size(), "name:", pg.name())
        print("test new group api ok")

        # test allreduce sum
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = dist.all_reduce(tensor_x)
            assert np.array_equal(tensor_x, sum_result)
        else:
            task = dist.all_reduce(tensor_y)
            assert np.array_equal(tensor_y, sum_result)

        print("test allreduce sum api ok")

        # test allreduce sum with shape = []
        # rank 0
        x = np.random.random([])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random([])
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = dist.all_reduce(tensor_x)
            assert np.array_equal(tensor_x, sum_result)
        else:
            task = dist.all_reduce(tensor_y)
            assert np.array_equal(tensor_y, sum_result)

        print("test allreduce sum api with = [] ok")

        # test allreduce max
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        max_result = paddle.maximum(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.all_reduce(tensor_x, dist.ReduceOp.MAX, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, max_result)
        else:
            task = dist.all_reduce(tensor_y, dist.ReduceOp.MAX, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_y, max_result)

        print("test allreduce max api ok")

        # test allreduce max with shape = []
        # rank 0
        x = np.random.random([])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random([])
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        max_result = paddle.maximum(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.all_reduce(tensor_x, dist.ReduceOp.MAX, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, max_result)
        else:
            task = dist.all_reduce(tensor_y, dist.ReduceOp.MAX, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_y, max_result)

        print("test allreduce max api with shape = [] ok")

        # test allreduce min
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        min_result = paddle.minimum(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.all_reduce(tensor_x, dist.ReduceOp.MIN, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, min_result)
        else:
            task = dist.all_reduce(tensor_y, dist.ReduceOp.MIN, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_y, min_result)

        print("test allreduce min api ok")

        # test allreduce min with shape = []
        # rank 0
        x = np.random.random([])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random([])
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        min_result = paddle.minimum(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.all_reduce(tensor_x, dist.ReduceOp.MIN, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, min_result)
        else:
            task = dist.all_reduce(tensor_y, dist.ReduceOp.MIN, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_y, min_result)

        print("test allreduce min api with shape [] ok")

        if self.dtype != "bfloat16":
            # test allreduce prod
            # rank 0
            x = np.random.random(self.shape)
            tensor_x = paddle.to_tensor(x, dtype=self.dtype)
            # rank 1
            y = np.random.random(self.shape)
            tensor_y = paddle.to_tensor(y, dtype=self.dtype)

            prod_result = paddle.multiply(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = dist.all_reduce(tensor_x, dist.ReduceOp.PROD, sync_op=False)
                task.wait()
                assert np.array_equal(tensor_x, prod_result)
            else:
                task = dist.all_reduce(tensor_y, dist.ReduceOp.PROD, sync_op=False)
                task.wait()
                assert np.array_equal(tensor_y, prod_result)

            print("test allreduce prod api ok")

            # test allreduce prod with shape = []
            # rank 0
            x = np.random.random([]).astype(self.dtype)
            tensor_x = paddle.to_tensor(x)
            # rank 1
            y = np.random.random([]).astype(self.dtype)
            tensor_y = paddle.to_tensor(y)

            prod_result = paddle.multiply(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = dist.all_reduce(tensor_x, dist.ReduceOp.PROD, sync_op=False)
                task.wait()
                assert np.array_equal(tensor_x, prod_result)
            else:
                task = dist.all_reduce(tensor_y, dist.ReduceOp.PROD, sync_op=False)
                task.wait()
                assert np.array_equal(tensor_y, prod_result)

            print("test allreduce prod api with shape = [] ok")

        # test broadcast
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        broadcast_result = paddle.assign(tensor_x)
        if pg.rank() == 0:
            task = dist.broadcast(tensor_x, 0, sync_op=False)
            task.synchronize()
            paddle.device.synchronize()
            assert task.is_completed()
            assert np.array_equal(broadcast_result, tensor_x)
        else:
            task = dist.broadcast(tensor_y, 0)
            paddle.device.synchronize()
            assert np.array_equal(broadcast_result, tensor_y)

        print("test broadcast api ok")

        # test broadcast with shape=[]
        # rank 0
        x = np.random.random([])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random([])
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        broadcast_result = paddle.assign(tensor_x)
        if pg.rank() == 0:
            task = dist.broadcast(tensor_x, 0, sync_op=False)
            task.synchronize()
            paddle.device.synchronize()
            assert task.is_completed()
            assert np.array_equal(broadcast_result, tensor_x)
        else:
            task = dist.broadcast(tensor_y, 0)
            paddle.device.synchronize()
            assert np.array_equal(broadcast_result, tensor_y)
        assert tensor_y.shape == []

        print("test broadcast api with shape=[] ok")

        # test barrier
        # rank 0
        if pg.rank() == 0:
            pg.barrier(device_id)
        # rank 1
        else:
            task = pg.barrier(device_id)
            task.wait()

        print("test barrier api ok\n")

        # test allgather
        # rank 0
        x = np.random.random(self.shape)
        y = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        out_shape = list(self.shape)
        out_shape[0] *= 2
        out = np.random.random(out_shape)
        tensor_out = paddle.to_tensor(out, dtype=self.dtype)
        if pg.rank() == 0:
            task = pg.all_gather(tensor_x, tensor_out)
            task.wait()
            paddle.device.synchronize()
        # rank 1
        else:
            tensor_out_list = [
                paddle.empty_like(tensor_x),
                paddle.empty_like(tensor_x),
            ]
            task = dist.all_gather(tensor_out_list, tensor_y, sync_op=False)
            paddle.device.synchronize()
            tensor_out = paddle.concat(tensor_out_list)
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
        assert np.array_equal(tensor_x, out_1)
        assert np.array_equal(tensor_y, out_2)
        print("test allgather api ok\n")

        if pg.rank() == 0:
            task = pg.all_gather(tensor_x, tensor_out)
            task.wait()
            paddle.device.synchronize()
        # rank 1
        else:
            tensor_out_list = []
            task = dist.all_gather(tensor_out_list, tensor_y, sync_op=False)
            paddle.device.synchronize()
            tensor_out = paddle.concat(tensor_out_list)
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
        assert np.array_equal(tensor_x, out_1)
        assert np.array_equal(tensor_y, out_2)
        print("test allgather api2 ok\n")

        # test allgather with shape = []
        # rank 0
        x = np.random.random([])
        y = np.random.random([])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        tensor_out_list = []
        if pg.rank() == 0:
            task = dist.all_gather(tensor_out_list, tensor_x)
            task.wait()
            paddle.device.synchronize()
        # rank 1
        else:
            task = dist.all_gather(tensor_out_list, tensor_y, sync_op=False)
            paddle.device.synchronize()
        out_1 = tensor_out_list[0]
        out_2 = tensor_out_list[1]
        assert np.array_equal(tensor_x, out_1)
        assert np.array_equal(tensor_y, out_2)
        print("test allgather api with shape [] ok\n")

        # test alltoall
        # rank 0
        x = np.random.random(self.shape)
        y = np.random.random(self.shape)
        out1 = np.random.random(self.shape)
        out2 = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        tensor_out1 = paddle.to_tensor(out1, dtype=self.dtype)
        tensor_out2 = paddle.to_tensor(out2, dtype=self.dtype)
        raw_tensor_x_2 = paddle.slice(
            tensor_x, [0], [self.shape[0] // 2], [self.shape[0]]
        )
        raw_tensor_y_1 = paddle.slice(tensor_y, [0], [0], [self.shape[0] // 2])
        if pg.rank() == 0:
            task = pg.alltoall(tensor_x, tensor_out1)
            task.wait()
        # rank 1
        else:
            in_1, in_2 = paddle.split(tensor_y, 2)
            out_1, out_2 = paddle.split(tensor_out2, 2)
            out_tensor_list = [out_1, out_2]
            task = dist.alltoall([in_1, in_2], out_tensor_list)
            paddle.device.synchronize()
            tensor_out2 = paddle.concat(out_tensor_list)
        out1_2 = paddle.slice(tensor_out1, [0], [self.shape[0] // 2], [self.shape[0]])
        out2_1 = paddle.slice(tensor_out2, [0], [0], [self.shape[0] // 2])
        if pg.rank() == 0:
            assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
        else:
            assert np.array_equal(out2_1, raw_tensor_x_2)
        print("test alltoall api ok\n")

        x = np.random.random(self.shape)
        y = np.random.random(self.shape)
        out1 = np.random.random(self.shape)
        out2 = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        tensor_out1 = paddle.to_tensor(out1, dtype=self.dtype)
        tensor_out2 = paddle.to_tensor(out2, dtype=self.dtype)
        raw_tensor_x_2 = paddle.slice(
            tensor_x, [0], [self.shape[0] // 2], [self.shape[0]]
        )
        raw_tensor_y_1 = paddle.slice(tensor_y, [0], [0], [self.shape[0] // 2])
        if pg.rank() == 0:
            task = pg.alltoall(tensor_x, tensor_out1)
            task.wait()
        # rank 1
        else:
            in_1, in_2 = paddle.split(tensor_y, 2)
            out_1, out_2 = paddle.split(tensor_out2, 2)
            out_tensor_list = []
            task = dist.alltoall([in_1, in_2], out_tensor_list)
            paddle.device.synchronize()
            tensor_out2 = paddle.concat(out_tensor_list)
        out1_2 = paddle.slice(tensor_out1, [0], [self.shape[0] // 2], [self.shape[0]])
        out2_1 = paddle.slice(tensor_out2, [0], [0], [self.shape[0] // 2])
        if pg.rank() == 0:
            assert np.array_equal(out1_2.numpy(), raw_tensor_y_1.numpy())
        else:
            assert np.array_equal(out2_1, raw_tensor_x_2)
        print("test alltoall api2 ok\n")

        # test Reduce
        # rank 0
        x = np.random.random(self.shape)
        y = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = dist.reduce(tensor_x, 0, sync_op=True)
            paddle.device.synchronize()
        # rank 1
        else:
            task = dist.reduce(tensor_y, 0, sync_op=False)
            task.wait()
            paddle.device.synchronize()
        if pg.rank() == 0:
            assert np.array_equal(tensor_x, sum_result)
        print("test reduce sum api ok\n")

        # test reduce max
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        max_result = paddle.maximum(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.reduce(tensor_x, 0, dist.ReduceOp.MAX, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, max_result)
        else:
            task = dist.reduce(tensor_y, 0, dist.ReduceOp.MAX, sync_op=False)
            task.wait()

        print("test reduce max api ok")

        # test reduce min
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        min_result = paddle.minimum(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.reduce(tensor_x, 0, dist.ReduceOp.MIN, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, min_result)
        else:
            task = dist.reduce(tensor_y, 0, dist.ReduceOp.MIN, sync_op=False)
            task.wait()

        print("test reduce min api ok")

        if self.dtype != "bfloat16":
            # test reduce product
            # rank 0
            x = np.random.random(self.shape)
            tensor_x = paddle.to_tensor(x, dtype=self.dtype)
            # rank 1
            y = np.random.random(self.shape)
            tensor_y = paddle.to_tensor(y, dtype=self.dtype)

            prod_result = paddle.multiply(tensor_x, tensor_y)

            if pg.rank() == 0:
                task = dist.reduce(tensor_x, 0, dist.ReduceOp.PROD, sync_op=False)
                task.wait()
                assert np.array_equal(tensor_x, prod_result)
            else:
                task = dist.reduce(tensor_y, 0, dist.ReduceOp.PROD, sync_op=False)
                task.wait()

            print("test reduce prod api ok")

        test_reduce_with_zero_dim([], self.dtype, pg)

        # test Scatter
        # rank 0
        in_shape = list(self.shape)
        in_shape[0] *= 2
        x = np.random.random(in_shape)
        y = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        if pg.rank() == 0:
            in_1, in_2 = paddle.split(tensor_x, 2)
            task = dist.scatter(tensor_y, [in_1, in_2], 0, sync_op=True)
            # task.wait()
            paddle.device.synchronize()
        # rank 1
        else:
            task = dist.scatter(tensor_y, [], 0, sync_op=False)
            task.wait()
            paddle.device.synchronize()
        out1 = paddle.slice(tensor_x, [0], [0], [self.shape[0]])
        out2 = paddle.slice(tensor_x, [0], [self.shape[0]], [self.shape[0] * 2])
        if pg.rank() == 0:
            assert np.array_equal(tensor_y, out1)
        else:
            assert np.array_equal(tensor_y, out2)
        print("test scatter api ok\n")

        # test Scatter with shape=[]
        # rank 0
        x = np.random.random([])
        y = np.random.random([])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)
        if pg.rank() == 0:
            in_1, in_2 = tensor_x, tensor_x + 1
            task = dist.scatter(tensor_y, [in_1, in_2], 0, sync_op=True)
            paddle.device.synchronize()
        # rank 1
        else:
            task = dist.scatter(tensor_y, [], 0, sync_op=True)
            task.wait()
            paddle.device.synchronize()
        out1 = paddle.assign(tensor_x)
        out2 = paddle.assign(tensor_x + 1)
        if pg.rank() == 0:
            assert np.array_equal(tensor_y, out1)
        else:
            assert np.array_equal(tensor_y, out2), f"{tensor_y}, {out2}"
        assert tensor_y.shape == []
        print("test scatter api with shape=[] ok\n")

        # test send min
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        if pg.rank() == 0:
            task = dist.send(tensor_x, 1, sync_op=False)
            task.wait()
        else:
            task = dist.recv(tensor_y, 0, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_y, tensor_x)

        print("test send api ok")

        # test send min
        # rank 0
        x = np.random.random(self.shape)
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.random.random(self.shape)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        if pg.rank() == 0:
            task = dist.send(tensor_x, 1, sync_op=True)
        else:
            task = dist.recv(tensor_y, 0, sync_op=True)
            assert np.array_equal(tensor_y, tensor_x)

        print("test send api ok")

        # test send 0-d tensor
        # rank 0
        x = np.random.uniform(-1, 1, [])
        tensor_x = paddle.to_tensor(x, dtype=self.dtype)
        # rank 1
        y = np.array(0.2022)
        tensor_y = paddle.to_tensor(y, dtype=self.dtype)

        if pg.rank() == 0:
            task = dist.send(tensor_x, 1, sync_op=True)
        else:
            task = dist.recv(tensor_y, 0, sync_op=True)
            assert np.array_equal(tensor_y, tensor_x) and tensor_y.shape == []

        print("test send & recv 0-d tensor ok")

        if int(paddle_custom_device.npu.version()["cann"].split(".")[0]) >= 7:
            # test fused_allgather_mm
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            hcom_name = pg_group.process_group.get_comm_name(rank)
            x1 = np.random.uniform(-1, 1, [16, 512])
            x1 = paddle.to_tensor(x1).astype(self.dtype)
            x2 = np.random.uniform(-1, 1, [512, 256])
            x2 = paddle.to_tensor(x2).astype(self.dtype)
            bias = np.random.uniform(-1, 1, [x2.shape[1]])
            bias = None  # paddle.to_tensor(bias).astype(self.dtype)
            out, gather_out = paddle_custom_device.npu.fused_allgather_mm(
                x1,
                x2,
                bias=bias,
                hcom=hcom_name,
                world_size=world_size,
                gather_index=0,
                gather_output=True,
                comm_turn=0,
            )
            gather_res = []
            paddle.distributed.all_gather(gather_res, x1, group=pg_group)
            gather_res = paddle.concat(gather_res)
            out_res = paddle.matmul(gather_res, x2)
            if bias is not None:
                out_res += bias
            np.testing.assert_allclose(out.cpu().numpy(), out_res.cpu().numpy())
            np.testing.assert_allclose(
                gather_out.cpu().numpy(), gather_res.cpu().numpy()
            )

            # test fused_mm_reduce_scatter
            x1 = np.random.uniform(-1, 1, [128, 512])
            x1 = paddle.to_tensor(x1).astype(self.dtype)
            x2 = np.random.uniform(-1, 1, [512, 256])
            x2 = paddle.to_tensor(x2).astype(self.dtype)
            bias = np.random.uniform(-1, 1, [x2.shape[1]])
            bias = None  # paddle.to_tensor(bias).astype(self.dtype)
            out = paddle_custom_device.npu.fused_mm_reduce_scatter(
                x1,
                x2,
                bias=bias,
                hcom=hcom_name,
                world_size=world_size,
                reduce_op="sum",
                comm_turn=0,
            )
            mm = paddle.matmul(x1, x2)
            if bias is not None:
                mm += bias
            reduce_scatter_res = paddle.empty(
                [mm.shape[0] // world_size, mm.shape[1]], dtype=mm.dtype
            )
            paddle.distributed.reduce_scatter(reduce_scatter_res, [mm], group=pg_group)
            np.testing.assert_allclose(
                out.cpu().numpy(), reduce_scatter_res.cpu().numpy()
            )

            # test fused_mm_allreduce
            x1 = np.random.uniform(-1, 1, [128, 512]).astype(self.dtype)
            x1 = paddle.to_tensor(x1)
            x2 = np.random.uniform(-1, 1, [512, 256]).astype(self.dtype)
            x2 = paddle.to_tensor(x2)
            out = paddle_custom_device.npu.fused_mm_allreduce(
                x1, x2, bias=None, hcom=hcom_name, reduce_op="sum", comm_turn=0
            )
            mm = paddle.matmul(x1, x2)
            paddle.distributed.all_reduce(mm, group=pg_group, sync_op=True)
            np.testing.assert_allclose(out.cpu().numpy(), mm.cpu().numpy())
            print(
                "test fused_allgather_mm & fused_mm_reduce_scatter & fused_mm_allreduce ok!"
            )


class TestProcessGroupFp16(TestProcessGroupFp32):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float16"
        self.shape = (4, 20, 20)


class TestProcessGroupBF16(TestProcessGroupFp32):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "bfloat16"
        self.shape = (4, 20, 20)


def test_reduce_with_zero_dim(shape, dtype, pg):
    # test Reduce With Zero Dim
    # rank 0
    x = np.random.random(shape)
    y = np.random.random(shape)
    tensor_x = paddle.to_tensor(x, dtype=dtype)
    tensor_y = paddle.to_tensor(y, dtype=dtype)
    sum_result = tensor_x + tensor_y
    if pg.rank() == 0:
        task = dist.reduce(tensor_x, 0, sync_op=True)
        paddle.device.synchronize()
    # rank 1
    else:
        task = dist.reduce(tensor_y, 0, sync_op=False)
        task.wait()
        paddle.device.synchronize()
    if pg.rank() == 0:
        assert np.array_equal(tensor_x, sum_result) and len(tensor_x.shape) == 0
    print("test reduce with zero dim sum api ok\n")

    # test reduce with zero dim max
    # rank 0
    x = np.random.random(shape)
    tensor_x = paddle.to_tensor(x, dtype=dtype)
    # rank 1
    y = np.random.random(shape)
    tensor_y = paddle.to_tensor(y, dtype=dtype)

    max_result = paddle.maximum(tensor_x, tensor_y)

    if pg.rank() == 0:
        task = dist.reduce(tensor_x, 0, dist.ReduceOp.MAX, sync_op=False)
        task.wait()
        assert np.array_equal(tensor_x, max_result) and len(tensor_x.shape) == 0
    else:
        task = dist.reduce(tensor_y, 0, dist.ReduceOp.MAX, sync_op=False)
        task.wait()

    print("test reduce with zero dim max api ok")

    # test reduce with zero dim min
    # rank 0
    x = np.random.random(shape)
    tensor_x = paddle.to_tensor(x, dtype=dtype)
    # rank 1
    y = np.random.random(shape)
    tensor_y = paddle.to_tensor(y, dtype=dtype)

    min_result = paddle.minimum(tensor_x, tensor_y)

    if pg.rank() == 0:
        task = dist.reduce(tensor_x, 0, dist.ReduceOp.MIN, sync_op=False)
        task.wait()
        assert np.array_equal(tensor_x, min_result) and len(tensor_x.shape) == 0
    else:
        task = dist.reduce(tensor_y, 0, dist.ReduceOp.MIN, sync_op=False)
        task.wait()

    print("test reduce with zero dim min api ok")

    if dtype != "bfloat16":
        # test reduce with zero dim product
        # rank 0
        x = np.random.random(shape)
        tensor_x = paddle.to_tensor(x, dtype=dtype)
        # rank 1
        y = np.random.random(shape)
        tensor_y = paddle.to_tensor(y, dtype=dtype)

        prod_result = paddle.multiply(tensor_x, tensor_y)

        if pg.rank() == 0:
            task = dist.reduce(tensor_x, 0, dist.ReduceOp.PROD, sync_op=False)
            task.wait()
            assert np.array_equal(tensor_x, prod_result) and len(tensor_x.shape) == 0
        else:
            task = dist.reduce(tensor_y, 0, dist.ReduceOp.PROD, sync_op=False)
            task.wait()

        print("test reduce with zero dim prod api ok")


if __name__ == "__main__":
    unittest.main()
