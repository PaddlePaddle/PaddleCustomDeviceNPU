import paddle
import paddle.distributed as dist

paddle.set_device("intel_hpu")

dist.init_parallel_env()
if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]], dtype="float32")
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]], dtype="float32")
dist.all_reduce(data)
print(data)
