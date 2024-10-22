
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
tensor_list = []
if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]], dtype="float32")
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]], dtype="float32")
dist.all_gather(tensor_list, data)
print(tensor_list)
# [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)

