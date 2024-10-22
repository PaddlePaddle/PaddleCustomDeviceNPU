
# cmd


`INTEL_HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PADDLE_DISTRI_BACKEND=xccl PADDLE_XCCL_BACKEND=intel_hpu  python -m paddle.distributed.launch --devices "6,7" --log_level=DEBUG allreduce.py`
