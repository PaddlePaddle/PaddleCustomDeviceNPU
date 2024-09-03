import paddle
from paddle.static import InputSpec
from paddle.distributed import fleet
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

paddle.set_device("intel_hpu")

model = "meta-llama/Llama-2-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model)
strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {"dp_degree": 1,
                           "mp_degree": 2,
                           "pp_degree": 1,
                           "sharding_degree": 1,
                           }
fleet.init(is_collective=True, strategy=strategy)
hcg = fleet.get_hybrid_communicate_group()
tensor_parallel_rank = hcg.get_model_parallel_rank()

model = AutoModelForCausalLM.from_pretrained(model, tensor_parallel_degree=2, tensor_parallel_rank=tensor_parallel_rank, dtype="float32")
input_features = tokenizer("I am a student", return_tensors="pd")

outputs = model.generate(**input_features, max_length=20)
print(outputs[0])
print(tokenizer.batch_decode(outputs[0]))

# python -m paddle.distributed.launch --devices "3,4" test_llama.py