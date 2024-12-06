#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import argparse
import json
import logging
import math
import time
from itertools import cycle
from pathlib import Path

import torch


os.environ["HABANA_LOGS"] = "logs"
os.environ["LOG_LEVEL_ALL"] = "0"
os.environ["GLOG_v"] = "0"

import paddle
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from paddle.distributed import fleet


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_parser(parser):
    # Arguments management
    parser.add_argument(
        "--device_id",
        "-device",
        type=lambda value: f"intel_hpu:{int(value)}",
        help="Device to run",
        default="0",
    )
    parser.add_argument(
        "--model_name_or_path",
        "-model",
        default="meta-llama/Llama-2-7b-chat",
        type=str,
        # required=True,
        help="Path to pre-trained model (on the BOS Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_false",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Number of tokens to generate."
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup iterations for benchmarking.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1,
        help="Number of inference iterations for benchmarking.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, metavar="N", help="Local process rank."
    )
    parser.add_argument("--world_size", type=int, default=1, help="HPU node size.")

    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="Size of candidate set used for re-ranking in contrastive search. top_k > 1 enables contrastive search.",
    )
    parser.add_argument(
        "--penalty_alpha",
        default=None,
        type=float,
        help="Degeneration penalty for contrastive search. penalty_alpha > 0 enables contrastive search.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument(
        "--verbose_workers",
        action="store_true",
        help="Enable output from non-master workers",
    )
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )
    parser.add_argument(
        "--use_fused_rms_norm",
        action="store_true",
        help="Whether to enable Habana RMS Norm fused Op, provided that the model supports it.",
    )
    parser.add_argument(
        "--use_fused_rope",
        action="store_true",
        help="Whether to enable Habana Rotray position embedding fused Op, provided that the model supports it.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        action="store_true",
        help="Whether to enable Habana Flash Attention in fast softmax mode.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable stopping with eos token when calling `generate`. --no-ignore_eos to disable it",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Temperature value for text generation",
    )
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="Top_p value for generating text via sampling",
    )
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--static_shapes",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--input_embeds",
        action="store_true",
        help="Whether to enable inputs_embeds or not.",
    )

    args = parser.parse_args()

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    if args.use_flash_attention and not args.flash_attention_fast_softmax:
        args.flash_attention_fast_softmax = True

    args.quant_config = os.getenv("QUANT_CONFIG", "")

    return args


def setup_model(args, model_dtype):
    paddle.set_device(args.device_id)
    paddle.set_default_dtype(model_dtype)

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": args.world_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    tensor_parallel_rank = hcg.get_model_parallel_rank()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=model_dtype,
        tensor_parallel_degree=args.world_size,
        tensor_parallel_rank=tensor_parallel_rank,
    )

    return model


def setup_tokenizer(args, model):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def setup_generation_config(args, model, tokenizer):
    bad_words_ids = None
    force_words_ids = None
    if args.bad_words is not None:
        bad_words_ids = [
            tokenizer.encode(bad_word, add_special_tokens=False)
            for bad_word in args.bad_words
        ]
    if args.force_words is not None:
        force_words_ids = [
            tokenizer.encode(force_word, add_special_tokens=False)
            for force_word in args.force_words
        ]

    # Generation configuration
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = args.use_kv_cache
    generation_config.static_shapes = args.static_shapes
    generation_config.bucket_size = args.bucket_size
    generation_config.bucket_internal = args.bucket_internal
    generation_config.do_sample = args.do_sample
    generation_config.num_beams = args.num_beams
    generation_config.top_k = args.top_k
    generation_config.penalty_alpha = args.penalty_alpha
    generation_config.bad_words_ids = bad_words_ids
    generation_config.force_words_ids = force_words_ids
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.trim_logits = args.trim_logits
    generation_config.attn_softmax_bf16 = args.attn_softmax_bf16
    generation_config.limit_hpu_graphs = args.limit_hpu_graphs
    generation_config.reuse_cache = args.reuse_cache
    generation_config.reduce_recompile = args.reduce_recompile
    if generation_config.reduce_recompile:
        assert generation_config.bucket_size > 0
    generation_config.use_flash_attention = args.use_flash_attention
    generation_config.flash_attention_recompute = args.flash_attention_recompute
    generation_config.flash_attention_causal_mask = args.flash_attention_causal_mask
    generation_config.flash_attention_fast_softmax = args.flash_attention_fast_softmax

    generation_config.use_fused_rms_norm = args.use_fused_rms_norm
    generation_config.use_fused_rope = args.use_fused_rope

    generation_config.trust_remote_code = args.trust_remote_code

    return generation_config


def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)
    logger.info(f"Args: {args}")
    logger.info(f"device: {args.device_id}, bf16: {args.bf16 == True}")

    input_sentences = [
        "Please introduce Large Language Models",
        "What are the benefits of using Large Language Models?",
        "Explain who you are",
        "DeepSpeed is a machine learning framework",
        "He is working on",
        "He has a",
        "He got all",
        "Everyone is happy and I can",
        "The new movie that got Oscar this year",
        "In the far far distance from our galaxy,",
        "Peace is the only way",
    ]

    if args.batch_size > len(input_sentences):
        # Dynamically extends to support larger batch sizes
        num_sentences_to_add = args.batch_size - len(input_sentences)
        for i in range(num_sentences_to_add):
            input_sentences.append(input_sentences[i % len(input_sentences)])
    elif args.batch_size < len(input_sentences):
        input_sentences = input_sentences[: args.batch_size]

    model = setup_model(args, "bfloat16")

    tokenizer, model = setup_tokenizer(args, model)
    generation_config = setup_generation_config(args, model, tokenizer)

    use_lazy_mode = True

    def generate(size=None, reduce_recompile=False):
        """Generates sequences from the input sentences and returns them."""

        # Tokenization
        input_tokens = tokenizer(
            input_sentences, return_tensors="pd", padding=True, truncation=True
        )

        if not reduce_recompile:
            # Move inputs to target device(s)
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(args.device_id)

        with paddle.amp.auto_cast(
            dtype="bfloat16", custom_white_list={"elementwise_add", "rms_norm"}
        ):
            outputs = model.generate(
                **input_tokens,
                generation_config=generation_config,
            )

        return tokenizer.batch_decode(outputs[0])

    # Compilation
    logger.info("Graph compilation...")
    dyn_prompt_lens = args.simulate_dyn_prompt
    t0 = time.perf_counter()
    # The first three iterations take longer because of graph compilation
    if dyn_prompt_lens is None or len(set(dyn_prompt_lens)) == 1:
        for i in range(args.warmup):
            if dyn_prompt_lens is None:
                print(f"Warming up iteration {i+1}/{args.warmup}", flush=True)
                generate(None, args.reduce_recompile)
            else:
                print(
                    f"Warming up for shape {dyn_prompt_lens[0]} iteration {i+1}/{args.warmup}",
                    flush=True,
                )
                generate(dyn_prompt_lens[0], args.reduce_recompile)
    else:
        if args.bucket_size > 0:
            mn = min(dyn_prompt_lens)
            mx = max(dyn_prompt_lens)

            def rounder(x):
                return int(math.ceil(x / args.bucket_size) * args.bucket_size)

            min_prompt_len = rounder(mn)
            max_sentence_len = rounder(mx)
            for i in range(args.warmup):
                lst = list(
                    range(min_prompt_len, max_sentence_len + 1, args.bucket_size)
                )
                for sz in lst:
                    print(
                        f"Warming up for shape {sz - 1} iteration {i+1}/{args.warmup}",
                        flush=True,
                    )
                    generate(sz - 1, args.reduce_recompile)

    total_new_tokens_generated = 0
    logger.info("Running generate...")
    print("Running generate...")
    t0 = time.perf_counter()
    # Benchmark over n_iterations iterations
    if dyn_prompt_lens is None:
        for i in range(args.n_iterations):
            generated = generate(None, args.reduce_recompile)
    else:
        repeated_prompt_len = cycle(dyn_prompt_lens)
        for i in range(args.n_iterations):
            prompt_len = next(repeated_prompt_len)
            print("Generating for shape,", prompt_len)
            generated = generate(prompt_len, args.reduce_recompile)
    duration = time.perf_counter() - t0
    total_new_tokens_generated = (
        args.n_iterations * args.batch_size * args.max_new_tokens
    )
    throughput = total_new_tokens_generated / duration

    print()
    print("Input/outputs:")
    for i, input_sentence in enumerate(zip(input_sentences)):
        print(f"input {i+1}: {input_sentence}")
        for j, output in enumerate(
            zip(
                generated[
                    args.num_return_sequences * i : args.num_return_sequences * (i + 1)
                ]
            )
        ):
            print(f"output {j+1}: {output}")
        print()

    # Store results if necessary
    if args.output_dir is not None and args.global_rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "throughput": throughput,
            "output": output,
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    stats = "Input embeds" if args.input_embeds else "Input tokens"
    stats = (
        stats + f"\nThroughput (including tokenization) = {throughput} tokens/second"
    )
    separator = "-" * len(stats)
    print()
    print("Stats:")
    print(separator)
    print(stats)
    print(separator)
    print()


if __name__ == "__main__":
    main()
