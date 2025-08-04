import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
import torch
import random
import numpy as np
import json
import pandas as pd
import asyncio
import time
from openai import AsyncOpenAI
from datasets import load_dataset
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from transformers import AutoTokenizer

from util_func import strip_string, last_boxed_only_string, remove_boxed, is_equiv
from monitor_power_func import monitor_npu_power_usage, monitor_gpu_power_usage, calculate_avg_power_usage

# ------------------------- Config -------------------------
config = {
    "output_len_fix": "1000",
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model": "EMPTY",
    "dataset_name": "HuggingFaceH4/MATH-500",
    "dataset_split": "test",
    "prompt": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem."""
}

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def count_prompt_tokens(prompt: str) -> int:
    return len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

# ------------------------- OpenAI API Client -------------------------
def create_client():
    return AsyncOpenAI(base_url=config["api_url"], api_key="EMPTY")

client = create_client()

def system(msg):
    return {"role": "system", "content": msg}

def user(msg):
    return {"role": "user", "content": msg}

from typing import Optional, Union

import aiohttp
from tqdm import tqdm
from dataclasses import dataclass, field
import traceback
import sys 

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=600)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    prompt: str = ""
    initiated_timestamp: datetime = datetime.now()
    completed_timestamp: datetime = datetime.now()

async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("chat/completions", "profile")
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt
                },
            ],
            "temperature": 0.8,
            "min_tokens": request_func_input.output_len,
            "max_tokens": request_func_input.output_len,
            # You may remove "max_completion_tokens" if previously used
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


# ------------------------- Utility -------------------------
def set_seed(seed: int):
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_majority_answer(completions: List[str]) -> Tuple[str, Dict[str, int]]:
    canonical_groups = defaultdict(int)
    canonical_to_original = {}

    for output in completions:
        boxed = last_boxed_only_string(output)
        if boxed:
            try:
                answer = remove_boxed(boxed)
            except Exception:
                continue
        else:
            continue

        canonical = strip_string(answer)
        canonical_groups[canonical] += 1
        if canonical not in canonical_to_original:
            canonical_to_original[canonical] = answer

    if not canonical_groups:
        return "", {}

    max_count = max(canonical_groups.values())
    for canonical, count in canonical_groups.items():
        if count == max_count:
            return canonical_to_original[canonical], dict(canonical_groups)

def majority_vote(batch: Dict, completions: List[str]) -> Dict:
    pred, votes = find_majority_answer(completions)
    return {
        "completions": completions,
        "pred": pred,
        "votes": votes,
    }


def save_results(run_dir: Path, result: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


async def one_request_from_input(req: RequestFuncInput) -> RequestFuncOutput:
    output = await async_request_openai_chat_completions(req)
    return output

async def generate_n_responses_from_inputs(
    req_inputs: List[RequestFuncInput],
    timestamps: dict
) -> List[RequestFuncOutput]:
    timestamps["api_start"] = time.time()
    tasks = [one_request_from_input(req) for req in req_inputs]
    responses = await asyncio.gather(*tasks)
    timestamps["batch_done"] = time.time()
    return responses

async def generate_n_responses(prompt: str, n: int, timestamps: dict) -> List[str]:
    prompt_len = count_prompt_tokens(prompt)
    req_inputs = [
        RequestFuncInput(
            prompt=prompt,
            api_url=config["api_url"],
            prompt_len=prompt_len,
            output_len=int(config["output_len_fix"]),
            model=config["model"],
        )
        for _ in range(n)
    ]
    outputs = await generate_n_responses_from_inputs(req_inputs, timestamps)
    filtered_responses = [o.generated_text for o in outputs if o.success and o.generated_text.strip()]
    return filtered_responses, outputs

def main():
    set_seed(42)

    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    dataset = load_dataset(config['dataset_name'], split=config['dataset_split'])

    n_values = [64, 1, 8] # [64, 1, 2, 4, 8, 16, 32, 48]
    for idx in range(500):
        for n in n_values:

            run_dir = Path(f"results/batch_{n}/problem_{idx}")
            run_dir.mkdir(parents=True, exist_ok=True)
            power_monitor_path = run_dir / "device_status.csv"

            stop_queue = Queue()
            monitor_process = Process(
                target=monitor_npu_power_usage,
                args=(str(power_monitor_path), stop_queue)
            )
            monitor_process.start()

            timestamps = {}
            try:
                batch = dataset[idx]
                
                full_prompt = f"{config['prompt'].strip()}\n\n### Problem:\n{batch['problem'].strip()}"
                print(full_prompt)
                responses, outputs = asyncio.run(generate_n_responses(full_prompt, n, timestamps))
                with open(run_dir / "all_responses_debug.json", "w") as f:
                    json.dump([o.generated_text for o in outputs], f, indent=2)
                with open(run_dir / "raw_outputs_debug.json", "w") as f:
                    json.dump([o.__dict__ for o in outputs], f, indent=2, default=str)
                result = majority_vote(batch, responses)
                print(f"Debug: Majority vote result for idx={idx}, n={n}: pred={result['pred']}, votes={result['votes']}")
                timestamps["majority_done"] = time.time()

                is_correct = is_equiv(result["pred"], batch['answer'], verbose=False)
                duration = timestamps["majority_done"] - timestamps["api_start"]
                batch_duration = timestamps["batch_done"] - timestamps["api_start"]
                num_all_tokens = int(config["output_len"])*n
                tps = num_all_tokens / batch_duration if batch_duration > 0 else 0.0

            except Exception as e:
                result = {"error": str(e)}
                is_correct = False
                duration = 0.0
                batch_duration = 0.0
                tps = 0.0
                outputs = []

            stop_queue.put("stop")
            monitor_process.join()
            avg_power = calculate_avg_power_usage(str(power_monitor_path))

            save_results(run_dir, {
                "index": idx,
                "batch_size": n,
                "prediction": result.get("pred"),
                "votes": result.get("votes"),
                "is_correct": is_correct,
                "avg_power_usage": avg_power,
                "e2e_latency_sec": duration,
                "inference_latency_sec": batch_duration,
                "output_tps": tps,
                "timestamps": timestamps,
                "error": result.get("error", None),
                "ttft_avg": np.median([o.ttft for o in outputs if o.success]) if outputs else 0.0,
                "tpot_avg": np.median([o.itl for o in outputs if o.success]) if outputs else 0.0,
                "fail_rate": 1.0 - len(responses) / len(outputs) if outputs else 1.0,
                "num_success": len([o for o in outputs if o.success]),
                "num_fail": len([o for o in outputs if not o.success]),
            })

            print(f"Finished problem_{idx}, batch_{n}, correct={is_correct}")


if __name__ == "__main__":
    main()
