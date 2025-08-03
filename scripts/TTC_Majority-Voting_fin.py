# filename: run_math_majority_vote.py

import os
import torch
import random
import numpy as np
import json
import pandas as pd
import aiohttp
import asyncio
import time

from vllm import LLM, SamplingParams
from datasets import load_dataset
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from utils import strip_string, last_boxed_only_string, remove_boxed, is_equiv
from monitor_power import monitor_npu_power_usage, monitor_gpu_power_usage, calculate_avg_power_usage


# ------------------------- Config -------------------------
config = {
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
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

def save_results_and_csv(run_dir: Path, result: dict, is_correct: bool):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "result.json", "w") as f:
        json.dump({
            "pred": result.get("pred"),
            "votes": result.get("votes"),
            "completions": result.get("completions"),
            "is_correct": is_correct
        }, f, indent=2)


# API request
async def send_prompt(session: aiohttp.ClientSession, prompt: str, idx: int, context_len: int) -> str:
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 1.0,
        "min_tokens": context_len-1,
        "max_tokens": context_len
    }
    try:
        async with session.post(config["api_url"], json=payload) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[{idx}] request failed: {e}")
        return ""

async def generate_n_responses(prompt: str, n: int) -> List[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [send_prompt(session, prompt, i, 1500) for i in range(n)]
        return await asyncio.gather(*tasks)



def main():
    set_seed(42)

    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    
    dataset = load_dataset(config['dataset_name'], split=config['dataset_split'])

    n_values = [64, 1, 2, 4, 8, 16, 32, 48]

    for n in n_values:
        for idx in range(500):
            run_dir = Path(f"results/batch_{n}/problem_{idx}")
            run_dir.mkdir(parents=True, exist_ok=True)
            power_monitor_path = run_dir / "device_status.csv"

            stop_queue = Queue()
            monitor_process = Process(
                target=monitor_gpu_power_usage,
                args=(str(power_monitor_path), stop_queue)
            )
            monitor_process.start()
            print(f'Problem # {idx} .. \n')
            try:
                batch = dataset[idx]
                full_prompt = f"{config['prompt'].strip()}\n\n### Problem:\n{batch['problem'].strip()}"
                response_n = asyncio.run(generate_n_responses(full_prompt, n))

                result = majority_vote(batch, response_n)
                is_correct = is_equiv(result["pred"], batch['answer'], verbose=False)
            except Exception as e:
                result = {"error": str(e)}
                is_correct = False

            stop_queue.put("stop")
            monitor_process.join()
            avg_power = calculate_avg_power_usage(str(power_monitor_path))


            with open(run_dir / "result.json", "w") as f:
                json.dump({
                    "index": idx,
                    "batch_size": n,
                    "prediction": result.get("pred"),
                    "votes": result.get("votes"),
                    "is_correct": is_correct,
                    "avg_power_usage": avg_power,
                    "error": result.get("error", None)
                }, f, indent=2)

            print(f"✅ Finished problem_{idx}, batch_{n}, correct={is_correct}")

if __name__ == "__main__":
    main()