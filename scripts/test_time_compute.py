#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging

import torch
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from collections import defaultdict, Counter
from latex2sympy2 import latex2sympy
from sympy import latex, simplify



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_canonical_form(expr: str) -> str:
    try:
        return latex(simplify(latex2sympy(expr)))
    except Exception:
        return expr.strip()

from collections import defaultdict

def find_majority_answer(answers: list) -> str:
    canonical_groups = defaultdict(int)
    canonical_to_original = {}

    for answer in answers:
        answer_text = getattr(answer, "text", str(answer))

        canonical = get_canonical_form(answer_text)
        canonical_groups[canonical] += 1

        if canonical not in canonical_to_original:
            canonical_to_original[canonical] = answer_text

    max_count = max(canonical_groups.values())
    for canonical, count in canonical_groups.items():
        if count == max_count:
            return canonical_to_original[canonical]

def majority_vote(batch, config, sampling_params, llm, prm=None):
    
    prompts = batch["problem"]
    all_completions = []
    preds = []

    for prompt in prompts:
        full_prompt = f"{config.prompt.strip()}\n\n### Problem:\n{prompt.strip()}"
        completions = llm.generate(full_prompt, sampling_params=sampling_params)
        all_completions.append(completions)
        pred = find_majority_answer(completions)
        preds.append(pred)

    return {
        "completions": all_completions,
        "pred": preds,
    }

APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "majority_vote": majority_vote,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
        n=config.n,
    )

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = load_prm(config)

    dataset = get_dataset(config)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm, "sampling_params": sampling_params},
        desc="Running search",
        load_from_cache_file=False,
        
    )

    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
