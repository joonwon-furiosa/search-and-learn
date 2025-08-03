import os
from collections import defaultdict
from typing import List
from pathlib import Path
import yaml

from datasets import load_dataset
from sympy import simplify, latex
from latex2sympy2 import latex2sympy
from vllm import LLM, SamplingParams


# --------------------------
# Util functions
# --------------------------
def get_canonical_form(expr: str) -> str:
    try:
        parsed_expr = latex2sympy(expr)
        simplified_expr = simplify(parsed_expr)
        return latex(simplified_expr)
    except Exception:
        return expr.strip()


def find_majority_answer(answers: List[str]) -> str:
    canonical_groups = defaultdict(int)
    canonical_to_original = {}

    for answer in answers:
        canonical = get_canonical_form(answer)
        canonical_groups[canonical] += 1
        if canonical not in canonical_to_original:
            canonical_to_original[canonical] = answer

    max_count = max(canonical_groups.values())
    for canonical, count in canonical_groups.items():
        if count == max_count:
            return canonical_to_original[canonical]


# --------------------------
# Main
# --------------------------
def main():
    config_path = "/root/search-and-learn/recipes/Llama-3.1-8B-Instruct/majority_voting_re.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
    llm = LLM(model=config["model_path"])

    sampling_params = SamplingParams(
        n=int(config.get("num_completions", 3)),
        temperature=float(config.get("temperature", 0.8)),
        top_p=float(config.get("top_p", 1.0)),
        max_tokens=int(config.get("max_tokens", 2048)),
    )

    def majority_vote(example):
        full_prompt = f"{config['prompt']}\n\n{example['question']}"
        outputs = llm.generate(full_prompt, sampling_params=sampling_params)
        completions = [o.outputs[0].text for o in outputs]
        pred = find_majority_answer(completions)
        return {"prediction": pred}

    dataset = dataset.map(majority_vote)
    dataset.save_to_disk(config.get("output_dir", "./mv_results"))


if __name__ == "__main__":
    main()