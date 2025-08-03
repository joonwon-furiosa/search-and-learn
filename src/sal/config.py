from dataclasses import dataclass
from typing import Optional, List, Literal

from huggingface_hub import get_full_repo_name
from sal.utils.hub import get_dataset_revisions

@dataclass
class Config:
    approach: Literal["best_of_n", "beam_search", "dvts", "majority_vote"] = "best_of_n"
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    gpu_memory_utilization: float = 0.5
    prm_path: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    
    # SamplingParams 관련 파라미터 (vllm)
    top_k: int = -1
    best_of: int = None
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature_decay: Optional[float] = None
    use_beam_search: bool = False


    # Output Related Options
    output_dir: str = None
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    hub_dataset_private: bool = False
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = None

    # Chat template related options
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    custom_chat_template: str = "..."  # omitted for brevity

    # Search Related Options
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 2048
    agg_strategy: str = "last"

    # DVTS / Beam Search options
    beam_width: int = 4
    num_iterations: int = 40
    lookahead: int = 0

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False

    # Majority voting specific options
    num_completions: int = 3
    prompt: str = ""

    def __post_init__(self):
        if self.approach == "dvts":
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.n // self.beam_width

        if self.approach == "beam_search":
            if self.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for beam_search")

        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            if self.approach in ["beam_search", "dvts"]:
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--m-{self.beam_width}--iters-{self.num_iterations}--look-{self.lookahead}--seed-{self.seed}--agg_strategy--{self.agg_strategy}"
            elif self.approach in ["best_of_n", "majority_vote"]:
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--seed-{self.seed}--agg_strategy-{self.agg_strategy}"
            else:
                raise ValueError(f"Unknown approach {self.approach}")

            if self.dataset_start is not None and self.dataset_end is not None:
                self.revision = (
                    f"{self.revision}--chunk-{self.dataset_start}_{self.dataset_end}"
                )

            if not self.overwrite_hub_revision and self.revision in revisions:
                exit()