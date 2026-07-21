"""Eval task registry + dynamic task-key resolution for run_experiment.py.

Kept dependency-free (stdlib only) so it can be unit-tested without importing
torch / lm_eval / vLLM, which run_experiment.py pulls in at module load.
"""

import re

# The tasks the reproduced results are measured on.
#
#   Table 6      all six below
#   Figure 6(a)  wikitext
#   Figure 11    gsm8k_cot, humaneval_instruct
#
# A run may also ask for a truncated variant of any of these as '<key>-<N>',
# which evaluates only the first N samples. See resolve_task_config.
TASKS = {
    "wikitext": {
        "task_name": "wikitext",
        "metrics": ["word_perplexity"],
        "short_name": "WikiText-2",
    },
    "lambada_openai": {
        "task_name": "lambada_openai",
        "metrics": ["acc"],
        "short_name": "LAMBADA",
    },
    "arc_challenge": {
        "task_name": "arc_challenge",
        "metrics": ["acc_norm"],
        "short_name": "ARC-Challenge",
    },
    "arc_easy": {
        "task_name": "arc_easy",
        "metrics": ["acc_norm"],
        "short_name": "ARC-Easy",
    },
    "piqa": {
        "task_name": "piqa",
        "metrics": ["acc_norm"],
        "short_name": "PIQA",
    },
    "winogrande": {
        "task_name": "winogrande",
        "metrics": ["acc"],
        "short_name": "WinoGrande",
    },
    "gsm8k_cot": {
        "task_name": "gsm8k_cot",
        "metrics": ["exact_match,strict-match"],
        "short_name": "GSM8K-CoT",
        "num_fewshot": 8,  # Standard 8-shot chain-of-thought
    },
    "humaneval_instruct": {
        "task_name": "humaneval_instruct",
        "metrics": ["pass@1,create_test"],  # matches lm_eval results key exactly
        "short_name": "HumanEval",
        "num_fewshot": 0,
        "unsafe_code": True,  # gate: runner enables code execution for this task
    },
}


def resolve_task_config(task_key: str) -> dict | None:
    """Return the TASKS config for task_key.

    For a '<base>-<N>' key (hyphen separator, integer N) whose base is a known
    task, returns a copy of that base config with limit=N applied and the
    short_name suffixed with '-<N>'. Returns None if the key is neither a known
    task nor a '<base>-<N>' pattern with a base present in TASKS.
    """
    if task_key in TASKS:
        return TASKS[task_key]
    m = re.fullmatch(r"(.+)-(\d+)", task_key)
    if m:
        base, limit = m.group(1), int(m.group(2))
        if base in TASKS:
            cfg = dict(TASKS[base])  # copy: inherit task_name, metrics, num_fewshot
            cfg["short_name"] = f"{cfg['short_name']}-{limit}"
            cfg["limit"] = limit
            return cfg
    return None
