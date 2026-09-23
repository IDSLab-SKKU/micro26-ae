#!/usr/bin/env python3
"""
Runner for the LLM accuracy experiments.

Evaluates a quantized model with lm-eval-harness, either on the native tensor
cores or with the linear-layer GEMMs routed through the MMA emulation kernel
under a chosen accumulation configuration.

An experiment is a directory holding a config.yaml. The config declares which
element format it runs under ('precision: fp8' or 'nvfp4'), the emulation
settings, and the evaluation tasks. It may also declare a sweep, in which case
every combination is run in turn.

Usage:
    # List the experiments that ship with the artifact
    python scripts/run_experiment.py --list

    # Run one
    python scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda

    # Preview the sweep combinations without loading a model
    python scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda --dry-run

    # Pin to a GPU
    python scripts/run_experiment.py --exp exp1_table6_cross_arch/h100/native --gpu 1

    # Re-run every combination, replacing results that already exist
    python scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda --overwrite

A sweep resumes by default: a combination whose result JSON already exists in
results/ (same sweep parameters, every task evaluated without error) is skipped,
so an interrupted sweep continues where it stopped. --overwrite disables this.
"""

import os
import sys

# Limit BLAS/OpenMP threads BEFORE importing numpy/scipy/sklearn
# This prevents thread exhaustion when using multiprocessing with lm_eval
# OpenBLAS defaults to 64 threads, which causes "Resource temporarily unavailable"
# errors when spawning multiple worker processes
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import gc
import multiprocessing
import warnings
from pathlib import Path
from itertools import product

# Suppress the resource_tracker semaphore leak warning
# This warning is benign - it occurs because vLLM's engine processes
# create semaphores (via ZMQ/multiprocessing) that aren't fully cleaned up
# before Python's resource_tracker runs at interpreter shutdown.
#
# The warning comes from multiprocessing/resource_tracker.py and is harmless.
# We suppress it by filtering warnings and patching the resource tracker.
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")

# Monkey-patch the resource_tracker to suppress the warning
# This is necessary because the warning is emitted during atexit cleanup
def _patch_resource_tracker():
    try:
        from multiprocessing import resource_tracker
        # Store the original warn function
        _original_warn = warnings.warn
        def _filtered_warn(message, *args, **kwargs):
            if "resource_tracker" in str(message) and "leaked" in str(message):
                return  # Suppress this specific warning
            return _original_warn(message, *args, **kwargs)
        # Apply the filter in the resource_tracker module's namespace
        resource_tracker.warnings = type(sys)('warnings')
        resource_tracker.warnings.warn = _filtered_warn
    except Exception:
        pass

_patch_resource_tracker()

# Set multiprocessing start method to 'spawn' BEFORE any CUDA operations
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
# when running multiple experiments in sequence
# MUST be at module level (not inside if __name__ == "__main__")
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set - this is fine
    pass


# =============================================================================
# Precision-Specific Configuration
# =============================================================================

PRECISION_CONFIG = {
    "fp8": {
        "label": "FP8",
        "default_model": "nvidia/Llama-3.1-8B-Instruct-FP8",
        "env_vars": {
            "enabled": "VLLM_USE_MMAEMU_GEMM_FP8",
            "algorithm": "VLLM_MMAEMU_FP8_ALGORITHM",
            "chunk_size": "VLLM_MMAEMU_FP8_CHUNK_SIZE",
            "f_bits": "VLLM_MMAEMU_FP8_F_BITS",
            "g_bits": "VLLM_MMAEMU_FP8_G_BITS",
            "group_size": "VLLM_MMAEMU_FP8_GROUP_SIZE",
        },
        "has_kv_cache_dtype": True,    # FP8 models support kv_cache_dtype
        "has_chunk_size": True,        # CoFDA chunk size
        "has_group_size": True,        # GDFS group size
    },
    "nvfp4": {
        "label": "NVFP4",
        "default_model": "nvidia/Llama-3.1-8B-Instruct-NVFP4",
        "env_vars": {
            "enabled": "VLLM_USE_MMAEMU_GEMM_NVFP4",
            "algorithm": "VLLM_MMAEMU_NVFP4_ALGORITHM",
            "f_bits": "VLLM_MMAEMU_NVFP4_F_BITS",
            "g_bits": "VLLM_MMAEMU_NVFP4_G_BITS",
        },
        "has_kv_cache_dtype": False,   # NVFP4 models do not use kv_cache_dtype
        "has_chunk_size": False,       # NVFP4 chunk size is fixed in-kernel
        "has_group_size": False,       # NVFP4 group size is fixed in-kernel
    },
}


# =============================================================================
# Parse arguments BEFORE any CUDA imports
# =============================================================================
def parse_args_early():
    """Parse arguments before CUDA initialization."""
    parser = argparse.ArgumentParser(
        description="Runner for the LLM accuracy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py --list
  python scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda
  python scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda --dry-run
  python scripts/run_experiment.py --exp exp1_table6_cross_arch/h100/native --gpu 1
  python scripts/run_experiment.py --exp exp2_figure6a_fp8_cofda --overwrite
        """
    )
    parser.add_argument(
        "--exp", type=str,
        help="Experiment directory, relative to artifact/"
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp8", "nvfp4"], default=None,
        help="Override the element format declared in config.yaml"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available experiments"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview sweep combinations without running"
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="CUDA device index to run on (e.g., '0' or '1'); use one GPU"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run every combination and replace existing results "
             "(default: skip combinations that already have a complete result)"
    )
    return parser.parse_args()


# Parse args immediately (before any CUDA imports)
_args = parse_args_early()


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def get_experiments_dir() -> Path:
    """Experiment roots live in artifact/, the parent of this scripts/ dir."""
    return get_script_dir().parent


def find_experiments(base_dir: Path) -> list[Path]:
    """Find all experiment directories (directories with config.yaml)."""
    experiments = []
    for config_file in base_dir.rglob("config.yaml"):
        experiments.append(config_file.parent)
    return sorted(experiments)


def list_experiments():
    """List all available experiments."""
    experiments_dir = get_experiments_dir()
    experiments = find_experiments(experiments_dir)

    if not experiments:
        print(f"No experiments found under {experiments_dir}")
        return

    print("\nAvailable experiments:")
    for exp_dir in experiments:
        print(f"  - {exp_dir.relative_to(experiments_dir)}")


def read_precision(config_path: Path) -> str:
    """Read the element format an experiment runs under from its config."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("precision")


# Handle --list before any CUDA initialization
if _args.list:
    list_experiments()
    sys.exit(0)

# Validate experiment path
if not _args.exp:
    print("Error: --exp is required (or use --list to see available experiments)")
    print("Usage: python scripts/run_experiment.py --exp <experiment_path>")
    sys.exit(1)

# Resolve experiment path
exp_path = Path(_args.exp)
if not exp_path.is_absolute():
    exp_path = get_experiments_dir() / exp_path

if not exp_path.exists():
    print(f"Error: Experiment directory not found: {exp_path}")
    list_experiments()
    sys.exit(1)

config_path = exp_path / "config.yaml"
if not config_path.exists():
    print(f"Error: config.yaml not found in {exp_path}")
    sys.exit(1)

# The element format comes from the config, not from the directory name.
_precision = _args.precision or read_precision(config_path)
if _precision not in PRECISION_CONFIG:
    choices = "|".join(PRECISION_CONFIG)
    print(f"Error: config.yaml must declare 'precision: {choices}' "
          f"(or pass --precision). Got: {_precision!r}")
    sys.exit(1)

_pconfig = PRECISION_CONFIG[_precision]

# Set GPU BEFORE importing anything that uses CUDA
if _args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = _args.gpu

# Set environment variables before importing
os.environ["VLLM_USE_V1"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# Now safe to import YAML and load config
# =============================================================================
import yaml

# vLLM default configuration (defined early for dry-run)
VLLM_CONFIG = {
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.70,
    "tensor_parallel_size": 1,
    "enforce_eager": False,
    "kv_cache_dtype": "auto",
    # How many requests lm_eval hands vLLM at once. "auto" batches them all so
    # vLLM can run them concurrently; 1 would feed them one at a time (serial).
    # This changes the batch the GEMMs see, so it moves last-bit-sensitive
    # metrics like perplexity — "auto" is what the paper's numbers used.
    "batch_size": "auto",
    # Running-batch-width cap. None keeps vLLM's own default; set it in a
    # config's `vllm:` block to bound how many sequences run concurrently.
    "max_num_seqs": None,
}


def load_config(config_path: Path) -> dict:
    """Load experiment config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_sweep_combinations(config: dict) -> list[dict]:
    """Generate all parameter combinations from sweep config."""
    sweep = config.get("sweep", {})
    if not sweep:
        # No sweep, return single empty combination
        return [{}]

    # Get all sweep parameters and their values
    param_names = list(sweep.keys())
    param_values = [sweep[name] if isinstance(sweep[name], list) else [sweep[name]]
                    for name in param_names]

    # Generate cartesian product
    combinations = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)

    return combinations


def get_run_name(combo: dict, config: dict) -> str:
    """Generate run name from parameter combination."""
    if not combo:
        # No sweep parameters - check for custom run_name or generate default
        experiment = config.get("experiment", {})
        if "run_name" in experiment:
            return experiment["run_name"]

        # Name the run after what it does: native tensor cores, or emulation
        # under a named algorithm.
        emu = config.get("mma_emu", {})
        if not emu.get("enabled", False):
            return "cutlass"
        return f"emulated_{emu.get('algorithm', 'unknown')}"

    # Build name from parameters
    parts = []
    for key, value in combo.items():
        if key == "f_bits":
            parts.append(f"f{value}")
        elif key == "g_bits":
            parts.append(f"g{value}")
        elif key == "group_size":
            parts.append(f"gs{value}")
        elif key == "chunk_size":
            parts.append(f"cs{value}")
        elif key == "algorithm":
            parts.append(str(value))
        else:
            parts.append(f"{key}_{value}")

    return "_".join(parts)


def completed_result_status(output_path: Path, combo: dict,
                            config: dict) -> str | None:
    """Why an existing result can NOT be reused, or None if it is complete.

    A result counts as complete when its JSON parses, it was written for the
    same precision and sweep parameters, it holds every configured task, and no
    task recorded an error (plus the samples file, if the config logs samples).
    The runner writes the result JSON last and atomically, so a run cut short
    never leaves one behind.
    """
    import json
    from tasks import resolve_task_config  # stdlib-only, safe before CUDA

    if not output_path.exists():
        return "no result yet"
    try:
        result = json.loads(output_path.read_text())
    except (OSError, ValueError):
        return "result JSON is unreadable"

    if result.get("precision") != _precision:
        return f"result is for precision {result.get('precision')!r}"
    if result.get("sweep_params", {}) != combo:
        return "result has different sweep parameters"

    eval_config = config.get("eval", {})
    # Unknown task keys are skipped by the run itself, so they are not expected.
    wanted = [k for k in eval_config.get("tasks", ["wikitext"])
              if resolve_task_config(k) is not None]
    tasks = result.get("tasks", {})
    missing = [k for k in wanted if k not in tasks]
    if missing:
        return f"result is missing task(s): {', '.join(missing)}"
    failed = [k for k in wanted if "error" in tasks[k]]
    if failed:
        return f"task(s) failed in the earlier run: {', '.join(failed)}"

    if eval_config.get("log_samples", False):
        samples_path = output_path.parent / f"{output_path.stem}_samples.json"
        if not samples_path.exists():
            return "samples file is missing"
    return None


def export_mma_emu_to_env(emu_config: dict):
    """Export the MMA emulation settings of a run to environment variables."""
    env_vars = _pconfig["env_vars"]

    enabled = emu_config.get("enabled", False)
    os.environ[env_vars["enabled"]] = "1" if enabled else "0"

    if not enabled:
        # Native run: the GEMM goes to the tensor cores and no accumulation
        # parameter is read.
        return

    if "algorithm" not in emu_config:
        raise ValueError(
            "An emulated run must name its accumulation algorithm "
            f"('algorithm:' in config.yaml) for {_pconfig['label']}.")
    os.environ[env_vars["algorithm"]] = str(emu_config["algorithm"])

    if _pconfig["has_chunk_size"] and "chunk_size" in emu_config:
        os.environ[env_vars["chunk_size"]] = str(emu_config["chunk_size"])

    if "f_bits" in emu_config:
        os.environ[env_vars["f_bits"]] = str(emu_config["f_bits"])

    if "g_bits" in emu_config:
        os.environ[env_vars["g_bits"]] = str(emu_config["g_bits"])

    if _pconfig["has_group_size"] and "group_size" in emu_config:
        os.environ[env_vars["group_size"]] = str(emu_config["group_size"])


def print_vllm_settings(merged_config: dict, indent: str = "  "):
    """Print vLLM settings in multiline format."""
    # Display order matches VLLM_CONFIG keys
    display_keys = [
        "max_model_len",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "enforce_eager",
        "kv_cache_dtype",
        "batch_size",
        "max_num_seqs",
    ]
    print("vLLM:")
    for key in display_keys:
        if key in merged_config and merged_config[key] is not None:
            print(f"{indent}{key}: {merged_config[key]}")


def format_mma_emu_settings(emu_config: dict) -> str:
    """Format the emulation settings of a run for display."""
    if not emu_config.get("enabled", False):
        return "disabled (native tensor cores via CUTLASS)"

    parts = [f"algorithm={emu_config.get('algorithm', 'N/A')}"]

    if _pconfig["has_chunk_size"]:
        parts.append(f"chunk_size={emu_config.get('chunk_size', 'N/A')}")

    parts.append(f"f_bits={emu_config.get('f_bits', 'N/A')}")

    parts.append(f"g_bits={emu_config.get('g_bits', 'N/A')}")

    if _pconfig["has_group_size"]:
        parts.append(f"group_size={emu_config.get('group_size', 'N/A')}")

    return ", ".join(parts)


# Load config and generate combinations
_config = load_config(config_path)
_combinations = generate_sweep_combinations(_config)

# Handle --dry-run
if _args.dry_run:
    exp_name = _config.get("experiment", {}).get("name", exp_path.name)
    print(f"\nExperiment: {exp_name}")
    print(f"Precision: {_pconfig['label']}")
    print(f"Config: {config_path}")

    # Display eval settings
    eval_config = _config.get("eval", {})
    model_name = eval_config.get("model", _pconfig["default_model"])
    task_keys = eval_config.get("tasks", ["wikitext"])

    # Display vLLM settings
    vllm_config = _config.get("vllm", {})
    merged_vllm_config = {**VLLM_CONFIG, **vllm_config}

    print(f"\nModel: {model_name}")
    print(f"Tasks: {', '.join(task_keys)}")
    print(f"Log samples: {eval_config.get('log_samples', False)}")
    print_vllm_settings(merged_vllm_config)

    sweep = _config.get("sweep", {})
    if sweep:
        print(f"\nSweep parameters:")
        for param, values in sweep.items():
            if isinstance(values, list):
                print(f"  {param}: {values}")
            else:
                print(f"  {param}: [{values}]")

    print(f"\nTotal runs: {len(_combinations)}"
          + (" (--overwrite: all re-run)" if _args.overwrite else ""))
    n_skip = 0
    for i, combo in enumerate(_combinations, 1):
        run_name = get_run_name(combo, _config)
        params = ", ".join(f"{k}={v}" for k, v in combo.items()) if combo else "(no sweep)"
        done = (not _args.overwrite and completed_result_status(
            exp_path / "results" / f"{run_name}.json", combo, _config) is None)
        n_skip += done
        mark = "[SKIP]" if done else "[RUN] "
        print(f"  {i}. {mark} {run_name:<15} ({params})")
    if n_skip:
        print(f"\n{n_skip} run(s) already complete and would be skipped "
              "(pass --overwrite to re-run them).")

    print(f"\nRun with: python scripts/run_experiment.py --exp {_args.exp}")
    sys.exit(0)


# =============================================================================
# Now safe to import CUDA-dependent libraries
#
# These are guarded by `if __name__ == "__main__"` for spawn-safety. Under the
# 'spawn' start method (forced above for CUDA), any library that uses
# multiprocessing re-imports THIS module in every worker it spawns. The
# HumanEval `code_eval` metric spawns a Manager()+Process() per candidate to
# execute model-generated code; if these heavy imports ran at module top level
# they would re-execute in each worker, blowing past code_eval's per-candidate
# timeout (worker killed before it runs the candidate -> every problem scores 0)
# and stalling the Manager() handshake -> the scoring phase hangs. Keeping them
# under the guard makes the re-import cheap and side-effect-free (Python docs:
# "Safe importing of main module"). The `if` block does not create a new scope,
# so these names remain module globals for run_benchmark()/main() to use.
# =============================================================================
if __name__ == "__main__":
    import time
    import json
    from datetime import datetime
    import torch
    import torch.distributed as dist

    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM

    # Task registry + dynamic task-key resolution live in tasks.py (stdlib-only,
    # so they stay unit-testable without importing torch / lm_eval / vLLM).
    from tasks import resolve_task_config


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


# Short architecture tags, keyed by compute-capability number (major*10+minor).
# Table 6 is a cross-architecture comparison, so results are stamped with the
# GPU they ran on and the two machines stay distinguishable once collected.
# Blackwell covers both SM100 (B200) and SM120 (the RTX PRO 6000 we report);
# run_table6.sh accepts either for the emulated-Hopper side.
_ARCH_TAGS = {90: "hopper", 100: "blackwell", 120: "blackwell"}


def get_device_info() -> dict:
    """Identify the GPU a run executed on, for the result's provenance."""
    import torch
    if not torch.cuda.is_available():
        return {"name": None, "capability": None, "sm": None, "arch": None}
    major, minor = torch.cuda.get_device_capability(0)
    sm = major * 10 + minor
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": f"{major}.{minor}",
        "sm": sm,
        "arch": _ARCH_TAGS.get(sm, f"sm{sm}"),
    }


def build_vllm_kwargs(model_name: str, config: dict) -> dict:
    """Build the kwargs lm_eval's VLLM wrapper passes on to LLM()."""
    kwargs = dict(
        pretrained=model_name,
        dtype="auto",
        tensor_parallel_size=config["tensor_parallel_size"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_model_len"],
        enforce_eager=config["enforce_eager"],
        batch_size=config["batch_size"],
        seed=42,
    )

    # kv_cache_dtype is only meaningful for the FP8 checkpoints.
    if _pconfig["has_kv_cache_dtype"]:
        kwargs["kv_cache_dtype"] = config["kv_cache_dtype"]

    # Running batch width; forwarded unless a config nulls it out to fall back
    # to vLLM's own default.
    if config.get("max_num_seqs") is not None:
        kwargs["max_num_seqs"] = config["max_num_seqs"]

    return kwargs


def evaluate_task(lm, task_key: str, num_fewshot: int,
                  log_samples: bool) -> dict | None:
    """Evaluate one task and return its record. None if the key is unknown."""
    task_config = resolve_task_config(task_key)
    if task_config is None:
        print(f"Unknown task: {task_key}, skipping")
        return None

    task_name = task_config["task_name"]
    task_fewshot = task_config.get("num_fewshot", num_fewshot)
    task_limit = task_config.get("limit", None)

    print(f"\n{'-'*40}")
    print(f"Running: {task_config['short_name']}")
    print(f"Task: {task_name}, Few-shot: {task_fewshot}"
          + (f", Limit: {task_limit}" if task_limit else ""))
    print(f"{'-'*40}")

    eval_start = time.time()
    try:
        # HumanEval (and any other task that opts in) executes model-generated
        # Python locally to check it. lm_eval gates that behind both
        # confirm_run_unsafe_code and the HF_ALLOW_CODE_EVAL env var.
        unsafe = task_config.get("unsafe_code", False)
        if unsafe:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("  [!] This task executes model-generated code locally.")
        else:
            os.environ.pop("HF_ALLOW_CODE_EVAL", None)

        eval_results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=task_fewshot,
            log_samples=log_samples,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42,
            limit=task_limit,
            confirm_run_unsafe_code=unsafe,
        )
        eval_time = time.time() - eval_start

        # A metric may carry a filter ("exact_match,strict-match"); one that
        # does not is stored by lm_eval under "<metric>,none".
        task_results = eval_results["results"].get(task_name, {})
        metrics = {}
        for metric in task_config["metrics"]:
            key = metric if "," in metric else f"{metric},none"
            if key in task_results:
                metrics[metric.split(",")[0]] = task_results[key]

        record = {
            "task_name": task_name,
            "short_name": task_config["short_name"],
            "metrics": metrics,
            "eval_time": eval_time,
            "num_fewshot": task_fewshot,
        }
        if log_samples and "samples" in eval_results:
            record["samples"] = eval_results["samples"].get(task_name, [])

        print(f"  Time: {format_time(eval_time)}")
        for metric, value in metrics.items():
            shown = f"{value:.4f}" if isinstance(value, float) else value
            print(f"  {metric}: {shown}")

        return record

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return {
            "task_name": task_name,
            "error": str(e),
            "eval_time": time.time() - eval_start,
        }


def _best_effort(what: str, action) -> None:
    """Run a teardown step; report a failure rather than raising."""
    try:
        action()
    except Exception as e:
        print(f"  - Warning: {what} failed: {e}")
    sys.stdout.flush()


def shutdown_vllm_engine(lm) -> None:
    """Signal the vLLM engine to exit, and give it a moment to act on that.

    vLLM V1 runs its engine in child processes watched by a monitor thread that
    reports an unsignalled exit as a crash. So the engine core is shut down
    through its own API first; only then is the model dropped.
    """
    print("\nCleaning up resources...")
    sys.stdout.flush()

    def shutdown():
        print("  - Shutting down vLLM engine...")
        # lm_eval's wrapper keeps the vllm.LLM in .model
        llm = getattr(lm, "model", None)
        engine = getattr(llm, "llm_engine", None) if llm is not None else None
        if engine is None:
            return
        core = getattr(engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
            print("  - Engine core shutdown complete")
        elif hasattr(engine, "shutdown"):
            engine.shutdown()
            print("  - LLM engine shutdown complete")

    _best_effort("engine shutdown", shutdown)
    _best_effort("grace period", lambda: time.sleep(0.5))


def release_gpu() -> None:
    """Reclaim the GPU after the model reference has been dropped.

    A sweep runs every combination in one process, so each run has to hand the
    device back before the next model is built.
    """
    _best_effort("garbage collection", gc.collect)

    def reap_children():
        children = multiprocessing.active_children()
        if not children:
            return
        print(f"  - Waiting for {len(children)} child processes to exit...")
        for child in children:
            child.join(timeout=5)
        # engine shutdown should have brought them down; force whatever is left
        alive = [c for c in children if c.is_alive()]
        if alive:
            print(f"  - Warning: terminating {len(alive)} stuck processes")
            for child in alive:
                child.terminate()
                child.join(timeout=2)
        print("  - Child processes exited")

    _best_effort("reaping child processes", reap_children)

    def destroy_process_group():
        if dist.is_initialized():
            dist.destroy_process_group()
            print("  - Process group destroyed")

    _best_effort("destroying the process group", destroy_process_group)

    def empty_cuda_cache():
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("  - CUDA cache emptied")

    _best_effort("emptying the CUDA cache", empty_cuda_cache)
    _best_effort("final garbage collection", gc.collect)

    print("Cleanup complete")
    sys.stdout.flush()


def run_benchmark(
    model_name: str,
    task_keys: list[str],
    num_fewshot: int = 0,
    vllm_config: dict = None,
    log_samples: bool = False,
) -> dict:
    """Load the model, evaluate every task on it, then release the GPU."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")

    config = {**VLLM_CONFIG, **(vllm_config or {})}

    load_start = time.time()
    lm = VLLM(**build_vllm_kwargs(model_name, config))
    load_time = time.time() - load_start
    print(f"Model loaded in {format_time(load_time)}")

    results = {
        "model": model_name,
        "load_time": load_time,
        "tasks": {},
        "total_eval_time": 0,
    }

    for task_key in task_keys:
        record = evaluate_task(lm, task_key, num_fewshot, log_samples)
        if record is None:
            continue
        results["tasks"][task_key] = record
        results["total_eval_time"] += record["eval_time"]

    shutdown_vllm_engine(lm)
    del lm              # drop the last reference before reclaiming the device
    release_gpu()

    return results



def print_results_table(results: dict):
    """Print results in a table format."""
    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Task':<15} {'Metric':<15} {'Value':<10} {'Time':<10}")
    print("-" * 70)

    for task_key, task_data in results.get("tasks", {}).items():
        if "error" in task_data:
            print(f"{task_data.get('short_name', task_key):<15} {'ERROR':<15} {task_data['error'][:30]}")
            continue

        metrics = task_data.get("metrics", {})
        eval_time = task_data.get("eval_time", 0)

        for i, (metric, value) in enumerate(metrics.items()):
            task_label = task_data.get("short_name", task_key) if i == 0 else ""
            time_str = format_time(eval_time) if i == 0 else ""
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"{task_label:<15} {metric:<15} {value_str:<10} {time_str:<10}")

    print("-" * 70)
    print(f"Total eval time: {format_time(results.get('total_eval_time', 0))}")
    print("=" * 70)


def write_json_atomic(path: Path, data) -> None:
    """Write JSON via a temp file + rename, so a crash never leaves a partial file."""
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)


def run_single_combination(
    config: dict,
    combo: dict,
    run_name: str,
    output_path: Path,
) -> bool:
    """Run a single parameter combination."""
    # The sweep parameters override the base emulation settings.
    emu_config = config.get("mma_emu", {}).copy()
    for key, value in combo.items():
        emu_config[key] = value

    export_mma_emu_to_env(emu_config)

    eval_config = config.get("eval", {})
    task_keys = eval_config.get("tasks", ["wikitext"])
    model_name = eval_config.get("model", _pconfig["default_model"])
    log_samples = eval_config.get("log_samples", False)

    vllm_config = config.get("vllm", {})

    print("\n" + "=" * 60)
    print(f"{_pconfig['label']} Experiment: {run_name}")
    print("=" * 60)
    print(f"Model:  {model_name}")
    print(f"Tasks:  {', '.join(task_keys)}")
    print(f"MMA:    {format_mma_emu_settings(emu_config)}")

    merged_vllm_config = {**VLLM_CONFIG, **vllm_config}
    print_vllm_settings(merged_vllm_config, indent="        ")

    total_start = time.time()
    try:
        results = run_benchmark(model_name, task_keys, vllm_config=vllm_config, log_samples=log_samples)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR during benchmark: {e}")
        print(f"{'='*60}")
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False

    total_time = time.time() - total_start

    results["run_name"] = run_name
    results["precision"] = _precision
    results["sweep_params"] = combo
    results["device"] = get_device_info()
    results["timestamp"] = datetime.now().isoformat()
    results["total_time"] = total_time
    results["mma_emu"] = emu_config

    try:
        print_results_table(results)
        sys.stdout.flush()
    except Exception as e:
        print(f"\nError printing results table: {e}")
        sys.stdout.flush()

    print(f"\nTotal benchmark time: {format_time(total_time)}")
    sys.stdout.flush()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract samples to separate file before saving main results
        if log_samples:
            samples_data = {}
            for task_key, task_data in results.get("tasks", {}).items():
                if "samples" in task_data:
                    samples_data[task_key] = task_data.pop("samples")
            if samples_data:
                samples_path = output_path.parent / f"{output_path.stem}_samples.json"
                write_json_atomic(samples_path, samples_data)
                print(f"\nSamples saved to: {samples_path}")
                sys.stdout.flush()

        # Written last: its presence is what marks the run as complete.
        write_json_atomic(output_path, results)
        print(f"\nResults saved to: {output_path}")
        sys.stdout.flush()
    except Exception as e:
        print(f"\nError saving results: {e}")
        sys.stdout.flush()
        return False

    return True


def cleanup_multiprocessing():
    """Clean up multiprocessing resources to prevent leaked semaphore warnings."""
    gc.collect()

    try:
        for child in multiprocessing.active_children():
            child.terminate()
            child.join(timeout=5)
    except Exception:
        pass

    gc.collect()


def main():
    # Results land straight in results/<run_name>.json — flat and easy to find.
    # A re-run skips combinations that already have a complete result (resume);
    # --overwrite re-runs them. The run's own timestamp is kept inside the JSON.
    output_dir = exp_path / "results"

    # Decide up front which combinations still need to run.
    plan = []
    for combo in _combinations:
        run_name = get_run_name(combo, _config)
        output_path = output_dir / f"{run_name}.json"
        status = (None if _args.overwrite
                  else completed_result_status(output_path, combo, _config))
        # Re-run anything not complete; an existing but stale result is reported.
        skip = not _args.overwrite and status is None
        plan.append((combo, run_name, output_path, skip, status))
    n_skip = sum(skip for *_, skip, _ in plan)

    exp_name = _config.get("experiment", {}).get("name", exp_path.name)
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Precision: {_pconfig['label']}")
    print(f"Output: {output_dir}")
    print(f"Total runs: {len(_combinations)}")
    if _args.overwrite:
        print("Mode: --overwrite (re-running every combination)")
    else:
        print(f"Resume: {n_skip} already complete, "
              f"{len(plan) - n_skip} to run (pass --overwrite to re-run all)")
    print(f"{'='*60}")
    sys.stdout.flush()

    run_results = []
    for combo, run_name, output_path, skip, status in plan:
        if skip:
            print(f"\n[SKIP] {run_name}: complete result exists -> {output_path.name}")
            sys.stdout.flush()
            run_results.append((run_name, "SKIPPED", output_path))
            continue
        if not _args.overwrite and status != "no result yet":
            print(f"\n[RERUN] {run_name}: existing result not reusable ({status})")
            sys.stdout.flush()

        try:
            success = run_single_combination(_config, combo, run_name, output_path)
            run_results.append((run_name, "OK" if success else "FAILED", output_path))
        except Exception as e:
            print(f"\nERROR running {run_name}: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            run_results.append((run_name, "FAILED", output_path))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    sys.stdout.flush()

    for run_name, status, output_path in run_results:
        print(f"  [{status}] {run_name} -> {output_path.name}")
        sys.stdout.flush()

    print(f"\nResults saved to: {output_dir}")
    sys.stdout.flush()

    print("\nExperiment complete. Exiting...")
    sys.stdout.flush()

    # Clean up multiprocessing resources before exit
    cleanup_multiprocessing()

    # Force clean exit to avoid any remaining cleanup issues
    os._exit(0)


if __name__ == "__main__":
    main()
