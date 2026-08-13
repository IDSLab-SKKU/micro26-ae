#!/usr/bin/env bash
#
# Optional: build vLLM + the MMA-Emu kernels from source with uv.
# The prebuilt Docker image (./run-docker.sh) is the default path; use this only
# to compile the artifact yourself on the host.
#
# Run from artifact/:  ./build.sh
#
# Host prerequisites:
#   - CUDA Toolkit 12.8 (nvcc on PATH)
#   - a C++ compiler (g++ 10 or newer)
#   - Python 3.12 and uv  (https://docs.astral.sh/uv/)
#
# The compile takes ~30 min. Set MAX_JOBS to cap parallel compile jobs if the
# build runs out of memory (e.g. MAX_JOBS=4 ./build.sh).
set -euo pipefail

# vLLM builds from the repo root (setup.py, requirements/, csrc/); run there.
cd "$(dirname "$0")/.."
ROOT="$PWD"

# Only the architectures the artifact runs on — Hopper (SM90), Blackwell
# (SM100 data-center, SM120 workstation) — and skip FlashAttention-3 (unused
# here).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0 12.0}"
export VLLM_BUILD_FA3="${VLLM_BUILD_FA3:-0}"

CU="https://download.pytorch.org/whl/cu128"

echo ">> [1/5] create the Python 3.12 virtual environment (.venv)"
uv venv --python 3.12

echo ">> [2/5] install PyTorch 2.8.0 (CUDA 12.8 build)"
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$CU"

echo ">> [3/5] install build + runtime dependencies"
uv pip install -r requirements/build.txt
uv pip install -r requirements/cuda.txt --extra-index-url "$CU"

echo ">> [4/5] compile & install vLLM + MMA-Emu kernels (~30 min)"
uv pip install -e . --no-build-isolation

echo ">> [5/5] install the evaluation stack (pinned)"
uv pip install lm_eval==0.4.9.1 transformers==4.55.2 matplotlib

echo
echo "Build complete. Activate the environment and verify:"
echo
echo "  source $ROOT/.venv/bin/activate"
echo "  python3 -c \"import vllm; print('vLLM', vllm.__version__)\""
echo
echo "Then run any experiment without Docker, e.g.:"
echo
echo "  cd $ROOT/artifact/exp2_figure6a_fp8_cofda && ./run_figure6a.sh"
