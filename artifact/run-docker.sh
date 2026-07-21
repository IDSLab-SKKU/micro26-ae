#!/usr/bin/env bash
# Start the artifact container (or re-enter it) and drop into a shell.
# Inside, see README.md for what to run.

set -euo pipefail

IMAGE="${MICRO26_AE_IMAGE:-docker.io/jongyeop1999/micro26-ae:v1}"
CONTAINER="${MICRO26_AE_CONTAINER:-micro26-ae}"
HF_VOLUME="${MICRO26_AE_HF_VOLUME:-micro26-ae-hf}"

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$ARTIFACT_DIR")"

# Re-enter an existing container so its downloads persist.
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Re-entering existing container '$CONTAINER'."
    exec docker start --attach --interactive "$CONTAINER"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Pulling $IMAGE"
    docker pull "$IMAGE"
fi

# Models go to a named volume (not your HF cache), surviving `docker rm`; csrc is
# mounted read-only so the artifact/kernels symlink resolves inside the container.
exec docker run --gpus all -it \
    --name "$CONTAINER" \
    --ipc=host \
    -v "$ARTIFACT_DIR":/workspace/artifact \
    -v "$REPO_DIR/csrc":/workspace/csrc:ro \
    -v "$HF_VOLUME":/root/.cache/huggingface \
    "$IMAGE"
