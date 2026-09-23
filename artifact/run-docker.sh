#!/usr/bin/env bash
# Start the artifact container (or re-enter it) and drop into a shell.
# Inside, see README.md for what to run.
#
# The container runs as YOU (your uid:gid), not root, so everything it writes
# into artifact/ — results, figures — is yours to edit or delete without sudo.
# Downloads (models, datasets, compile caches) go to a cache directory on the
# host that you own, so they survive `docker rm`:
#   ${MICRO26_AE_CACHE_DIR:-~/.cache/micro26-ae}
# The container also takes the host's timezone, so timestamps line up with
# host-side logs.

set -euo pipefail

IMAGE="${MICRO26_AE_IMAGE:-docker.io/jongyeop1999/micro26-ae:v1}"
CONTAINER="${MICRO26_AE_CONTAINER:-micro26-ae}"
CACHE_DIR="${MICRO26_AE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/micro26-ae}"

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$ARTIFACT_DIR")"

# Re-enter an existing container so its state persists. Its flags were fixed
# when it was created, so one made by an older run-docker.sh still runs as root.
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    if [[ -z "$(docker container inspect -f '{{.Config.User}}' "$CONTAINER")" ]]; then
        echo "NOTE: '$CONTAINER' was created by an older run-docker.sh and runs as root," >&2
        echo "      so its outputs are root-owned. To recreate it: docker rm $CONTAINER" >&2
    fi
    echo "Re-entering existing container '$CONTAINER'."
    exec docker start --attach --interactive "$CONTAINER"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Pulling $IMAGE"
    docker pull "$IMAGE"
fi

# The host's timezone as an IANA name (e.g. Asia/Seoul); UTC if it cannot be
# determined. The image itself is set to America/Los_Angeles.
host_tz() {
    if [[ -n "${TZ:-}" ]]; then
        echo "$TZ"
    elif command -v timedatectl >/dev/null 2>&1 \
            && tz="$(timedatectl show -p Timezone --value 2>/dev/null)" && [[ -n "$tz" ]]; then
        echo "$tz"
    elif [[ -L /etc/localtime ]]; then
        readlink /etc/localtime | sed 's#.*/zoneinfo/##'
    elif [[ -s /etc/timezone ]]; then
        cat /etc/timezone
    else
        echo UTC
    fi
}

# Created here, before docker would create it as root.
mkdir -p "$CACHE_DIR/home"

# Your uid has no /etc/passwd entry in the image, so USER/LOGNAME name it for
# getpass.getuser() (torch's compile cache needs it). csrc is mounted read-only
# so the artifact/kernels symlink resolves inside the container.
# HF_HUB_DISABLE_XET=1 downloads through the classic Hugging Face CDN: on
# networks that block the Xet backend, the safetensors shards otherwise stall at
# 0 bytes with no error. Export HF_HUB_DISABLE_XET=0 to use Xet.
exec docker run --gpus all -it \
    --name "$CONTAINER" \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e HOME=/cache/home \
    -e XDG_CACHE_HOME=/cache \
    -e HF_HOME=/cache/huggingface \
    -e TZ="$(host_tz)" \
    -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    -v "$ARTIFACT_DIR":/workspace/artifact \
    -v "$REPO_DIR/csrc":/workspace/csrc:ro \
    -v "$CACHE_DIR":/cache \
    "$IMAGE"
