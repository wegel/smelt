#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/build-container.sh <backend> [output-path]

Backends:
  vulkan
  cuda

Examples:
  bash scripts/build-container.sh vulkan
  bash scripts/build-container.sh cuda dist/custom-smelt-cuda
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

backend="$1"
case "$backend" in
  vulkan|cuda)
    ;;
  *)
    echo "unknown backend: $backend" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
  container_runtime="$CONTAINER_RUNTIME"
elif command -v podman >/dev/null 2>&1; then
  container_runtime=podman
elif command -v docker >/dev/null 2>&1; then
  container_runtime=docker
else
  echo "neither podman nor docker is installed" >&2
  exit 1
fi

case "$container_runtime" in
  podman|docker)
    ;;
  *)
    echo "unsupported container runtime: $container_runtime" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
containerfile="$repo_root/Containerfile.$backend"
artifact_name="smelt-$backend"
output_path="${2:-$repo_root/dist/$artifact_name}"
image_tag="smelt-build-$backend:local"

mkdir -p "$(dirname "$output_path")"

"$container_runtime" build \
  --file "$containerfile" \
  --target artifact \
  --tag "$image_tag" \
  "$repo_root"

container_id=$("$container_runtime" create "$image_tag")
cleanup() {
  "$container_runtime" rm -f "$container_id" >/dev/null
}
trap cleanup EXIT

"$container_runtime" cp "$container_id:/$artifact_name" "$output_path"
chmod 755 "$output_path"

echo "built $output_path"
