#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install-backend.sh <backend> [output-name]

Backends:
  vulkan
  cuda
  metal
  cpu

Examples:
  scripts/install-backend.sh vulkan
  scripts/install-backend.sh cuda
  scripts/install-backend.sh vulkan smelt
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

backend="$1"
output_name="${2:-smelt-$backend}"

case "$backend" in
  vulkan|cuda|metal)
    feature="$backend"
    ;;
  cpu)
    feature=""
    ;;
  *)
    echo "unknown backend: $backend" >&2
    usage >&2
    exit 2
    ;;
esac

build_args=(build --release)
if [[ -n "$feature" ]]; then
  build_args+=(--features "$feature")
fi

cargo "${build_args[@]}"
install -Dm755 target/release/smelt "$HOME/.local/bin/$output_name"

echo "installed $HOME/.local/bin/$output_name"
