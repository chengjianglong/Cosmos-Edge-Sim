#!/usr/bin/env bash
# Download nvidia/Cosmos3-Edge (~8.6 GB) into checkpoints/Cosmos3-Edge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
hf download nvidia/Cosmos3-Edge --local-dir "$ROOT/checkpoints/Cosmos3-Edge"
