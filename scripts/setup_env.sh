#!/usr/bin/env bash
# Create the conda env used by lowvram/run_edge.py.
set -euo pipefail
ENV_NAME="${1:-cosmosim}"
conda create -y -n "$ENV_NAME" python=3.12
conda run -n "$ENV_NAME" pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n "$ENV_NAME" pip install -r "$(dirname "$0")/../requirements.txt"
