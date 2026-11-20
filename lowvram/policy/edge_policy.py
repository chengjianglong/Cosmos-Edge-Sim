"""Cosmos3-Edge-Policy-DROID on one 6 GB GPU.

Wraps cosmos-framework's own RoboLab policy service (``action_policy_server_robolab``), so request
building, the action/state transform, sampling and output formatting are exactly NVIDIA's. Only the
model setup changes:

* the decoder-layer linears stay in pinned CPU RAM as exact bf16 (~6 GB) and stream to the GPU one
  layer ahead of use (``lowvram/simauto/int8_stream.py`` with ``LOWVRAM_STREAM_DTYPE=bf16``; the
  framework's own ``enable_layerwise_offload`` refuses this checkpoint because Edge uses
  generation-specific text K normalization);
* no torch.compile / CUDA graphs / diffusion cache, and a single GPU (no CP / CFG parallel).

``EdgePolicy.infer(obs)`` takes the RoboLab/openpi observation dict and returns
``{"action": [32, 8] joint positions + gripper, "video": [33, H, W, 3] imagined future}``.
``serve()`` exposes the same thing over the RoboLab websocket protocol, so a RoboLab client on a
bigger machine can drive this GPU.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Must be set before anything initializes CUDA (the framework import does): without expandable
# segments the VAE decode of the second chunk fails on ~0.9 GB of fragmented cache.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("LOWVRAM_STREAM_DTYPE", "bf16")
sys.path.insert(0, str(ROOT / "lowvram" / "simauto"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_CKPT = ROOT / "checkpoints" / "Cosmos3-Edge-Policy-DROID"

LOW_VRAM_SETUP = {
    "use_torch_compile": False,
    "use_cuda_graphs": False,
    "diffusion_cache": False,
    "cp_size": 1,
    "cfgp_size": 1,
    "dp_shard_size": 1,
    "dp_replicate_size": 1,
}

# Settings from cosmos-framework docs/action_policy_droid_server.md for the Edge policy.
EDGE_POLICY_ARGS = {
    "format_prompt_as_json": True,
    "guidance_interval": (960, 1001),
}


def _install_low_vram_setup() -> Any:
    from cosmos_framework.scripts import action_policy_server_robolab as srv

    real = srv.OmniSetupOverrides
    if getattr(real, "_lowvram_proxy", False):
        return srv
    import run_sim_auto  # weight streaming hook on Cosmos3OmniModel.from_pretrained_dcp

    run_sim_auto.install_patches()

    class LowVramSetupOverrides:
        _lowvram_proxy = True
        CHECKPOINTS = real.CHECKPOINTS

        @staticmethod
        def model_validate(values: dict[str, Any]) -> Any:
            return real.model_validate({**values, **LOW_VRAM_SETUP})

    srv.OmniSetupOverrides = LowVramSetupOverrides
    return srv


def server_args(checkpoint: Path = DEFAULT_CKPT, **overrides: Any) -> Any:
    srv = _install_low_vram_setup()
    return srv.RobolabServerArgs(checkpoint_path=str(checkpoint), **{**EDGE_POLICY_ARGS, **overrides})


class EdgePolicy:
    def __init__(self, checkpoint: Path = DEFAULT_CKPT, *, decode_video: bool = True, seed: int = 0, **overrides: Any):
        srv = _install_low_vram_setup()
        t0 = time.time()
        self.service = srv.RobolabPolicyService(
            server_args(checkpoint, decode_video=decode_video, seed=seed, deterministic_seed=True, **overrides)
        )
        self.cfg = self.service.cfg
        print(f"[policy] loaded in {time.time() - t0:.0f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = self.service.infer(obs)
        torch.cuda.empty_cache()
        out["seconds"] = time.time() - t0
        out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        return out


def serve(checkpoint: Path = DEFAULT_CKPT, port: int = 8000, **overrides: Any) -> None:
    srv = _install_low_vram_setup()
    srv.serve(server_args(checkpoint, port=port, **overrides))
