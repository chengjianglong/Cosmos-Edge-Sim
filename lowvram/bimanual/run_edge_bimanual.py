#!/usr/bin/env python
"""Bimanual robot simulation with Cosmos3-Edge forward dynamics on one 6 GB GPU.

A lightweight counterpart to nvidia/Cosmos3-Nano-Sim-Bimanual: given the robot head-camera image
and an AgiBot action sequence, roll the scene forward chunk by chunk. Each chunk feeds 16 actions
(1.6 s at 10 fps) and the last generated frame of the previous chunk to Cosmos3-Edge's
``forward_dynamics`` mode (embodiment ``agibotworld``), exactly like the Edge model card's
autoregressive UMI example.

    # 1) convert Sim-Bimanual actions to the base model's 29D agibotworld format (framework venv)
    third_party/cosmos-framework/.venv/bin/python lowvram/bimanual/convert_actions.py \
        $ASSETS/assets/training/agibot/actions/agibot_006_normalized_unifiedv1.json -o outputs/bimanual/agibot_006_edge29d.json
    # 2) roll out (cosmosim env)
    python lowvram/bimanual/run_edge_bimanual.py --gpu 0 \
        --image $ASSETS/assets/training/agibot/images/agibot_006_initial.png \
        --actions outputs/bimanual/agibot_006_edge29d.json -o outputs/bimanual/agibot_006_edge.mp4

Differences from Sim-Bimanual (16B, post-trained for this task, KV-cached history): Edge is the 4B
base model, conditions each chunk only on the previous chunk's last frame, and runs at 10 fps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lowvram"))
from run_edge import DEFAULT_CKPT, report_memory  # noqa: E402

DEFAULT_TASK = "The robot uses both arms to manipulate the objects on the table"


def build_pipeline(ckpt: Path, device: torch.device, offload_blocks: int) -> Cosmos3OmniPipeline:
    """Transformer layers streamed from pinned CPU RAM; the VAE stays on the GPU.

    A 16-step chunk is only 17 frames (5 latent frames), so the Wan VAE (~1.4 GB) fits next to the
    streamed transformer (~2 GB peak) on a 6 GB card, and the pipeline can encode/decode directly.
    """
    transformer = Cosmos3OmniTransformer.from_pretrained(ckpt, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.enable_group_offload(
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=offload_blocks,
        use_stream=True,
    )
    pipe = Cosmos3OmniPipeline.from_pretrained(
        ckpt, transformer=transformer, torch_dtype=torch.bfloat16, enable_safety_checker=False
    )
    pipe._get_execution_device = lambda: device
    pipe.vae.to(device)
    pipe.vae.enable_tiling()
    return pipe


def to_frames(video) -> list[np.ndarray]:
    """Pipeline output (list of PIL / [T,H,W,C] float array) -> list of uint8 HxWx3."""
    if isinstance(video, list) and video and isinstance(video[0], list):
        video = video[0]
    frames = []
    for f in video:
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
        frames.append(arr)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="initial head-camera frame")
    ap.add_argument("--actions", type=Path, required=True, help="output of convert_actions.py")
    ap.add_argument("--task", default=DEFAULT_TASK, help="short action description for the caption")
    ap.add_argument("--chunks", type=int, default=0, help="number of 16-step chunks to roll out (0 = all)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--flow-shift", type=float, default=10.0)
    ap.add_argument("--resolution-tier", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--offload-blocks", type=int, default=2)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.actions.read_text())
    chunks = np.asarray(spec["action_chunks"], dtype=np.float32)  # [N,16,29]
    if args.chunks:
        chunks = chunks[: args.chunks]
    fps = float(spec["fps"])
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    t0 = time.time()
    pipe = build_pipeline(args.ckpt, device, args.offload_blocks)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )
    print(f"[time] load {time.time() - t0:.1f}s; {len(chunks)} chunks x {chunks.shape[1]} actions at {fps:g} fps", flush=True)

    image = load_image(str(args.image)).convert("RGB")
    frames: list[np.ndarray] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        t1 = time.time()
        result = pipe(
            prompt=args.task,
            action=CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=int(chunk.shape[0]),
                domain_name=spec["domain_name"],
                resolution_tier=args.resolution_tier,
                raw_actions=torch.from_numpy(chunk),
                image=image,
                view_point="ego_view",
            ),
            fps=fps,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            use_system_prompt=False,
            generator=torch.Generator(device=device).manual_seed(args.seed + i),
            output_type="np",
        )
        out = to_frames(result.video)
        frames.extend(out if i == 0 else out[1:])  # later chunks start with the frame we fed in
        image = Image.fromarray(out[-1])
        print(f"[chunk {i + 1}/{len(chunks)}] {len(out)} frames {out[0].shape[1]}x{out[0].shape[0]} in {time.time() - t1:.1f}s", flush=True)
        export_to_video([f.astype(np.float32) / 255 for f in frames], str(args.output), fps=int(fps), macro_block_size=1)
    report_memory("rollout", device)
    print(f"[done] {len(frames)} frames ({len(frames) / fps:.1f} s at {fps:g} fps) -> {args.output} (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
