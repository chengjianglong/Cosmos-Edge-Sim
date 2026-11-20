#!/usr/bin/env python
"""Build a reduced Sim-Auto request from NVIDIA's published daytime sample.

    python lowvram/simauto/prepare_request.py --views camera_front_wide_120fov --num-frames 49 \
        --mode camera -o outputs/sim_auto/requests/front_49f.json

Uses the cosmos-framework skill helpers (fetch_sample / native_request), then trims the
camera list, horizon and resolution so the request fits two 6 GB GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "third_party/cosmos-framework/.agents/skills/cosmos3-inference-simulation-auto/scripts"
sys.path.insert(0, str(SKILL))

from helpers.daytime_sample import fetch_sample, native_request  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="+", default=["camera_front_wide_120fov"])
    ap.add_argument("--num-frames", type=int, default=49, help="must be 4k+1; the sample has 300 frames")
    ap.add_argument("--resolution", choices=["480", "720"], default="480")
    ap.add_argument("--mode", choices=["camera", "text2video", "joint"], default="camera")
    ap.add_argument("--caption-folder", default="captions_with_timestamp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    if (args.num_frames - 1) % 4:
        raise SystemExit("--num-frames must be 4k+1")

    root = fetch_sample()
    source = "input-sim-auto.jsonl" if (root / "input-sim-auto.jsonl").is_file() else "daytime_scene/sim_auto_joint.json"
    request_path = root / source
    if request_path.suffix == ".jsonl":
        # one JSON request per line; take the first
        tmp = args.output.with_suffix(".source.json")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(request_path.read_text().splitlines()[0])
        # keep relative paths resolvable from the sample root
        request = json.loads(tmp.read_text())
        tmp.unlink()
        resolved = root / "_lowvram_request.json"
        resolved.write_text(json.dumps(request))
        request_path = resolved

    request = native_request(request_path, mode=args.mode, caption_folder=args.caption_folder)
    views = request["multiview"]["views"]
    keep = [v for v in views if v["camera_key"] in args.views]
    if len(keep) != len(args.views):
        raise SystemExit(f"unknown view(s); available: {[v['camera_key'] for v in views]}")
    request["multiview"]["views"] = keep
    if "prompt_from_camera" in request and request["prompt_from_camera"] not in args.views:
        request["prompt_from_camera"] = args.views[0]
    request["num_frames"] = args.num_frames
    if "num_video_frames_per_chunk" in request:
        request["num_video_frames_per_chunk"] = args.num_frames
    request["resolution"] = args.resolution
    request["aspect_ratio"] = "16,9"
    request["seed"] = args.seed
    request.update(guidance=1, control_guidance=1, normalize_cfg=False, autoregressive=True)
    for field in ("num_steps", "shift", "per_view_negative_prompt", "negative_prompt"):
        request.pop(field, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2) + "\n")
    print(f"{len(keep)} view(s), {args.num_frames} frames, {args.resolution}p, mode={args.mode} -> {args.output}")


if __name__ == "__main__":
    main()
