#!/usr/bin/env python
"""Convert Cosmos3-Nano-Sim-Bimanual actions (59D ``normalized_unified_v1``) into the 29D
``agibotworld`` actions the Cosmos3 base models (Edge/Nano/Super) take for forward dynamics.

Run with the cosmos-framework venv (it uses the framework's own pose and normalization code):

    third_party/cosmos-framework/.venv/bin/python lowvram/bimanual/convert_actions.py \
        checkpoints/Cosmos3-Nano-Sim-Bimanual-assets/assets/training/agibot/actions/agibot_006_normalized_unifiedv1.json \
        -o outputs/bimanual/agibot_006_edge29d.json

The two action contracts describe the same AgiBot robot (both datasets use the framework's
AgiBotWorld forward kinematics; ``unified_v1`` is the 29D ``humanoid_pose_mapping`` layout scattered
into 59 slots), but differ in every encoding detail:

| | Sim-Bimanual (input) | base model ``agibotworld`` (output) |
|---|---|---|
| layout | ego9, R wrist9, R fingertips15, R gripper, L wrist9, L fingertips15, L gripper | head9, R wrist9, R open, L wrist9, L open |
| poses | ``backward_chunk_anchored_16f``: relative to the pose before each 16-row group | ``backward_framewise``: relative to the previous frame |
| gripper | metric width in meters (0 closed, 0.082 open) | open fraction (0 closed, 1 open) |
| rate | 30 fps | 10 fps |
| normalization | ``global_asinh`` (offset/scale from q01/q99, then asinh) | ``quantile`` (2*(a-q01)/(q99-q01)-1) |
| masking | per-row domain: ``agibotworld`` rows mask the ego slots, ``camera_pose`` rows mask the robot slots | none |

Steps: undo the asinh normalization; rebuild absolute head/wrist trajectories (a masked slot means
"did not move", so it holds the previous pose); subsample 30 -> 10 fps; re-encode as framewise
deltas plus open fractions; apply the base model's quantile normalization; split into 16-step chunks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.generator.action.utils.pose_utils import (
    _pose_vector_to_delta_transform,
    pose_abs_to_rel,
)

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / "third_party" / "cosmos-framework" / "cosmos_framework" / "data" / "generator" / "action"
UNIFIED_STATS = FRAMEWORK / "normalizers" / "global_asinh_v1_backward_anchored_gripper_metric.json"
AGIBOT_STATS = FRAMEWORK / "normalizer_stats" / "agibotworld_beta_lerobot_stats.json"

EGO, R_WRIST, R_GRIP, L_WRIST, L_GRIP = slice(0, 9), slice(9, 18), 33, slice(34, 43), 58
GRIPPER_OPEN_WIDTH_M = 0.082  # unified_v1 metric gripper reference (fully open)
ANCHOR_CHUNK = 16  # backward_chunk_anchored_16f


def denormalize_unified(action: np.ndarray) -> np.ndarray:
    """Invert global_asinh: raw = sinh(y * asinh(1)) * scale + offset."""
    stats = json.loads(UNIFIED_STATS.read_text())["global"]
    lo, hi = np.asarray(stats["q01"], np.float64), np.asarray(stats["q99"], np.float64)
    offset, scale = (hi + lo) / 2, np.maximum(hi - lo, 1e-8) / 2
    return np.sinh(action * math.asinh(1.0)) * scale + offset


def absolute_track(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rebuild absolute 4x4 poses [T+1] from chunk-anchored 9D deltas [T]; invalid rows hold still."""
    poses = [np.eye(4)]
    anchor = np.eye(4)
    for t, (vec, ok) in enumerate(zip(raw, valid)):
        if t % ANCHOR_CHUNK == 0:
            anchor = poses[-1]  # each 16-row group is relative to the pose just before it
        if ok:
            poses.append(anchor @ _pose_vector_to_delta_transform(vec, "rot6d", 1.0, normalize_rotation=True))
        else:
            poses.append(poses[-1])
    return np.stack(poses)


def convert(path: Path, src_fps: float, dst_fps: float, chunk: int) -> dict:
    data = json.loads(path.read_text())
    if data.get("action_space") != "normalized_unified_v1":
        raise SystemExit(f"{path}: expected action_space normalized_unified_v1, got {data.get('action_space')!r}")
    norm = np.asarray(data["action"], np.float64)  # [T,59]
    domains = data["domain_names"]
    domains = [domains] * len(norm) if isinstance(domains, str) else list(domains)
    unknown = sorted(set(domains) - {"agibotworld", "agibot_gear_gripper", "agibot_gear_gripper_ext", "camera_pose"})
    if unknown:
        raise SystemExit(f"{path}: rows routed to {unknown} have no agibotworld equivalent")
    robot_rows = np.array([d != "camera_pose" for d in domains])
    raw = denormalize_unified(norm)

    head = absolute_track(raw[:, EGO], ~robot_rows)  # camera_pose rows carry the head motion
    right = absolute_track(raw[:, R_WRIST], robot_rows)
    left = absolute_track(raw[:, L_WRIST], robot_rows)

    # Gripper state per target frame (row t describes frame t+1); hold through masked rows.
    def openness(col: int) -> np.ndarray:
        values, last = [], None
        for t in range(len(norm)):
            if robot_rows[t]:
                last = float(np.clip(raw[t, col] / GRIPPER_OPEN_WIDTH_M, 0.0, 1.0))
            values.append(last)
        first = next((v for v in values if v is not None), 1.0)
        return np.array([first] + [first if v is None else v for v in values])  # [T+1], frame-indexed

    r_open, l_open = openness(R_GRIP), openness(L_GRIP)

    stride = src_fps / dst_fps
    frames = np.round(np.arange(0, len(norm) + 1e-9, stride)).astype(int)
    frames = frames[frames <= len(norm)]
    native = np.concatenate(
        [
            pose_abs_to_rel(head[frames], "rot6d", "backward_framewise"),
            pose_abs_to_rel(right[frames], "rot6d", "backward_framewise"),
            r_open[frames[1:], None],
            pose_abs_to_rel(left[frames], "rot6d", "backward_framewise"),
            l_open[frames[1:], None],
        ],
        axis=1,
    )  # [T_dst,29]

    stats = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in load_action_stats(str(AGIBOT_STATS)).items()}
    normalized = normalize_action(torch.as_tensor(native, dtype=torch.float64), "quantile", stats).numpy()
    # Dimensions the training data never moved (q01 == q99, e.g. the head x-translation) map to
    # (x - q01) / 1e-8; float noise there would explode, so pin them to the training value (-1).
    lo, hi = stats["q01"].numpy(), stats["q99"].numpy()
    frozen = np.abs(hi - lo) < 1e-6
    normalized[:, frozen] = -1.0
    clipped = float(np.mean(np.abs(normalized) > 1.0))
    normalized = np.clip(normalized, -1.0, 1.0)  # the quantile normalizer clamps during training

    n_chunks = len(normalized) // chunk
    chunks = normalized[: n_chunks * chunk].reshape(n_chunks, chunk, 29)
    return {
        "source": str(path),
        "domain_name": "agibotworld",
        "action_space": "agibotworld_framewise_quantile",
        "fps": dst_fps,
        "action_chunk_size": chunk,
        "num_chunks": n_chunks,
        "layout": "head9, right_wrist9, right_open, left_wrist9, left_open (rot6d, framewise deltas)",
        "fraction_clipped": clipped,
        "robot_rows": int(robot_rows.sum()),
        "camera_rows": int((~robot_rows).sum()),
        "action_chunks": chunks.round(6).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("actions", type=Path, help="Sim-Bimanual *_normalized_unifiedv1.json")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--src-fps", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=10.0, help="base-model AgiBot training rate")
    ap.add_argument("--chunk", type=int, default=16)
    args = ap.parse_args()
    out = convert(args.actions, args.src_fps, args.fps, args.chunk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out) + "\n")
    print(
        f"{args.actions.name}: {out['robot_rows']} robot + {out['camera_rows']} camera rows -> "
        f"{out['num_chunks']} chunks x {args.chunk} steps at {args.fps:g} fps "
        f"({out['num_chunks'] * args.chunk / args.fps:.1f} s); {100 * out['fraction_clipped']:.1f}% of values clipped"
        f" -> {args.output}"
    )


if __name__ == "__main__":
    main()
