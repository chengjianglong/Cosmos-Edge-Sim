#!/usr/bin/env python
"""Run Cosmos3-Edge-Policy-DROID on one 6 GB GPU against DROID episodes.

    # open loop: at every 32-frame step of a recorded episode, predict the next 32 joint targets
    # from the real cameras + robot state and compare them with what the teleoperator did
    python lowvram/policy/droid_sim.py eval --episode 3 --chunks 6

    # closed loop in imagination: the policy's own predicted video and joints become its next
    # observation (the "world action model" acts as its own simulator)
    python lowvram/policy/droid_sim.py dream --episode 3 --chunks 6

    # RoboLab/openpi websocket server (same protocol as cosmos_framework's policy server)
    python lowvram/policy/droid_sim.py serve --port 8000

Run with the cosmos-framework venv. Choose the GPU with CUDA_VISIBLE_DEVICES.
Data: one shard of nvidia/Cosmos3-DROID (see README) in checkpoints/droid_plus_lerobot_640x360_20260412.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_policy import DEFAULT_CKPT, ROOT, EdgePolicy, serve  # noqa: E402

DATA = ROOT / "checkpoints" / "droid_plus_lerobot_640x360_20260412" / "success"
CAMERAS = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}
FPS = 15
CHUNK = 32


class DroidEpisode:
    """One episode of the LeRobot v3 Cosmos3-DROID dataset: frames, joint states and actions."""

    def __init__(self, index: int, root: Path = DATA) -> None:
        episodes = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
        row = episodes[episodes.episode_index == index]
        if row.empty:
            raise SystemExit(f"episode {index} is not in the downloaded metadata shard")
        row = row.iloc[0]
        self.index = index
        self.length = int(row["length"])
        self.task = str(list(row["tasks"])[0]).split(" | ")[0]
        data_file = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
        table = pq.read_table(data_file, filters=[("episode_index", "=", index)]).to_pandas().sort_values("frame_index")
        col = lambda name: np.stack(table[name].to_numpy()).astype(np.float32)  # noqa: E731
        self.joints = col("observation.state.joint_positions")  # [T,7]
        self.gripper = col("observation.state.gripper_position").reshape(-1)  # [T], 1 = closed
        self.action_joints = col("action.joint_position")  # [T,7]
        self.action_gripper = col("action.gripper_position").reshape(-1)  # [T]
        self.frames: dict[str, np.ndarray] = {}
        for name, key in CAMERAS.items():
            path = root / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            self.frames[name] = _decode(path, float(row[f"videos/{key}/from_timestamp"]), self.length)

    def observation(self, t: int) -> dict:
        return {
            "observation/wrist_image_left": self.frames["wrist"][t],
            "observation/exterior_image_1_left": self.frames["left"][t],
            "observation/exterior_image_2_left": self.frames["right"][t],
            "observation/joint_position": self.joints[t : t + 1],
            "observation/gripper_position": self.gripper[t : t + 1, None],
            "prompt": self.task,
        }

    def composite(self, t: int) -> np.ndarray:
        """The policy's 'concat_view' layout: wrist on top, both exterior views half-size below."""
        from PIL import Image

        half = lambda f: np.asarray(Image.fromarray(f).resize((320, 180), Image.BILINEAR))  # noqa: E731
        bottom = np.concatenate([half(self.frames["left"][t]), half(self.frames["right"][t])], axis=1)
        return np.concatenate([self.frames["wrist"][t], bottom], axis=0)  # [540,640,3]


def _decode(path: Path, start: float, count: int) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time < start - 0.5 / FPS:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
    return np.stack(frames)


def _save_video(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=fps, codec="libx264", macro_block_size=1)


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    if frame.shape[:2] == (height, width):
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def _side_by_side(real: np.ndarray, imagined: np.ndarray) -> np.ndarray:
    imagined = _resize(imagined, real.shape[1], real.shape[0])
    return np.concatenate([real, np.full((real.shape[0], 8, 3), 255, np.uint8), imagined], axis=1)


def run_eval(args) -> None:
    ep = DroidEpisode(args.episode)
    starts = list(range(0, ep.length - CHUNK - 1, CHUNK))[: args.chunks or None]
    print(f"[eval] episode {ep.index}: {ep.length} frames, task {ep.task!r}, {len(starts)} chunks", flush=True)
    policy = EdgePolicy(args.ckpt, decode_video=True, seed=args.seed)
    rows, video = [], []
    for t in starts:
        out = policy.infer(ep.observation(t))
        pred = out["action"]  # [32,8]: 7 joint targets (rad) + gripper (1 = closed)
        true_j, true_g = ep.action_joints[t : t + CHUNK], ep.action_gripper[t : t + CHUNK]
        hold = np.abs(ep.joints[t][None] - true_j).mean()  # baseline: keep the current pose
        row = {
            "t": t,
            "joint_mae_rad": float(np.abs(pred[:, :7] - true_j).mean()),
            "hold_still_mae_rad": float(hold),
            "final_joint_mae_rad": float(np.abs(pred[-1, :7] - true_j[-1]).mean()),
            "gripper_mae": float(np.abs(pred[:, 7] - true_g).mean()),
            "seconds": round(out["seconds"], 1),
            "peak_gib": round(out["peak_gib"], 2),
        }
        rows.append(row)
        print(f"[eval] {json.dumps(row)}", flush=True)
        for k in range(CHUNK):
            video.append(_side_by_side(ep.composite(t + k + 1), out["video"][min(k + 1, len(out["video"]) - 1)]))
        np.save(args.output.with_suffix(f".t{t:04d}.npy"), pred)
    summary = {
        "episode": ep.index,
        "task": ep.task,
        "chunks": rows,
        "mean_joint_mae_rad": float(np.mean([r["joint_mae_rad"] for r in rows])),
        "mean_hold_still_mae_rad": float(np.mean([r["hold_still_mae_rad"] for r in rows])),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    _save_video(video, args.output.with_suffix(".mp4"))
    print(f"[eval] mean joint MAE {summary['mean_joint_mae_rad']:.4f} rad vs hold-still {summary['mean_hold_still_mae_rad']:.4f} rad -> {args.output}.json/.mp4", flush=True)


def run_dream(args) -> None:
    ep = DroidEpisode(args.episode)
    print(f"[dream] episode {ep.index}: task {ep.task!r}, {args.chunks} chunks of {CHUNK} actions", flush=True)
    policy = EdgePolicy(args.ckpt, decode_video=True, seed=args.seed)
    obs = ep.observation(0)
    video, joints = [ep.composite(0)], [np.concatenate([ep.joints[0], ep.gripper[:1]])]
    for i in range(args.chunks):
        out = policy.infer(obs)
        pred = out["action"]
        # The decoded video is cropped to the model canvas's content (e.g. 640x528); bring it back to
        # the 640x540 observation layout so the frames stack and feed back like real camera images.
        frames = [_resize(f, video[0].shape[1], video[0].shape[0]) for f in out["video"]]
        video.extend(frames[1:])
        joints.extend(list(pred))
        print(f"[dream] chunk {i + 1}/{args.chunks}: {len(frames)} frames (decoded {out['video'].shape[2]}x{out['video'].shape[1]}) in {out['seconds']:.1f}s, peak {out['peak_gib']:.2f} GiB", flush=True)
        # The imagined last frame and the last commanded joints become the next observation.
        obs = {
            "observation/image": frames[-1],
            "observation/joint_position": pred[-1:, :7],
            "observation/gripper_position": pred[-1:, 7:8],
            "prompt": ep.task,
        }
        _save_video([np.asarray(f) for f in video], args.output.with_suffix(".mp4"))
    np.save(args.output.with_suffix(".joints.npy"), np.stack(joints))
    print(f"[dream] {len(video)} frames ({len(video) / FPS:.1f} s) -> {args.output}.mp4", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "dream"):
        p = sub.add_parser(name)
        p.add_argument("--episode", type=int, default=3)
        p.add_argument("--chunks", type=int, default=6, help="32-action chunks (eval: 0 = whole episode)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
        p.add_argument("-o", "--output", type=Path, default=None)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.ckpt, port=args.port)
        return
    if args.output is None:
        args.output = ROOT / "outputs" / "policy" / f"droid_ep{args.episode:03d}_{args.cmd}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    (run_eval if args.cmd == "eval" else run_dream)(args)
    print(f"[done] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
