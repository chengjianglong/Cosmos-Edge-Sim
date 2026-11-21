#!/usr/bin/env python
"""Make a LeRobot v3 root that indexes only the DROID episodes that are on disk.

The Cosmos3-DROID metadata lists all 57,639 episodes, but a single downloaded shard
(``*/chunk-000/file-000``) holds the videos of only the first few dozen. The training loader
indexes episodes from the metadata, so this writes a root whose ``meta/`` lists just those
episodes and whose ``data/`` and ``videos/`` are symlinks to the downloaded shards. The
directory keeps the dataset's name because the loader derives its schema from it.

    third_party/cosmos-framework/.venv/bin/python lowvram/train/make_droid_subset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
NAME = "droid_plus_lerobot_640x360_20260412"
VIDEO_KEYS = (
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
    "observation.image.wrist_image_left",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "checkpoints" / NAME / "success")
    ap.add_argument("--dst", type=Path, default=ROOT / "checkpoints" / "droid_subset" / NAME / "success")
    args = ap.parse_args()

    episodes = pq.read_table(args.src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    on_disk = None
    for key in VIDEO_KEYS:  # every camera's video and the state data must be in a downloaded shard
        for field in ("chunk_index", "file_index"):
            ok = pc.equal(episodes[f"videos/{key}/{field}"], 0)
            on_disk = ok if on_disk is None else pc.and_(on_disk, ok)
    on_disk = pc.and_(on_disk, pc.and_(pc.equal(episodes["data/chunk_index"], 0), pc.equal(episodes["data/file_index"], 0)))
    # Episode indexes must stay contiguous from 0, so keep the leading run that is on disk.
    flags = on_disk.to_pylist()
    count = flags.index(False) if False in flags else len(flags)
    subset = episodes.slice(0, count)

    info = json.loads((args.src / "meta" / "info.json").read_text())
    info["total_episodes"] = count
    info["total_frames"] = int(pc.sum(subset["length"]).as_py())
    info["splits"] = {"train": f"0:{count}"}

    meta = args.dst / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, meta / "episodes" / "chunk-000" / "file-000.parquet")
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    for name in ("tasks.parquet", "stats.json"):
        link = meta / name
        if not link.exists():
            link.symlink_to((args.src / "meta" / name).resolve())
    for name in ("data", "videos"):
        link = args.dst / name
        if not link.exists():
            link.symlink_to((args.src / name).resolve())
    print(f"{args.dst}: {count} episodes, {info['total_frames']} frames")


if __name__ == "__main__":
    main()
