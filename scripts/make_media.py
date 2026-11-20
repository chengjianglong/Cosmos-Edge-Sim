#!/usr/bin/env python
"""Build the README figures in docs/media/ from local run outputs.

    python scripts/make_media.py        # needs imageio[ffmpeg], Pillow, numpy, matplotlib

Inputs (produced by the commands in README.md):
  outputs/sim_auto/2cam_49f/daytime_scene/{vision,control}_view*.mp4   2-camera Sim-Auto run
  outputs/sim_auto/2cam_mem.log                                         memory trace of that run
  outputs/edge_121f.mp4                                                 Cosmos3-Edge default example
  outputs/bimanual/agibot_{006,034,001}_edge.mp4                         Edge bimanual rollouts
  Sim-Bimanual ground-truth videos (checkpoints/Cosmos3-Nano-Sim-Bimanual-assets)
  NVIDIA's daytime sample (Hugging Face cache) for the real reference frames
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
SIM = ROOT / "outputs" / "sim_auto" / "2cam_49f" / "daytime_scene"
SAMPLE = glob.glob(
    str(Path.home() / ".cache/huggingface/hub/models--nvidia--Cosmos3-Nano-Transfer-Auto/snapshots/*/assets/sample_data/daytime_scene")
)

try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
except OSError:
    FONT = ImageFont.load_default()


def video(path) -> np.ndarray:
    return iio.imread(path)  # [T,H,W,3] uint8


def resize(frame: np.ndarray, width: int) -> Image.Image:
    img = Image.fromarray(frame)
    return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = draw.textbbox((8, 6), text, font=FONT)
    draw.rectangle((x0 - 5, y0 - 4, x1 + 5, y1 + 4), fill=(0, 0, 0))
    draw.text((8, 6), text, font=FONT, fill=(255, 255, 255))
    return img


def grid(tiles: list[list[Image.Image]], gap: int = 4) -> Image.Image:
    w, h = tiles[0][0].size
    rows, cols = len(tiles), len(tiles[0])
    canvas = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            canvas.paste(tile, (c * (w + gap), r * (h + gap)))
    return canvas


def save_gif(frames: list[Image.Image], path: Path, fps: float) -> None:
    # One shared palette keeps the file small and avoids per-frame color flicker.
    palette = frames[len(frames) // 2].convert("P", palette=Image.ADAPTIVE, colors=255)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"{path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 2**20:.1f} MiB")


def sim_auto_gif() -> None:
    views = [("front_wide_120fov", "view01"), ("cross_left_120fov", "view00")]
    clips = []
    for name, view in views:
        clips.append((name, video(SIM / f"control_{view}_camera_{name}.mp4"), video(SIM / f"vision_{view}_camera_{name}.mp4")))
    frames = []
    for t in range(0, 49, 2):  # 49 frames at 30 fps -> 25 frames at 15 fps, same real-time speed
        rows = []
        for name, control, generated in clips:
            cam = name.replace("_120fov", "").replace("_", "-")
            rows.append(
                [label(resize(control[t], 400), f"{cam}: WSM control input"), label(resize(generated[t], 400), f"{cam}: generated")]
            )
        frames.append(grid(rows))
    save_gif(frames, OUT / "simauto_2cam.gif", fps=15)


def sim_auto_frames() -> None:
    real = video(Path(SAMPLE[0]) / "rgb" / "camera_front_wide_120fov.mp4") if SAMPLE else None
    control = video(SIM / "control_view01_camera_front_wide_120fov.mp4")
    generated = video(SIM / "vision_view01_camera_front_wide_120fov.mp4")
    rows = []
    if real is not None:
        rows.append(("real camera", real))
    rows += [("WSM control (lanes, boxes)", control), ("Sim-Auto output", generated)]
    tiles = [[label(resize(clip[t], 320), f"{name} - t={t}" if c == 0 else f"t={t}") for c, t in enumerate((0, 24, 48))] for name, clip in rows]
    img = grid(tiles)
    img.save(OUT / "simauto_front_frames.jpg", quality=88)
    print(f"docs/media/simauto_front_frames.jpg: {img.size}")


def edge_gif() -> None:
    clip = video(ROOT / "outputs" / "edge_121f.mp4")
    frames = [resize(clip[t], 480) for t in range(0, len(clip), 3)]  # 24 fps -> 8 fps sampled, played at 8 fps
    save_gif(frames, OUT / "edge_i2v.gif", fps=8)


BIMANUAL = ROOT / "outputs" / "bimanual"
BIMANUAL_GT = ROOT / "checkpoints" / "Cosmos3-Nano-Sim-Bimanual-assets" / "assets" / "training" / "agibot" / "gt"
EPISODES = ("006", "034", "001")


def bimanual_pair(episode: str) -> tuple[np.ndarray, np.ndarray]:
    """(ground truth subsampled 30 -> 10 fps, Edge rollout at 10 fps)."""
    gt = video(next(BIMANUAL_GT.glob(f"*{episode}*")))[::3]
    return gt, video(BIMANUAL / f"agibot_{episode}_edge.mp4")


def bimanual_gif() -> None:
    pairs = [bimanual_pair(e) for e in EPISODES[:2]]
    frames = []
    for t in range(0, 65, 2):  # first 4 chunks = 6.4 s (where the rollout still tracks the real episode), 5 fps
        rows = [
            [label(resize(gt[t], 300).resize((300, 165)), f"ep {e}: real robot"), label(resize(ed[t], 300).resize((300, 165)), f"ep {e}: Edge, same actions")]
            for e, (gt, ed) in zip(EPISODES[:2], pairs)
        ]
        frames.append(grid(rows))
    save_gif(frames, OUT / "bimanual_edge.gif", fps=5)


def bimanual_frames() -> None:
    gt, ed = bimanual_pair("006")
    times = (0, 16, 32, 64, 128, 288)
    tiles = [
        [label(resize(clip[t], 260).resize((260, 143)), f"{name} {t / 10:.1f}s" if c == 0 else f"{t / 10:.1f}s") for c, t in enumerate(times)]
        for name, clip in (("real", gt), ("Edge", ed))
    ]
    img = grid(tiles)
    img.save(OUT / "bimanual_drift.jpg", quality=88)
    print(f"docs/media/bimanual_drift.jpg: {img.size}")


def memory_chart() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for line in (ROOT / "outputs" / "sim_auto" / "2cam_mem.log").read_text().splitlines():
        nums = [float(x) for x in re.findall(r"[\d.]+", line.split(" ", 1)[1])]
        h, m, s = (int(x) for x in line.split()[0].split(":"))
        rows.append((h * 3600 + m * 60 + s, *nums))
    data = np.array(rows)
    t = (data[:, 0] - data[0, 0]) / 60  # minutes
    ram_used, gpu0, util0, gpu1, util1 = data[:, 1] / 1024, data[:, 3] / 1024, data[:, 4], data[:, 5] / 1024, data[:, 6]
    # Generation starts at the framework's "Rolling replay" log line.
    log = (ROOT / "outputs" / "sim_auto" / "2cam_49f.log").read_text()
    h, m, s = (int(x) for x in re.search(r"(\d\d):(\d\d):(\d\d)\|[^\n]*Rolling replay", log).groups())
    gen = (h * 3600 + m * 60 + s - data[0, 0]) / 60

    themes = {
        "light": dict(surface="#fcfcfb", ink="#1f1f1d", muted="#6b6a65", grid="#e4e3de", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
        "dark": dict(surface="#1a1a19", ink="#f0efec", muted="#a3a29a", grid="#33332f", s1="#3987e5", s2="#d95926", s3="#199e70"),
    }
    for mode, th in themes.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4, 2], hspace=0.38))
        fig.patch.set_facecolor(th["surface"])
        for ax in axes:
            ax.set_facecolor(th["surface"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(th["grid"])
            ax.tick_params(colors=th["muted"], labelsize=9)
            ax.grid(axis="y", color=th["grid"], linewidth=0.8)
            ax.axvline(gen, color=th["muted"], linewidth=1, linestyle=(0, (3, 3)))
        ax = axes[0]
        ax.plot(t, gpu0, color=th["s1"], linewidth=2, label="GPU 0: layers 0-29")
        ax.plot(t, gpu1, color=th["s2"], linewidth=2, label="GPU 1: layers 30-35 + VAE (+0.6 GB desktop)")
        ax.axhline(6138 / 1024, color=th["muted"], linewidth=1)
        ax.text(t[-1], 6138 / 1024 + 0.1, "6 GB per GPU", color=th["muted"], fontsize=9, ha="right", va="bottom")
        ax.set_ylim(0, 6.6)
        ax.set_title("VRAM used (GB)", loc="left", color=th["ink"], fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, labelcolor=th["ink"], fontsize=9)
        ax.text(gen + 0.05, 0.25, "generating 49 frames x 2 cameras", color=th["muted"], fontsize=9)
        ax.text(gen - 0.05, 0.25, "loading + int8 quantizing", color=th["muted"], fontsize=9, ha="right")
        ax = axes[1]
        ax.fill_between(t, 0, (util0 > 50).astype(float), step="mid", color=th["s1"], linewidth=0)
        ax.fill_between(t, 1.15, 1.15 + (util1 > 50).astype(float), step="mid", color=th["s2"], linewidth=0)
        ax.set_yticks([0.5, 1.65], ["GPU 0", "GPU 1"])
        ax.set_ylim(-0.1, 2.25)
        ax.grid(False)
        ax.set_title("Which GPU is computing (2 s samples)", loc="left", color=th["ink"], fontsize=11, fontweight="bold")
        ax = axes[2]
        ax.plot(t, ram_used, color=th["s3"], linewidth=2)
        ax.axhline(30, color=th["muted"], linewidth=1)
        ax.text(t[-1], 30.3, "30 GB system RAM", color=th["muted"], fontsize=9, ha="right", va="bottom")
        ax.set_ylim(0, 33)
        ax.set_title("System RAM used (GB) - 12.9 GB of int8 weights stay pinned", loc="left", color=th["ink"], fontsize=11, fontweight="bold")
        ax.set_xlabel("minutes since launch", color=th["muted"], fontsize=9)
        fig.savefig(OUT / f"memory_timeline_{mode}.png", dpi=110, bbox_inches="tight", facecolor=th["surface"])
        plt.close(fig)
        print(f"docs/media/memory_timeline_{mode}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim_auto_gif()
    sim_auto_frames()
    edge_gif()
    memory_chart()
    bimanual_gif()
    bimanual_frames()


if __name__ == "__main__":
    main()
