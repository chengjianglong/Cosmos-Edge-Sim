#!/usr/bin/env python
"""Run nvidia/Cosmos3-Edge image-to-video on one small GPU (tested: RTX A2000 6 GB).

Memory layout:
  * transformer (~6.7 GB bf16) stays in pinned CPU RAM; diffusers group offloading streams
    N decoder layers at a time onto the GPU (peak ~2 GB VRAM for 33 frames at 480p).
  * VAE (Wan2.2) lives on CPU and is moved to the GPU only for the conditioning-frame
    encode and the final (tiled) decode.
  * For image-to-video only the first frame is VAE-encoded (the pipeline would otherwise
    encode the image repeated over every frame, and only latent frame 0 is used).
  * The cosmos_guardrail safety checker is disabled by default (it needs several extra GB
    of VRAM); pass --safety to enable it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "Cosmos3-Edge"


def gib(x: float) -> str:
    return f"{x / 2**30:.2f} GiB"


def report_memory(tag: str, device: torch.device) -> None:
    alloc = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    print(f"[mem] {tag}: {device} alloc={gib(alloc)} peak={gib(peak)}", flush=True)


def read_prompt(value: str | None) -> str:
    """Accept a literal prompt, a .txt file, or a .json (upsampled) prompt file."""
    if value is None:
        return ""
    p = Path(value)
    if p.is_file():
        if p.suffix == ".json":
            return json.dumps(json.loads(p.read_text()))
        return p.read_text().strip()
    return value


def build_pipeline(
    ckpt: Path, device: torch.device, offload_blocks: int = 2, safety: bool = False
) -> Cosmos3OmniPipeline:
    transformer = Cosmos3OmniTransformer.from_pretrained(ckpt, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.enable_group_offload(
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=offload_blocks,
        use_stream=True,
    )
    print(f"[placement] group offload: {offload_blocks} layer(s) per group streamed CPU -> {device}")

    pipe = Cosmos3OmniPipeline.from_pretrained(
        ckpt,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        enable_safety_checker=safety,
    )
    # The pipeline infers its device from parameter placement, which is "cpu" under group
    # offloading; pin it to the GPU the layers are streamed to.
    pipe._get_execution_device = lambda: device
    # VAE stays on CPU until needed; tiling keeps the 480p decode within a couple of GB.
    pipe.vae.to("cpu")
    pipe.vae.enable_tiling()
    patch_vae_encode(pipe, device)
    return pipe


def patch_vae_encode(pipe: Cosmos3OmniPipeline, device: torch.device) -> None:
    """Move the VAE to the GPU just for conditioning-frame encoding, then back to CPU."""
    original = pipe._encode_video

    def _encode_video(x: torch.Tensor) -> torch.Tensor:
        # Image-to-video feeds one image repeated over all frames, but only latent frame 0
        # is kept as clean conditioning (the rest is replaced by noise). The Wan VAE is
        # causal, so encoding frame 0 alone gives that same latent at a fraction of the memory.
        t = x.shape[2]
        lat_t = 1 + (t - 1) // pipe.vae_scale_factor_temporal
        single = t > 1 and bool((x == x[:, :, :1]).all())
        src = x[:, :, :1] if single else x

        try:
            pipe.vae.to(device)
            out = original(src.to(device))
        except torch.OutOfMemoryError:
            print("[warn] VAE encode OOM on GPU; retrying on CPU (slow)", flush=True)
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
            out = original(src.to("cpu"))
        finally:
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
        if single:
            out = out.expand(-1, -1, lat_t, -1, -1).contiguous()
        return out.to(x.device)

    pipe._encode_video = _encode_video


@torch.no_grad()
def decode_latents(pipe: Cosmos3OmniPipeline, latents: torch.Tensor, device: torch.device):
    """Mirror of the pipeline's own decode step, with the VAE moved to the GPU only for it."""
    torch.cuda.empty_cache()
    pipe.vae.to(device)
    try:
        dtype = pipe.vae.dtype
        latents = latents.to(device)
        mean = pipe._vae_latents_mean.to(device=device, dtype=dtype)
        inv_std = pipe._vae_latents_inv_std.to(device=device, dtype=dtype)
        z_raw = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = pipe.vae.decode(z_raw).sample.to(torch.float32)
        return pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
    finally:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--image", type=str, default=None, help="conditioning first frame (omit for text-to-video)")
    ap.add_argument("--prompt", type=str, default=None, help="text, .txt file or upsampled .json prompt file")
    ap.add_argument("--negative-prompt", type=str, default=None)
    ap.add_argument("--num-frames", type=int, default=121)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--flow-shift", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index to run on")
    ap.add_argument(
        "--offload-blocks", type=int, default=2, help="decoder layers streamed to the GPU per group (more = faster, more VRAM)"
    )
    ap.add_argument("--safety", action="store_true", help="enable cosmos_guardrail safety checker")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "outputs" / "edge_i2v.mp4")
    args = ap.parse_args()

    if args.image is None and args.prompt is None:
        assets = args.ckpt / "assets"
        args.image = str(assets / "example_i2v_input.jpg")
        args.prompt = str(assets / "example_i2v_prompt.json")
        args.negative_prompt = args.negative_prompt or str(assets / "negative_prompt.json")
        print(f"[info] no inputs given; using the model card example from {assets}")

    device = torch.device(f"cuda:{args.gpu}")
    t0 = time.time()
    pipe = build_pipeline(args.ckpt, device, args.offload_blocks, args.safety)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )
    print(f"[time] load {time.time() - t0:.1f}s", flush=True)

    kwargs = dict(
        prompt=read_prompt(args.prompt),
        negative_prompt=read_prompt(args.negative_prompt),
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        add_resolution_template=False,
        add_duration_template=False,
        output_type="latent",
    )
    if args.image:
        kwargs["image"] = load_image(args.image)

    t1 = time.time()
    result = pipe(**kwargs)
    print(f"[time] denoise {time.time() - t1:.1f}s ({args.steps} steps)", flush=True)
    report_memory("after denoise", device)

    t2 = time.time()
    video = decode_latents(pipe, result.video, device)
    print(f"[time] vae decode {time.time() - t2:.1f}s", flush=True)
    report_memory("after decode", device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(args.output), fps=int(args.fps), macro_block_size=1)
    print(f"[done] {len(video)} frames -> {args.output}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
