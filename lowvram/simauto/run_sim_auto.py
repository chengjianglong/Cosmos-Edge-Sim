#!/usr/bin/env python
"""Run Cosmos3-Nano-Sim-Auto through NVIDIA cosmos-framework on two 6 GB Ampere GPUs.

Usage (from the cosmos-framework venv, any working directory):
    python lowvram/simauto/run_sim_auto.py -i request.json -o outputs/sim_auto/run1 \
        --checkpoint-path checkpoints/Cosmos3-Nano-Sim-Auto

All arguments are passed through to ``cosmos_framework.scripts.inference``; the low-VRAM flags
(--no-use-torch-compile --no-use-cuda-graphs --no-diffusion-cache --no-guardrails, cp=1) are
appended automatically. Patches applied before the framework starts:

  0. The process re-execs with nop2p/libnop2p.so preloaded: direct GPU peer copies are corrupt on
     this machine, so CUDA is told there is no peer access and copies are staged through host RAM.
  1. FlashAttention-4 gapped replay -> fa_shim (PyTorch memory-efficient attention, sm_80+).
  2. MoT decoder linears -> int8 weights in pinned CPU RAM streamed to the GPU running each layer;
     token embedding kept on CPU.
  3. Decoder layers [SIM_AUTO_SPLIT_LAYER:] (default 30-35) run on the second GPU with their share
     of the rolling KV cache (layer_split.py).
  4. Wan2.2 video VAE on the second GPU (SIM_AUTO_VAE_GPU, default 1), loaded on CPU first,
     encoding 4 frames per chunk and decoding each camera in overlapping width strips
     (vae_tiling.py). The LiDAR VAE stays on CPU unless SIM_AUTO_LIDAR_GPU is set.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _ensure_no_p2p() -> None:
    """Re-exec with nop2p/libnop2p.so preloaded so no CUDA library uses direct GPU peer copies.

    On this machine CUDA reports peer access between the GPUs, but direct peer copies silently
    return zeros/garbage (IOMMU/ACS blocks PCIe P2P). The framework moves tensors with plain
    ``.to(device)`` in many places, so instead of patching each one, the shim makes
    cudaDeviceCanAccessPeer report 0; cross-device copies then go through cudaMemcpy, which the
    driver stages through host memory. Set SIM_AUTO_ALLOW_P2P=1 on machines where P2P works.
    """
    if os.environ.get("SIM_AUTO_ALLOW_P2P"):
        return
    lib = HERE / "nop2p" / "libnop2p.so"
    if str(lib) in os.environ.get("LD_PRELOAD", ""):
        return
    if not lib.exists():
        import subprocess

        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(lib), str(lib.with_name("nop2p.c"))], check=True)
    os.environ["LD_PRELOAD"] = ":".join(filter(None, [str(lib), os.environ.get("LD_PRELOAD")]))
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    _ensure_no_p2p()

os.environ.setdefault("COSMOS_TRAINING", "0")
os.environ.setdefault("COSMOS_INTERNAL", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

TRANSFORMER_DEVICE = torch.device("cuda:0")
VAE_DEVICE = torch.device(f"cuda:{os.environ.get('SIM_AUTO_VAE_GPU', '1')}")
# The LiDAR VAE is small and the transformer GPU has spare room (its weights are streamed),
# so it goes there by default, leaving the second GPU to the Wan video VAE.
# The LiDAR VAE (~0.6 GB) is only used in joint camera+LiDAR mode, so by default it stays in host
# RAM; set SIM_AUTO_LIDAR_GPU=0 (or 1) for joint requests.
_lidar = os.environ.get("SIM_AUTO_LIDAR_GPU", "cpu")
LIDAR_DEVICE = torch.device("cpu" if _lidar == "cpu" else f"cuda:{_lidar}")
# Decoder layers [SPLIT_LAYER:] run on the VAE GPU, so both GPUs share the KV cache and activations.
SPLIT_LAYER = int(os.environ.get("SIM_AUTO_SPLIT_LAYER", "30"))
if torch.cuda.device_count() < 2:
    VAE_DEVICE = TRANSFORMER_DEVICE
    LIDAR_DEVICE = TRANSFORMER_DEVICE if _lidar != "cpu" else LIDAR_DEVICE
    SPLIT_LAYER = 0


def xfer(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor between devices, staging GPU->GPU copies through host RAM.

    On this machine CUDA reports peer access between the two A2000s, but direct peer copies
    silently return zeros (IOMMU/ACS blocks PCIe P2P), so never copy GPU->GPU directly.
    """
    device = torch.device(device)
    if t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda":
        t = t.cpu()
    return t.to(device)


def _moved(fn, arg_index: int = 1, device: torch.device = VAE_DEVICE):
    """Run a VAE method on ``device`` and return the result on the caller's device."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = list(args)
        x = args[arg_index]
        home = x.device
        args[arg_index] = xfer(x, device)
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            out = fn(*args, **kwargs)
        return xfer(out, home) if torch.is_tensor(out) else out

    return wrapper


def _tensor_stats(obj, depth: int = 0) -> list[str]:
    """Summaries of the tensors inside a (possibly nested) layer output."""
    if torch.is_tensor(obj):
        if not obj.is_floating_point() or obj.numel() == 0:
            return []
        f = obj.detach().float()
        return [f"{tuple(obj.shape)} std={f.std().item():.3g} absmax={f.abs().max().item():.3g} nan={bool(torch.isnan(f).any())}"]
    if depth > 2:
        return []
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(obj, "__dict__"):
        items = vars(obj).items()
    else:
        items = []
    out = []
    for key, value in items:
        out += [f"{key}:{s}" for s in _tensor_stats(value, depth + 1)]
    return out


def install_debug_hooks(net) -> None:
    """SIM_AUTO_DEBUG=1: print per-layer output stats for the first forwards and VAE decode inputs."""
    calls = {"n": 0}
    layers = net.language_model.model.layers

    def hook(module, args, output):
        if calls["n"] < 2 * len(layers):
            idx = calls["n"] % len(layers)
            if idx in (0, 1, 2, len(layers) // 2, len(layers) - 1):
                print(f"[debug] fwd{calls['n'] // len(layers)} layer{idx}: " + " | ".join(_tensor_stats(output[0])[:4]), flush=True)
        calls["n"] += 1

    for layer in layers:
        layer.register_forward_hook(hook)

    from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan

    original_decode = wan.WanVAE_.decode

    def decode(self, z, scale, clear_decoder_cache):
        print(f"[debug] {time.strftime('%H:%M:%S')} vae decode input: {_tensor_stats(z)}", flush=True)
        return original_decode(self, z, scale, clear_decoder_cache)

    wan.WanVAE_.decode = decode

    for name in ("encode", "encode_streaming"):
        original_encode = getattr(wan.WanVAE_, name)

        def make(orig, label):
            def encode(self, x, scale, *args, **kwargs):
                out = orig(self, x, scale, *args, **kwargs)
                f = x.detach().float()
                print(
                    f"[debug] vae {label} input {tuple(x.shape)} {x.dtype} min={f.min().item():.3g} "
                    f"max={f.max().item():.3g} mean={f.mean().item():.3g} contiguous={x.is_contiguous()} "
                    f"-> {_tensor_stats(out)}",
                    flush=True,
                )
                return out

            return encode

        setattr(wan.WanVAE_, name, make(original_encode, name))


def release_heap() -> None:
    """Return freed heap pages (CPU-side VAE load, int8 conversion temporaries) to the OS."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def install_patches() -> None:
    import fa_shim
    import int8_stream
    import vae_tiling

    fa_shim.install()
    vae_tiling.install()

    # --- int8 streaming of the MoT weights -------------------------------------------------
    from cosmos_framework.model.generator import omni_mot_model

    original_parallelize = omni_mot_model.parallelize_vfm_network

    @functools.wraps(original_parallelize)
    def parallelize_then_swap(net, *args, **kwargs):
        counts = int8_stream.swap_on_meta(net)
        print(f"[int8] swapped on meta: {counts}", flush=True)
        return original_parallelize(net, *args, **kwargs)

    omni_mot_model.parallelize_vfm_network = parallelize_then_swap

    from cosmos_framework.inference import model as inference_model

    cls = inference_model.Cosmos3OmniModel
    original_from_pretrained = cls.from_pretrained_dcp.__func__

    def from_pretrained_dcp(klass, checkpoint_path, *args, **kwargs):
        model = original_from_pretrained(klass, checkpoint_path, *args, **kwargs)
        int8_stream.report_resident(model.model.net, TRANSFORMER_DEVICE)
        print(f"[mem] cuda:0 allocated before int8 load: {torch.cuda.memory_allocated(TRANSFORMER_DEVICE) / 2**30:.2f} GiB", flush=True)
        int8_stream.load_streamed_weights(
            model.model.net, Path(checkpoint_path), inference_model._diffusers_to_net_key, TRANSFORMER_DEVICE
        )
        if SPLIT_LAYER:
            import layer_split

            layer_split.install()
            moved = layer_split.split_layers(
                model.model.net.language_model.model, SPLIT_LAYER, VAE_DEVICE, TRANSFORMER_DEVICE
            )
            n = len(model.model.net.language_model.model.layers)
            print(f"[split] layers {SPLIT_LAYER}-{n - 1} on {VAE_DEVICE} ({moved / 2**20:.0f} MiB resident moved)", flush=True)
        torch.cuda.empty_cache()
        release_heap()
        if os.environ.get("SIM_AUTO_DEBUG"):
            install_debug_hooks(model.model.net)
        print(
            f"[mem] transformer GPU resident after load: {torch.cuda.memory_allocated(TRANSFORMER_DEVICE) / 2**30:.2f} GiB",
            flush=True,
        )
        return model

    cls.from_pretrained_dcp = classmethod(from_pretrained_dcp)

    # --- VAEs on the second GPU -------------------------------------------------------------
    if VAE_DEVICE != TRANSFORMER_DEVICE:
        from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan

        original_init = wan.WanVAE.__init__

        @functools.wraps(original_init)
        def init_on_vae_gpu(self, *args, **kwargs):
            # Build + load the fp32 checkpoint on CPU (on GPU it needs ~2x 2.8 GB), cast to bf16
            # there, then move the ~1.4 GB model to the VAE GPU.
            kwargs["device"] = "cpu"
            # Encode 4 frames per chunk (default 24 at 480p) so activations fit next to the VAE.
            frames = int(os.environ.get("SIM_AUTO_VAE_ENCODE_FRAMES", "4"))
            kwargs["temporal_window"] = {"256": frames, "480": frames, "720": frames}
            original_init(self, *args, **kwargs)
            self.model = self.model.to(VAE_DEVICE)
            self.scale = tuple(t.to(VAE_DEVICE) for t in self.scale)
            self.device = VAE_DEVICE
            release_heap()

        wan.WanVAE.__init__ = init_on_vae_gpu
        for name in ("encode", "encode_streaming", "decode"):
            setattr(wan.WanVAE, name, _moved(getattr(wan.WanVAE, name)))

        from cosmos_framework.model.generator.tokenizers.lidar import lidar_tokenizer_v1 as lidar

        lidar_cls = lidar.LidarTokenizerV1Interface
        original_lidar_init = lidar_cls.__init__

        @functools.wraps(original_lidar_init)
        def lidar_init_on_vae_gpu(self, *args, **kwargs):
            kwargs["device"] = LIDAR_DEVICE
            original_lidar_init(self, *args, **kwargs)

        lidar_cls.__init__ = lidar_init_on_vae_gpu
        for name in ("encode", "decode"):
            if hasattr(lidar_cls, name):
                setattr(lidar_cls, name, _moved(getattr(lidar_cls, name), device=LIDAR_DEVICE))


LOW_VRAM_FLAGS = [
    "--no-use-torch-compile",
    "--no-use-cuda-graphs",
    "--no-diffusion-cache",
    "--no-guardrails",
    "--cp-size",
    "1",
    "--cfgp-size",
    "1",
    "--dp-shard-size",
    "1",
    "--dp-replicate-size",
    "1",
]


def main() -> None:
    torch.cuda.set_device(TRANSFORMER_DEVICE)
    if torch.cuda.device_count() > 1:
        probe = torch.arange(1, 9, device=TRANSFORMER_DEVICE, dtype=torch.float32)
        ok = bool((probe.to(VAE_DEVICE).cpu() == probe.cpu()).all())
        print(f"[run] cross-GPU copy check: {'ok' if ok else 'BROKEN (data corrupted)'}", flush=True)
        if not ok:
            raise SystemExit("Direct GPU-to-GPU copies are corrupt; run without SIM_AUTO_ALLOW_P2P.")
    install_patches()
    passed = sys.argv[1:]
    extra = [f for f in LOW_VRAM_FLAGS if f.startswith("--") and f not in passed]
    # keep value-taking flags paired
    argv, i = [], 0
    while i < len(LOW_VRAM_FLAGS):
        flag = LOW_VRAM_FLAGS[i]
        takes_value = i + 1 < len(LOW_VRAM_FLAGS) and not LOW_VRAM_FLAGS[i + 1].startswith("--")
        if flag in extra:
            argv.append(flag)
            if takes_value:
                argv.append(LOW_VRAM_FLAGS[i + 1])
        i += 2 if takes_value else 1
    sys.argv = [sys.argv[0], *passed, *argv]
    print("[run] args:", " ".join(sys.argv[1:]), flush=True)
    t0 = time.time()
    from cosmos_framework.scripts import inference as inference_script

    snap = os.environ.get("SIM_AUTO_MEMSNAP")  # path: dump a CUDA memory snapshot if generation fails
    if snap:
        torch.cuda.memory._record_memory_history(max_entries=200000, device=TRANSFORMER_DEVICE)
    try:
        inference_script.main()
    except BaseException:
        if snap:
            torch.cuda.memory._dump_snapshot(snap)
            print(f"[mem] snapshot written to {snap}", flush=True)
        raise
    print(f"[run] total {time.time() - t0:.0f}s", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"[mem] cuda:{i} peak allocated {torch.cuda.max_memory_allocated(i) / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
