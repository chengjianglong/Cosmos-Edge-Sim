"""Weight-only int8 Linear layers streamed from pinned CPU RAM, for running Cosmos3-Nano-Sim-Auto
(~30 GB of bf16 weights) on 6 GB GPUs.

* ``swap_on_meta(net)`` replaces, while the network is still on the meta device, every ``nn.Linear``
  in the MoT decoder layers (and the unused text ``lm_head``) with ``StreamedInt8Linear``, and the
  token embedding with ``CpuEmbedding``. These hold no ``nn.Parameter``s, so the framework's
  ``to_empty(device=cuda)`` + checkpoint load only materialize the small remaining tensors on GPU.
* ``load_streamed_weights(net, checkpoint_dir, key_mapper)`` then reads the matching safetensors
  tensors one by one, quantizes each to int8 (symmetric, per output channel) and keeps it in
  pinned CPU memory (~15 GB in total).
* At forward time each layer copies its int8 weight to the GPU, dequantizes to bf16 and runs
  ``F.linear``. The next layer's weight is prefetched on a side stream while the current one runs.
"""

from __future__ import annotations

import json
import mmap
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


_PINNED_REGIONS: list[mmap.mmap] = []  # keep the mappings alive for the life of the process


def _pinned_empty(shape, dtype) -> torch.Tensor:
    """Page-locked host tensor of exactly the requested size, excluded from fork().

    * ``Tensor.pin_memory()`` goes through PyTorch's caching host allocator, which rounds every
      block up to a power of two (12.9 GB of int8 weights became ~19.6 GB of locked RAM), so we
      register our own allocation with cudaHostRegister instead.
    * Linux copies DMA-pinned anonymous pages eagerly on fork(). The framework forks ffmpeg to
      write the MP4s, which then tried to duplicate all 12.9 GB and got the run OOM-killed.
      The buffer is therefore an anonymous mmap marked MADV_DONTFORK before it is registered.
    """
    numel = 1
    for s in shape if isinstance(shape, (tuple, list, torch.Size)) else (shape,):
        numel *= int(s)
    nbytes = max(1, numel * torch.empty((), dtype=dtype).element_size())
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    region.madvise(mmap.MADV_DONTFORK)
    _PINNED_REGIONS.append(region)
    t = torch.frombuffer(region, dtype=dtype, count=numel).view(shape)
    # cudaHostRegisterPortable: pinned for every GPU's context, since layers may stream to either GPU.
    rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 1)
    if int(rc) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {rc}")
    return t


class _Prefetcher:
    """Copies the next streamed layer's int8 weight to the GPU on a side stream."""

    def __init__(self) -> None:
        self.order: list[StreamedInt8Linear] = []
        self.streams: dict[torch.device, torch.cuda.Stream] = {}

    def stream(self, device: torch.device) -> torch.cuda.Stream:
        if device not in self.streams:
            self.streams[device] = torch.cuda.Stream(device=device)
        return self.streams[device]


PREFETCH = _Prefetcher()


class StreamedInt8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, name: str) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net_key = name + ".weight"
        self.q_cpu: torch.Tensor | None = None  # [out, in] int8, pinned
        self.scale_cpu: torch.Tensor | None = None  # [out] fp32, pinned
        self.index = -1
        self.home: torch.device | None = None  # GPU this layer runs on, if pinned to one (layer_split)
        self._staged: tuple[torch.device, torch.cuda.Event, torch.Tensor, torch.Tensor] | None = None

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8-streamed"

    def set_weight(self, weight: torch.Tensor, work_device: torch.device) -> None:
        # Per-output-row symmetric int8, quantized in row chunks to bound GPU scratch memory.
        q_cpu = _pinned_empty(weight.shape, torch.int8)
        scale_cpu = _pinned_empty(weight.shape[0], torch.float32)
        rows = max(1, (64 * 1024 * 1024) // weight.shape[1])
        for r0 in range(0, weight.shape[0], rows):
            w = weight[r0 : r0 + rows].to(work_device, dtype=torch.float32)
            scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            q_cpu[r0 : r0 + rows] = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8).cpu()
            scale_cpu[r0 : r0 + rows] = scale.cpu()
        self.q_cpu = q_cpu
        self.scale_cpu = scale_cpu

    def stage(self, device: torch.device) -> None:
        """Start the async host->device copy of this layer's weight."""
        if self._staged is not None and self._staged[0] == device:
            return
        stream = PREFETCH.stream(device)
        with torch.cuda.stream(stream):
            q = self.q_cpu.to(device, non_blocking=True)
            s = self.scale_cpu.to(device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        self._staged = (device, event, q, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_cpu is None:
            raise RuntimeError(f"{self.net_key} was never loaded (unused module called?)")
        device = x.device
        self.stage(device)
        _, event, q, s = self._staged
        self._staged = None
        torch.cuda.current_stream(device).wait_event(event)
        q.record_stream(torch.cuda.current_stream(device))
        s.record_stream(torch.cuda.current_stream(device))
        # Prefetch the next layer in execution order while this one computes.
        nxt = self.index + 1
        if 0 <= self.index and nxt < len(PREFETCH.order):
            following = PREFETCH.order[nxt]
            following.stage(following.home or device)
        w = q.to(x.dtype).mul_(s.to(x.dtype)[:, None])
        return F.linear(x, w)


class CpuEmbedding(nn.Module):
    """Token embedding kept in CPU RAM (saves ~1.2 GB of VRAM); lookups are tiny."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None, name: str) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.net_key = name + ".weight"
        self.weight_cpu: torch.Tensor | None = None

    def set_weight(self, weight: torch.Tensor, work_device: torch.device) -> None:
        self.weight_cpu = weight.to(torch.bfloat16).contiguous()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.cpu(), self.weight_cpu).to(ids.device, non_blocking=True)


def _replace(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def swap_on_meta(net: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    order: list[StreamedInt8Linear] = []
    for name, module in list(net.named_modules()):
        if isinstance(module, nn.Linear) and (".layers." in name or name.endswith("lm_head")):
            if module.bias is not None:
                raise NotImplementedError(f"{name}: biased linears are not streamed")
            new = StreamedInt8Linear(module.in_features, module.out_features, name)
            _replace(net, name, new)
            if ".layers." in name:
                order.append(new)
            counts["linear"] += 1
        elif isinstance(module, nn.Embedding) and name.endswith("embed_tokens"):
            _replace(net, name, CpuEmbedding(module.num_embeddings, module.embedding_dim, module.padding_idx, name))
            counts["embedding"] += 1
    # named_modules() yields layers in definition order, which is close to execution order;
    # a wrong guess only costs a wasted prefetch.
    for i, m in enumerate(order):
        m.index = i
    PREFETCH.order = order
    return dict(counts)


def _weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for index in sorted(checkpoint_dir.glob("transformer/*.safetensors.index.json")):
        for key, shard in json.loads(index.read_text())["weight_map"].items():
            out[key] = index.parent / shard
    for single in checkpoint_dir.glob("transformer/*.safetensors"):
        if not any(v == single for v in out.values()):
            with safe_open(str(single), "pt") as f:
                for key in f.keys():
                    out.setdefault(key, single)
    return out


def report_resident(net: nn.Module, device: torch.device, top: int = 12) -> None:
    """Print which modules own the GPU-resident parameters/buffers."""
    sizes: dict[str, int] = defaultdict(int)
    for name, t in list(net.named_parameters()) + list(net.named_buffers()):
        if t.device == device:
            key = ".".join(p for p in name.split(".")[:4] if not p.isdigit())
            sizes[key] += t.numel() * t.element_size()
    total = sum(sizes.values())
    print(f"[resident] {device}: {total / 2**30:.2f} GiB in net params/buffers", flush=True)
    for key, n in sorted(sizes.items(), key=lambda kv: -kv[1])[:top]:
        print(f"[resident]   {n / 2**20:9.1f} MiB  {key}", flush=True)


def load_streamed_weights(
    net: nn.Module,
    checkpoint_dir: Path,
    key_mapper: Callable[[str, str], str | None],
    work_device: torch.device,
) -> None:
    t0 = time.time()
    targets = {
        m.net_key: m for m in net.modules() if isinstance(m, (StreamedInt8Linear, CpuEmbedding))
    }
    by_shard: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for ckpt_key, shard in _weight_map(checkpoint_dir).items():
        # The framework's mapper expects the weight path relative to the snapshot ("transformer/...").
        net_key = key_mapper(ckpt_key, str(shard.relative_to(checkpoint_dir)))
        if net_key in targets and not net_key.endswith("lm_head.weight"):  # lm_head: text output only, unused
            by_shard[shard].append((ckpt_key, net_key))
    found = set()
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(shard), "pt") as f:
            for ckpt_key, net_key in keys:
                targets[net_key].set_weight(f.get_tensor(ckpt_key), work_device)
                found.add(net_key)
    missing = sorted(k for k in targets if k not in found and not k.endswith("lm_head.weight"))
    if missing:
        raise KeyError(f"{len(missing)} streamed weights not found in checkpoint, e.g. {missing[:5]}")
    pinned = sum(m.q_cpu.numel() for m in targets.values() if isinstance(m, StreamedInt8Linear) and m.q_cpu is not None)
    print(f"[int8] loaded {len(found)} streamed tensors ({pinned / 2**30:.1f} GiB int8 pinned) in {time.time() - t0:.0f}s")
