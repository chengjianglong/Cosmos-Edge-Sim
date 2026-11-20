"""Pipeline-split the MoT decoder layers across two GPUs (no P2P required).

Layers ``[split, num_layers)`` run on ``device1``; the rest stay on the transformer GPU. Each layer's
rolling KV cache is allocated on the device that produced its K/V, so the cache splits too.

Only the layer boundary crosses GPUs: before the first device1 layer, the hidden states and the
per-forward metadata (attention plans, RoPE tables, memory read handles) are copied to device1,
staged through host RAM because direct peer copies return zeros on this machine. After the last
layer the hidden states go back to the transformer GPU for the final norm.
"""

from __future__ import annotations

import copy
import dataclasses

import torch
import torch.nn as nn

from int8_stream import StreamedInt8Linear


def _xfer(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda":
        t = t.cpu()
    return t.to(device)


class _Mover:
    """Deep-copies a (nested) object with every CUDA tensor moved to ``device``.

    Tensor identity is preserved within one mover (the same source tensor maps to one copy), and
    objects are memoized by id so shared metadata is only moved once per forward.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.memo: dict[int, object] = {}
        self.keep: list[object] = []  # hold sources alive so ids stay unique

    def __call__(self, obj, depth: int = 0):
        if obj is None or isinstance(obj, (int, float, bool, str, bytes, torch.device, torch.dtype, type)):
            return obj
        key = id(obj)
        if key in self.memo:
            return self.memo[key]
        if torch.is_tensor(obj):
            out = _xfer(obj, self.device) if obj.device.type == "cuda" else obj
        elif isinstance(obj, nn.Module) or depth > 8 or callable(obj) and not hasattr(obj, "__dict__"):
            out = obj
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            out = type(obj)(*(self(x, depth + 1) for x in obj))
        elif isinstance(obj, (list, tuple)):
            out = type(obj)(self(x, depth + 1) for x in obj)
        elif isinstance(obj, dict):
            out = {k: self(v, depth + 1) for k, v in obj.items()}
        elif dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__") or hasattr(obj, "__slots__"):
            out = copy.copy(obj)
            self.memo[key] = out  # break cycles before recursing
            self.keep.append(obj)
            names = list(getattr(obj, "__dict__", {}).keys())
            for cls in type(obj).__mro__:
                names += [s for s in getattr(cls, "__slots__", ()) if isinstance(s, str)]
            for name in dict.fromkeys(names):
                if name.startswith("__"):
                    continue
                try:
                    value = object.__getattribute__(obj, name)
                except AttributeError:
                    continue
                moved = self(value, depth + 1)
                if moved is not value:
                    object.__setattr__(out, name, moved)
            return out
        else:
            out = obj
        self.memo[key] = out
        self.keep.append(obj)
        return out


class _RemoteLayer(nn.Module):
    """Wraps a decoder layer that lives on ``device``; ``first``/``last`` mark the GPU boundary."""

    def __init__(self, layer: nn.Module, device: torch.device, home: torch.device, state: dict, first: bool, last: bool):
        super().__init__()
        self.layer = layer
        self.device = device
        self.home = home
        self.state = state
        self.first = first
        self.last = last

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["layer"], name)

    def forward(self, hidden_states, attention_mask, position_embeddings, **kwargs):
        if self.first or self.state.get("mover") is None:
            self.state["mover"] = _Mover(self.device)
        mover = self.state["mover"]
        hidden_states = mover(hidden_states)
        attention_mask = mover(attention_mask)
        position_embeddings = mover(position_embeddings)
        # memory_value differs per layer; move it with a fresh memo so stale ids are not reused
        kwargs = {k: (_Mover(self.device)(v) if k == "memory_value" else mover(v)) for k, v in kwargs.items()}
        with torch.cuda.device(self.device):
            hidden_states, lbl, kv_to_store = self.layer(hidden_states, attention_mask, position_embeddings, **kwargs)
        if self.last:
            hidden_states = _Mover(self.home)(hidden_states)
            self.state["mover"] = None
        return hidden_states, lbl, kv_to_store


def split_layers(language_model: nn.Module, split: int, device1: torch.device, home: torch.device) -> int:
    """Move layers [split:] (their resident params/buffers and streamed-weight targets) to device1."""
    layers = language_model.layers
    n = len(layers)
    if not 0 < split < n:
        return 0
    state: dict = {}
    moved = 0
    for i in range(split, n):
        layer = layers[i]
        for module in layer.modules():
            if isinstance(module, StreamedInt8Linear):
                module.home = device1
            for name, p in list(module._parameters.items()):
                if p is not None and p.device.type == "cuda":
                    module._parameters[name] = nn.Parameter(_xfer(p.data, device1), requires_grad=False)
                    moved += p.numel() * p.element_size()
            for name, b in list(module._buffers.items()):
                if b is not None and b.device.type == "cuda":
                    module._buffers[name] = _xfer(b, device1)
                    moved += b.numel() * b.element_size()
        layers[i] = _RemoteLayer(layer, device1, home, state, first=i == split, last=i == n - 1)
    return moved


def install() -> None:
    """Make framework-side per-plan GPU data follow the layer's device.

    The gapped replay attention keeps its offset/index tensors in a registry (``_RUNTIMES``) keyed
    by a CPU handle, created on the transformer GPU. Layers on the second GPU get a copy of that
    runtime on their device, cached on the runtime itself so it lives exactly as long as the plan.
    """
    from cosmos_framework.model.generator.mot import maskless_attention as m

    original = m._runtime

    def runtime_on_current_device(handle):
        runtime = original(handle)
        device = torch.device("cuda", torch.cuda.current_device())
        if runtime.q_offsets.device == device:
            return runtime
        copies = runtime.__dict__.setdefault("_lowvram_copies", {})
        if device not in copies:
            copies[device] = _Mover(device)(runtime)
            copies[device].__dict__.pop("_lowvram_copies", None)
        return copies[device]

    m._runtime = runtime_on_current_device
