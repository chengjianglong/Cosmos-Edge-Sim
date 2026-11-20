"""Width-tiled decoding for the Wan2.2 VAE used by Cosmos3 Sim-Auto.

A full-width 480p decode needs > 5 GB of activations, but a half-width one needs ~1.8 GB.
The decoder is causal only in time (spatial convs see neighbours), so we decode overlapping
vertical strips, each with its own temporal feature cache that persists across chunk calls
exactly like the original single cache, and blend the overlaps with a linear ramp.
"""

from __future__ import annotations

import os

import torch


def _tiles(width: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    if width <= tile:
        return [(0, width)]
    stride = tile - overlap
    starts = list(range(0, width - tile, stride)) + [width - tile]
    return [(s, s + tile) for s in sorted(set(starts))]


def _move(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Direct GPU->GPU copies return zeros on this machine (PCIe P2P blocked); stage through host.
    if t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda":
        t = t.cpu()
    return t.to(device)


def _to(cache: list, device) -> list:
    device = torch.device(device)
    return [_move(c, device) if torch.is_tensor(c) else c for c in cache]


def install() -> None:
    from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as wan

    tile = int(os.environ.get("SIM_AUTO_VAE_TILE_LATENT", "22"))  # latent columns per strip
    overlap = int(os.environ.get("SIM_AUTO_VAE_TILE_OVERLAP", "8"))
    # Where inactive strips' decoder caches wait (~0.8 GB per camera at 480p). The VAE decodes
    # every chunk during generation, so parking them on the transformer GPU competes with the
    # KV cache; host RAM is the default.
    park = torch.device(os.environ.get("SIM_AUTO_VAE_CACHE_DEVICE", "cpu"))
    original = wan.WanVAE_.decode

    def tiled_decode(self, z, scale, clear_decoder_cache: bool):
        # z: [B,C,T,H,W]; B is the number of cameras on the multiview path. Each camera and each
        # width strip is decoded on its own with its own temporal cache, so peak memory is that of
        # one strip of one camera.
        width = z.shape[-1]
        spans = _tiles(width, tile, overlap)
        keys = [(b, span) for b in range(z.shape[0]) for span in spans]
        if len(keys) == 1:
            return original(self, z, scale, clear_decoder_cache)
        caches = getattr(self, "_tile_dec_caches", None)
        if clear_decoder_cache or caches is None or set(caches) != set(keys):
            caches = {key: self._new_dec_cache() for key in keys}
        # Only the active strip's cache lives on the VAE GPU; the others wait on ``park``.
        outputs = {}
        for key in keys:
            b, (s, e) = key
            self._dec_cache = _to(caches[key], z.device)
            out = original(self, z[b : b + 1, ..., s:e], scale, clear_decoder_cache=False)
            caches[key] = _to(self._dec_cache, park)
            self._dec_cache = self._new_dec_cache()
            outputs[key] = out.cpu()
            del out
            torch.cuda.empty_cache()
        self._tile_dec_caches = None if clear_decoder_cache else caches

        first = outputs[keys[0]]
        factor = first.shape[-1] // (spans[0][1] - spans[0][0])  # pixels per latent column
        ramp_len = overlap * factor
        weight = torch.zeros(width * factor, dtype=torch.float32)
        ramps = []
        for i, (s, e) in enumerate(spans):
            w = torch.ones((e - s) * factor, dtype=torch.float32)
            if i > 0:
                w[:ramp_len] = torch.linspace(0, 1, ramp_len + 2)[1:-1]
            if i < len(spans) - 1:
                w[-ramp_len:] = torch.minimum(w[-ramp_len:], torch.linspace(1, 0, ramp_len + 2)[1:-1])
            weight[s * factor : e * factor] += w
            ramps.append(w)
        videos = []
        for b in range(z.shape[0]):
            result = torch.zeros((*first.shape[:-1], width * factor), dtype=torch.float32)
            for (s, e), w in zip(spans, ramps):
                result[..., s * factor : e * factor] += outputs[(b, (s, e))].float() * w
            videos.append(result / weight)
        return torch.cat(videos).to(dtype=first.dtype, device=z.device)

    wan.WanVAE_.decode = tiled_decode
