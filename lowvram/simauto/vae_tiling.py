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

    tile = int(os.environ.get("SIM_AUTO_VAE_TILE_LATENT", "30"))  # latent columns per strip
    overlap = int(os.environ.get("SIM_AUTO_VAE_TILE_OVERLAP", "8"))
    # Where inactive strips' decoder caches wait (~0.8 GB each at 480p). The transformer GPU has
    # room during decode; "cpu" works too but costs system RAM.
    park = torch.device(os.environ.get("SIM_AUTO_VAE_CACHE_DEVICE", "cuda:0"))
    original = wan.WanVAE_.decode

    def tiled_decode(self, z, scale, clear_decoder_cache: bool):
        width = z.shape[-1]
        spans = _tiles(width, tile, overlap)
        if len(spans) == 1:
            return original(self, z, scale, clear_decoder_cache)
        caches = getattr(self, "_tile_dec_caches", None)
        if clear_decoder_cache or caches is None or set(caches) != set(spans):
            caches = {span: self._new_dec_cache() for span in spans}
        # Only the active strip's cache lives on the VAE GPU; the others wait on ``park``.
        outputs = []
        for span in spans:
            self._dec_cache = _to(caches[span], z.device)
            out = original(self, z[..., span[0] : span[1]], scale, clear_decoder_cache=False)
            caches[span] = _to(self._dec_cache, park)
            self._dec_cache = self._new_dec_cache()
            outputs.append(out.cpu())
            del out
            torch.cuda.empty_cache()
        self._tile_dec_caches = None if clear_decoder_cache else caches

        factor = outputs[0].shape[-1] // (spans[0][1] - spans[0][0])  # pixels per latent column
        result = torch.zeros(
            (*outputs[0].shape[:-1], width * factor), dtype=torch.float32, device=outputs[0].device
        )
        weight = torch.zeros(width * factor, dtype=torch.float32, device=outputs[0].device)
        ramp_len = overlap * factor
        for i, ((s, e), out) in enumerate(zip(spans, outputs)):
            w = torch.ones(out.shape[-1], dtype=torch.float32, device=out.device)
            if i > 0:
                w[:ramp_len] = torch.linspace(0, 1, ramp_len + 2, device=out.device)[1:-1]
            if i < len(spans) - 1:
                w[-ramp_len:] = torch.minimum(w[-ramp_len:], torch.linspace(1, 0, ramp_len + 2, device=out.device)[1:-1])
            result[..., s * factor : e * factor] += out.float() * w
            weight[s * factor : e * factor] += w
            del out
        return (result / weight).to(dtype=outputs[0].dtype, device=z.device)

    wan.WanVAE_.decode = tiled_decode
