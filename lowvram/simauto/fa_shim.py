"""PyTorch stand-in for FlashAttention-4's varlen forward, used by Cosmos3 Sim-Auto's
"gapped maskless replay" attention on GPUs without FA4 (anything below Hopper).

cosmos_framework/model/generator/mot/maskless_attention.py calls
``flash_attn.cute.interface._flash_attn_fwd`` once per launch with packed varlen segments:

    q   [Nq, H, D]   k, v [Nk, H, D]          (GQA already folded into query rows)
    segment i: q rows  [cu_q[i], cu_q[i] + used_q[i])  attend (non-causal) to
               k rows  [cu_k[i], cu_k[i] + used_k[i])
    writes out[q rows] ([Nq, H, Dv]) and lse[:, q rows] ([H, Nq], natural log, fp32);
    every other row is left untouched.

This module reproduces that contract per segment with PyTorch's memory-efficient attention
kernel (``aten._scaled_dot_product_efficient_attention``, available on sm_80+), which returns the
log-sum-exp without materializing the score matrix. Heads are processed in groups to bound the
temporaries. ``SIM_AUTO_ATTN=exact`` selects the older fp32 matmul path (exact, more memory).
"""

from __future__ import annotations

import os
import types

import torch

# Max fp32 score-matrix elements materialized per chunk (~256 MB), exact path only.
_SCORE_BUDGET = 64 * 1024 * 1024
# Heads per efficient-attention launch.
_HEAD_GROUP = int(os.environ.get("SIM_AUTO_ATTN_HEAD_GROUP", "8"))
_EXACT = os.environ.get("SIM_AUTO_ATTN", "") == "exact"


def _segment_efficient(q, k, v, scale):
    """q [n,H,D], k/v [m,H,D] -> out [n,H,Dv], lse [H,n] (fp32)."""
    n, heads = q.shape[0], q.shape[1]
    outs, lses = [], []
    for h0 in range(0, heads, _HEAD_GROUP):
        h1 = min(heads, h0 + _HEAD_GROUP)
        qh = q[:, h0:h1].transpose(0, 1).unsqueeze(0)  # [1,h,n,D]
        kh = k[:, h0:h1].transpose(0, 1).unsqueeze(0)
        vh = v[:, h0:h1].transpose(0, 1).unsqueeze(0)
        o, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
            qh, kh, vh, None, True, 0.0, False, scale=scale
        )
        outs.append(o[0])  # [h,n,Dv]
        lses.append(lse[0, :, :n])  # lse is padded to a multiple of 32
    return torch.cat(outs).transpose(0, 1), torch.cat(lses)


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    return_lse: bool = True,
    causal: bool = False,
    **unused,
):
    if causal:
        raise NotImplementedError("fa_shim only implements the non-causal varlen forward used by gapped replay")
    heads, dim = q.shape[1], q.shape[2]
    scale = softmax_scale if softmax_scale is not None else dim**-0.5
    if out is None:
        out = q.new_zeros(q.shape[0], heads, v.shape[-1])
    if lse is None:
        lse = q.new_full((heads, q.shape[0]), -torch.inf, dtype=torch.float32)

    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    uq = seqused_q.tolist() if seqused_q is not None else [cq[i + 1] - cq[i] for i in range(len(cq) - 1)]
    uk = seqused_k.tolist() if seqused_k is not None else [ck[i + 1] - ck[i] for i in range(len(ck) - 1)]

    for i in range(len(uq)):
        nq, nk = int(uq[i]), int(uk[i])
        if nq <= 0 or nk <= 0:
            continue  # nothing to attend: keep zero output / -inf lse
        q0, k0 = int(cq[i]), int(ck[i])
        if not _EXACT and q.dtype in (torch.float16, torch.bfloat16):
            o, row_lse = _segment_efficient(q[q0 : q0 + nq], k[k0 : k0 + nk], v[k0 : k0 + nk], scale)
            out[q0 : q0 + nq] = o.to(out.dtype)
            lse[:, q0 : q0 + nq] = row_lse
            del o, row_lse
            continue
        ks = k[k0 : k0 + nk].transpose(0, 1).float()  # [H, m, D]
        vs = v[k0 : k0 + nk].transpose(0, 1).float()  # [H, m, Dv]
        step = max(16, _SCORE_BUDGET // (heads * nk))
        for s0 in range(0, nq, step):
            s1 = min(nq, s0 + step)
            qs = q[q0 + s0 : q0 + s1].transpose(0, 1).float()  # [H, n, D]
            scores = torch.matmul(qs, ks.transpose(1, 2)).mul_(scale)  # [H, n, m]
            row_lse = torch.logsumexp(scores, dim=-1)  # [H, n]
            probs = scores.sub_(row_lse.unsqueeze(-1)).exp_()
            o = torch.matmul(probs, vs)  # [H, n, Dv]
            out[q0 + s0 : q0 + s1] = o.transpose(0, 1).to(out.dtype)
            lse[:, q0 + s0 : q0 + s1] = row_lse
            del scores, probs, o, qs
        del ks, vs
    return out, lse


interface = types.ModuleType("fa_shim_interface")
interface._flash_attn_fwd = _flash_attn_fwd


def install() -> None:
    """Route cosmos_framework's gapped replay through this shim on non-Hopper/Blackwell GPUs."""
    from cosmos_framework.model.generator.mot import maskless_attention as m

    def unavailable_reason(device: torch.device) -> str | None:
        if device.type != "cuda":
            return f"it needs a CUDA device, got {device}"
        return None

    m._flash_attn_interface = lambda: interface
    m.gapped_replay_unavailable_reason = unavailable_reason
