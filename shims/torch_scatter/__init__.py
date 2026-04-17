"""Minimal torch_scatter shim for UniRig inference.

Implements only ``segment_csr`` with ``reduce in {sum, mean, min, max}``.
This is used by UniRig's skin inference path for per-segment reductions.
"""

from __future__ import annotations

import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    if src.ndim < 1:
        raise ValueError(f"Expected src.ndim >= 1, got {src.ndim}")
    if indptr.ndim != 1:
        raise ValueError(f"Expected indptr shape (S+1,), got {tuple(indptr.shape)}")
    if indptr.numel() < 2:
        raise ValueError("indptr must have at least two elements")

    reduce = reduce.lower()
    if reduce not in {"sum", "mean", "min", "max"}:
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    out = []
    for i in range(indptr.numel() - 1):
        s = int(indptr[i].item())
        e = int(indptr[i + 1].item())
        if s < 0 or e < s or e > src.shape[0]:
            raise ValueError(f"Invalid indptr segment [{s}, {e}) for src.shape[0]={src.shape[0]}")

        seg = src[s:e]
        if seg.numel() == 0:
            # Keep behavior simple and deterministic for empty segments.
            out.append(torch.zeros(src.shape[1:], dtype=src.dtype, device=src.device))
            continue

        if reduce == "sum":
            out.append(seg.sum(dim=0))
        elif reduce == "mean":
            out.append(seg.mean(dim=0))
        elif reduce == "min":
            out.append(seg.min(dim=0).values)
        else:  # max
            out.append(seg.max(dim=0).values)

    return torch.stack(out, dim=0)


__all__ = ["segment_csr"]
