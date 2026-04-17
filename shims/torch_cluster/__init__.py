"""Minimal torch_cluster shim.

Implements only ``fps`` used by UniRig's Michelangelo encoder.
"""

from __future__ import annotations

import torch


def _fps_single(pos: torch.Tensor, m: int, random_start: bool) -> torch.Tensor:
    n = pos.shape[0]
    if m >= n:
        return torch.arange(n, device=pos.device, dtype=torch.long)

    selected = torch.empty((m,), device=pos.device, dtype=torch.long)
    if random_start:
        farthest = torch.randint(0, n, (1,), device=pos.device, dtype=torch.long).item()
    else:
        farthest = 0
    selected[0] = farthest

    dist = torch.full((n,), float("inf"), device=pos.device, dtype=pos.dtype)
    centroid = pos[farthest : farthest + 1]
    dist = torch.minimum(dist, torch.sum((pos - centroid) ** 2, dim=1))

    for i in range(1, m):
        farthest = int(torch.argmax(dist).item())
        selected[i] = farthest
        centroid = pos[farthest : farthest + 1]
        dist = torch.minimum(dist, torch.sum((pos - centroid) ** 2, dim=1))

    return selected


def fps(pos: torch.Tensor, batch: torch.Tensor, ratio: float = 0.5, random_start: bool = True) -> torch.Tensor:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected pos shape (N,3), got {tuple(pos.shape)}")
    if batch.ndim != 1 or batch.shape[0] != pos.shape[0]:
        raise ValueError(f"Expected batch shape (N,), got {tuple(batch.shape)}")

    batch = batch.to(torch.long)
    out_idx = []
    unique_batches = torch.unique(batch)
    for b in unique_batches:
        mask = batch == b
        local_idx = torch.nonzero(mask, as_tuple=False).view(-1)
        local_pos = pos[local_idx]
        m = max(1, int(round(local_pos.shape[0] * float(ratio))))
        sel_local = _fps_single(local_pos, m=m, random_start=random_start)
        out_idx.append(local_idx[sel_local])
    return torch.cat(out_idx, dim=0)


__all__ = ["fps"]
