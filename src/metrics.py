from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def chamfer_l1_l2(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)
    d_ba, _ = tree_a.query(b, k=1)
    cd_l1 = float(np.mean(d_ab) + np.mean(d_ba))
    cd_l2 = float(np.mean(d_ab**2) + np.mean(d_ba**2))
    return cd_l1, cd_l2


def vertex_l2(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    d = np.linalg.norm(pred - target, axis=-1)
    return {
        "v_l2_mean": float(np.mean(d)),
        "v_l2_median": float(np.median(d)),
        "v_l2_max": float(np.max(d)),
    }


def evaluate_sequence(
    pred_seq: np.ndarray,
    target_seq: np.ndarray,
    compute_vertex_l2: bool = True,
) -> dict[str, Any]:
    if pred_seq.shape != target_seq.shape:
        raise ValueError(f"Shape mismatch in sequence metrics: {pred_seq.shape} vs {target_seq.shape}")

    per_frame: list[dict[str, Any]] = []
    for t in range(pred_seq.shape[0]):
        cd_l1, cd_l2 = chamfer_l1_l2(pred_seq[t], target_seq[t])
        row: dict[str, Any] = {
            "frame": int(t),
            "cd_l1": cd_l1,
            "cd_l2": cd_l2,
        }
        if compute_vertex_l2:
            row.update(vertex_l2(pred_seq[t], target_seq[t]))
        per_frame.append(row)

    agg = {
        "num_frames": int(pred_seq.shape[0]),
        "cd_l1": float(np.mean([r["cd_l1"] for r in per_frame])),
        "cd_l2": float(np.mean([r["cd_l2"] for r in per_frame])),
    }
    if compute_vertex_l2:
        agg.update(
            {
                "v_l2_mean": float(np.mean([r["v_l2_mean"] for r in per_frame])),
                "v_l2_median": float(np.mean([r["v_l2_median"] for r in per_frame])),
                "v_l2_max": float(np.max([r["v_l2_max"] for r in per_frame])),
            }
        )

    return {
        "aggregate": agg,
        "per_frame": per_frame,
    }
