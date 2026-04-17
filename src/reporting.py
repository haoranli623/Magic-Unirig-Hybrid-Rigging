from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_type",
        "character_id",
        "source_sample_id",
        "target_sample_id",
        "status",
        "cross_policy",
        "optimizer_backend",
        "weight_method",
        "n_iters",
        "lr",
        "trace_every",
        "cd_l1",
        "cd_l2",
        "v_l2_mean",
        "num_frames",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            m = r.get("metrics", {})
            w.writerow(
                {
                    "task_type": r.get("task_type"),
                    "character_id": r.get("character_id"),
                    "source_sample_id": r.get("source_sample_id"),
                    "target_sample_id": r.get("target_sample_id"),
                    "status": r.get("status"),
                    "cross_policy": r.get("cross_policy"),
                    "optimizer_backend": r.get("optimizer_backend"),
                    "weight_method": r.get("weight_method"),
                    "n_iters": r.get("n_iters"),
                    "lr": r.get("lr"),
                    "trace_every": r.get("trace_every"),
                    "cd_l1": m.get("cd_l1"),
                    "cd_l2": m.get("cd_l2"),
                    "v_l2_mean": m.get("v_l2_mean"),
                    "num_frames": m.get("num_frames"),
                    "notes": r.get("notes", ""),
                }
            )


def summarize_metrics(rows: list[dict[str, Any]], task_type: str) -> dict[str, float | None]:
    ok = [r for r in rows if r.get("task_type") == task_type and r.get("status") == "ok"]
    if not ok:
        return {"cd_l1": None, "cd_l2": None}
    cd_l1 = float(np.mean([r["metrics"]["cd_l1"] for r in ok]))
    cd_l2 = float(np.mean([r["metrics"]["cd_l2"] for r in ok]))
    return {"cd_l1": cd_l1, "cd_l2": cd_l2}


def write_table1_style_report(path: Path, rows: list[dict[str, Any]]) -> None:
    train = summarize_metrics(rows, "train_recon")
    cross = summarize_metrics(rows, "cross_motion")

    if train["cd_l1"] is None or cross["cd_l1"] is None:
        mean_l1 = None
        mean_l2 = None
    else:
        mean_l1 = 0.5 * (train["cd_l1"] + cross["cd_l1"])
        mean_l2 = 0.5 * (train["cd_l2"] + cross["cd_l2"])

    def fmt(v: float | None, scale: float = 1.0) -> str:
        if v is None:
            return "N/A"
        return f"{v * scale:.3f}"

    md = [
        "# Table 1 Style: Hybrid (Magic Skeleton + UniRig-Style Reskin + Adam Opt.)",
        "",
        "Values below are reported as Chamfer metrics in units of ×10^-3 (i.e., raw metric × 1000).",
        "",
        "| Method | Training Reconstruction CD-L1 | Training Reconstruction CD-L2 | Cross-Motion Transfer CD-L1 | Cross-Motion Transfer CD-L2 | Mean CD-L1 | Mean CD-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Hybrid Magic+UniRig-style downstream (this reproduction) "
        f"| {fmt(train['cd_l1'], 1e3)} | {fmt(train['cd_l2'], 1e3)} "
        f"| {fmt(cross['cd_l1'], 1e3)} | {fmt(cross['cd_l2'], 1e3)} "
        f"| {fmt(mean_l1, 1e3)} | {fmt(mean_l2, 1e3)} |",
        "",
        "## Notes",
        "- Training and cross-motion means are computed over successful tasks only.",
        "- See `outputs/eval/results.json` for per-task details and skipped/failed cases.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
