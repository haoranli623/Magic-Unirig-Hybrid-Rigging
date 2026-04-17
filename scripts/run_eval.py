#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str((ROOT / ".hf_cache").resolve()))
sys.path.insert(0, str(ROOT))

from src.dataset import DT4D  # noqa: E402
from src.fit_adam import AdamFitConfig, apply_motion_params_adam, fit_sequence_adam  # noqa: E402
from src.mesh_utils import safe_sample_id  # noqa: E402
from src.metrics import evaluate_sequence  # noqa: E402
from src.optimizer import OptimizeConfig, apply_motion_params, optimize_sequence  # noqa: E402
from src.protocol import (  # noqa: E402
    build_cross_motion_tasks,
    build_cross_motion_tasks_all_val_targets,
    build_train_recon_tasks,
    protocol_note_text,
)
from src.reporting import write_json, write_results_csv, write_table1_style_report  # noqa: E402
from src.rig_format import load_rig  # noqa: E402


def _load_rig_for_source(source_sample_id: str) -> tuple[Path, Path, Path]:
    safe = safe_sample_id(source_sample_id)
    rig_dir = ROOT / "outputs" / "rigs" / safe
    rig_json = rig_dir / "rig.json"
    weights_npy = rig_dir / "weights.npy"
    return rig_json, weights_npy, rig_dir


def _copy_weight_reports(rig_dir: Path, task_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cfg = None
    rep = None

    cfg_path = rig_dir / "weight_transfer_config.json"
    rep_path = rig_dir / "weight_transfer_report.json"

    if cfg_path.exists():
        shutil.copy2(cfg_path, task_dir / "weight_transfer_config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if rep_path.exists():
        shutil.copy2(rep_path, task_dir / "weight_transfer_report.json")
        rep = json.loads(rep_path.read_text(encoding="utf-8"))

    return cfg, rep


def _save_task_outputs(
    task_dir: Path,
    *,
    recon: np.ndarray,
    metric: dict[str, Any],
    fit_config: dict[str, Any],
    task_summary: dict[str, Any],
    axis_angle: np.ndarray | None,
    root_trans: np.ndarray | None,
    motion_legacy: np.ndarray | None,
    loss_trace: Any,
    frame_summary: Any,
    legacy_log: Any,
) -> None:
    np.save(task_dir / "recon_vertices.npy", recon)
    write_json(task_dir / "metrics.json", metric)
    write_json(task_dir / "fit_config.json", fit_config)
    write_json(task_dir / "task_summary.json", task_summary)

    if axis_angle is not None and root_trans is not None:
        np.savez_compressed(task_dir / "motion_params_adam.npz", axis_angle=axis_angle, root_trans=root_trans)
    if motion_legacy is not None:
        np.savez_compressed(task_dir / "motion_params.npz", motion_params=motion_legacy)

    if loss_trace is not None:
        write_json(task_dir / "loss_trace.json", loss_trace)
    if frame_summary is not None:
        write_json(task_dir / "fit_frame_summary.json", frame_summary)
    if legacy_log is not None:
        write_json(task_dir / "opt_log.json", legacy_log)


def _reset_task_artifacts(task_dir: Path) -> None:
    for name in [
        "recon_vertices.npy",
        "metrics.json",
        "fit_config.json",
        "task_summary.json",
        "motion_params_adam.npz",
        "motion_params.npz",
        "loss_trace.json",
        "fit_frame_summary.json",
        "opt_log.json",
        "weight_transfer_config.json",
        "weight_transfer_report.json",
    ]:
        p = task_dir / name
        if p.exists():
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid Magic+UniRig-style evaluation on DT4D")
    parser.add_argument("--h5", type=Path, default=ROOT.parent / "dt4d.hdf5")
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--cross-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-allowlist",
        type=Path,
        default=None,
        help="Optional JSON file containing an explicit list of source sample ids to use as train/source pool.",
    )
    parser.add_argument(
        "--allow-missing-rigs",
        action="store_true",
        default=False,
        help="Do not pre-filter train sources by available rigs (tasks without rigs will be skipped later)",
    )
    parser.add_argument("--cross-policy", choices=["fixed_rig_opt_test", "direct_transfer"], default="fixed_rig_opt_test")
    parser.add_argument(
        "--cross-enum",
        choices=["per_character", "all_val_targets"],
        default="per_character",
        help="Cross-motion task enumeration: one pair per character or one task per val target.",
    )

    parser.add_argument("--optimizer-backend", choices=["adam", "scipy_legacy"], default="adam")

    # Adam defaults (hybrid default)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--trace-every", type=int, default=25)
    parser.add_argument("--vertex-sample-count", type=int, default=1500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rot-l2", type=float, default=0.0)
    parser.add_argument("--root-l2", type=float, default=0.0)
    parser.add_argument("--temporal-smooth", type=float, default=0.0)

    # Legacy scipy fallback
    parser.add_argument("--max-nfev", type=int, default=30)

    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs" / "eval")
    parser.add_argument("--results-json", type=Path, default=ROOT / "outputs" / "eval" / "results.json")
    parser.add_argument("--results-csv", type=Path, default=ROOT / "outputs" / "eval" / "results.csv")
    parser.add_argument("--table-report", type=Path, default=ROOT / "reports" / "table1_magicarticulate_repro.md")
    parser.add_argument("--protocol-report", type=Path, default=ROOT / "reports" / "protocol.md")
    args = parser.parse_args()

    ds = DT4D(args.h5)
    train_ids = ds.list_split_sample_ids("train")
    val_ids = ds.list_split_sample_ids("val")

    if args.source_allowlist is not None:
        if not args.source_allowlist.exists():
            raise FileNotFoundError(f"source allowlist file not found: {args.source_allowlist}")
        payload = json.loads(args.source_allowlist.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            allow = payload.get("allowlist_sample_ids", payload.get("sample_ids", []))
        elif isinstance(payload, list):
            allow = payload
        else:
            raise ValueError("source allowlist must be a JSON list or a JSON object with allowlist_sample_ids")
        allow_set = set(str(x) for x in allow)
        train_ids = [sid for sid in train_ids if sid in allow_set]

    if not args.allow_missing_rigs:
        rigged = []
        for sid in train_ids:
            safe = safe_sample_id(sid)
            rig_json = ROOT / "outputs" / "rigs" / safe / "rig.json"
            weights_npy = ROOT / "outputs" / "rigs" / safe / "weights.npy"
            if rig_json.exists() and weights_npy.exists():
                rigged.append(sid)
        train_ids = rigged

    train_tasks = build_train_recon_tasks(train_ids, limit=args.train_limit, seed=args.seed)
    if args.cross_enum == "all_val_targets":
        cross_tasks = build_cross_motion_tasks_all_val_targets(train_ids, val_ids, limit=args.cross_limit, seed=args.seed)
    else:
        cross_tasks = build_cross_motion_tasks(train_ids, val_ids, limit=args.cross_limit, seed=args.seed)
    tasks = train_tasks + cross_tasks

    per_task_root = args.outputs_dir / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    adam_cfg = AdamFitConfig(
        n_iters=args.n_iters,
        lr=args.lr,
        trace_every=args.trace_every,
        vertex_sample_count=args.vertex_sample_count,
        random_seed=args.seed,
        device=args.device,
        rot_l2=args.rot_l2,
        root_l2=args.root_l2,
        temporal_smooth=args.temporal_smooth,
    )

    scipy_cfg = OptimizeConfig(
        max_nfev=args.max_nfev,
        temporal_smooth=args.temporal_smooth,
        vertex_sample_count=args.vertex_sample_count,
        random_seed=args.seed,
    )

    results = []
    for task in tqdm(tasks, desc="Evaluating tasks"):
        source_id = task["source_sample_id"]
        target_id = task["target_sample_id"]
        task_type = task["task_type"]

        source = ds.load_sample(source_id)
        target = ds.load_sample(target_id)

        rig_json, weights_npy, rig_dir = _load_rig_for_source(source_id)
        if not rig_json.exists() or not weights_npy.exists():
            results.append(
                {
                    **task,
                    "status": "skipped",
                    "cross_policy": args.cross_policy if task_type == "cross_motion" else "fixed_rig_opt_test",
                    "optimizer_backend": args.optimizer_backend,
                    "notes": f"missing rig files for source: {rig_json} / {weights_npy}",
                }
            )
            continue

        rig = load_rig(rig_json, weights_npy)
        if rig.weights.shape[0] != source.num_vertices:
            results.append(
                {
                    **task,
                    "status": "skipped",
                    "cross_policy": args.cross_policy if task_type == "cross_motion" else "fixed_rig_opt_test",
                    "optimizer_backend": args.optimizer_backend,
                    "notes": f"rig weights vertex mismatch ({rig.weights.shape[0]} vs {source.num_vertices})",
                }
            )
            continue

        if source.num_vertices != target.num_vertices:
            results.append(
                {
                    **task,
                    "status": "skipped",
                    "cross_policy": args.cross_policy if task_type == "cross_motion" else "fixed_rig_opt_test",
                    "optimizer_backend": args.optimizer_backend,
                    "notes": f"topology mismatch source/target ({source.num_vertices} vs {target.num_vertices})",
                }
            )
            continue

        task_tag = f"{safe_sample_id(source_id)}__to__{safe_sample_id(target_id)}"
        task_dir = per_task_root / task_tag
        task_dir.mkdir(parents=True, exist_ok=True)
        _reset_task_artifacts(task_dir)

        wt_cfg, wt_report = _copy_weight_reports(rig_dir, task_dir)
        weight_method = wt_report.get("weight_method") if wt_report else "unknown"

        fit_config: dict[str, Any]
        axis_angle = None
        root_trans = None
        motion_legacy = None
        loss_trace = None
        frame_summary = None
        legacy_log = None

        if args.optimizer_backend == "adam":
            fit_config = {"optimizer_backend": "adam", **adam_cfg.__dict__}
            if task_type == "cross_motion" and args.cross_policy == "direct_transfer":
                fit_src = fit_sequence_adam(
                    rig=rig,
                    canonical_vertices=source.canonical_vertices,
                    target_sequence=source.vertices,
                    config=adam_cfg,
                    show_progress=False,
                )
                t = min(fit_src["axis_angle"].shape[0], target.num_frames)
                axis_angle = fit_src["axis_angle"][:t]
                root_trans = fit_src["root_trans"][:t]
                recon = apply_motion_params_adam(
                    rig=rig,
                    canonical_vertices=source.canonical_vertices,
                    axis_angle=axis_angle,
                    root_trans=root_trans,
                    device=args.device,
                )
                target_eval = target.vertices[:t]
                loss_trace = {"format": "sampled", "frames": fit_src["loss_trace"]}
                frame_summary = {"frames": fit_src["fit_frame_summary"]}
                notes = "direct_transfer: optimized source motion then replayed on target"
            else:
                fit = fit_sequence_adam(
                    rig=rig,
                    canonical_vertices=source.canonical_vertices,
                    target_sequence=target.vertices,
                    config=adam_cfg,
                    show_progress=False,
                )
                axis_angle = fit["axis_angle"]
                root_trans = fit["root_trans"]
                recon = fit["recon_vertices"]
                target_eval = target.vertices
                loss_trace = {"format": "sampled", "frames": fit["loss_trace"]}
                frame_summary = {"frames": fit["fit_frame_summary"]}
                notes = "fixed_rig_opt_test: optimized transforms on target with source rig fixed (adam)"
        else:
            fit_config = {"optimizer_backend": "scipy_legacy", **scipy_cfg.__dict__}
            if task_type == "cross_motion" and args.cross_policy == "direct_transfer":
                fit_src = optimize_sequence(
                    rig=rig,
                    canonical_vertices=source.canonical_vertices,
                    target_sequence=source.vertices,
                    config=scipy_cfg,
                    show_progress=False,
                )
                motion = fit_src["motion_params"]
                t = min(len(motion), target.num_frames)
                motion_legacy = motion[:t]
                recon = apply_motion_params(rig=rig, canonical_vertices=source.canonical_vertices, motion_params=motion_legacy)
                target_eval = target.vertices[:t]
                legacy_log = fit_src["opt_log"]
                notes = "direct_transfer: optimized source motion then replayed on target"
            else:
                fit = optimize_sequence(
                    rig=rig,
                    canonical_vertices=source.canonical_vertices,
                    target_sequence=target.vertices,
                    config=scipy_cfg,
                    show_progress=False,
                )
                motion_legacy = fit["motion_params"]
                recon = fit["recon_vertices"]
                target_eval = target.vertices
                legacy_log = fit["opt_log"]
                notes = "fixed_rig_opt_test: optimized transforms on target with source rig fixed (scipy legacy)"

        metric = evaluate_sequence(recon, target_eval, compute_vertex_l2=True)

        task_summary = {
            "task_type": task_type,
            "source_sample_id": source_id,
            "target_sample_id": target_id,
            "cross_policy": args.cross_policy if task_type == "cross_motion" else "fixed_rig_opt_test",
            "optimizer_backend": args.optimizer_backend,
            "weight_method": weight_method,
            "fit_config_path": str((task_dir / "fit_config.json").resolve()),
            "metrics": metric["aggregate"],
            "notes": notes,
        }

        _save_task_outputs(
            task_dir,
            recon=recon,
            metric=metric,
            fit_config=fit_config,
            task_summary=task_summary,
            axis_angle=axis_angle,
            root_trans=root_trans,
            motion_legacy=motion_legacy,
            loss_trace=loss_trace,
            frame_summary=frame_summary,
            legacy_log=legacy_log,
        )

        row = {
            **task,
            "status": "ok",
            "cross_policy": args.cross_policy if task_type == "cross_motion" else "fixed_rig_opt_test",
            "optimizer_backend": args.optimizer_backend,
            "weight_method": weight_method,
            "n_iters": args.n_iters if args.optimizer_backend == "adam" else None,
            "lr": args.lr if args.optimizer_backend == "adam" else None,
            "trace_every": args.trace_every if args.optimizer_backend == "adam" else None,
            "metrics": metric["aggregate"],
            "notes": notes,
        }
        results.append(row)

    payload = {
        "h5_path": str(args.h5.resolve()),
        "default_pipeline": {
            "skeleton_source": "MagicArticulate",
            "weight_transfer": "UniRig learned skin + UniRig-style remap/reskin",
            "optimizer_backend": "adam",
        },
        "optimizer_backend": args.optimizer_backend,
        "cross_policy": args.cross_policy,
        "cross_enum": args.cross_enum,
        "protocol_note": protocol_note_text(),
        "num_tasks": len(tasks),
        "results": results,
    }

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_json, payload)
    write_results_csv(args.results_csv, results)
    write_table1_style_report(args.table_report, results)

    protocol_lines = [
        "# Protocol",
        "",
        "## Hybrid Default (This Repo)",
        "- Skeleton source: MagicArticulate public demo output.",
        "- Weight stage: UniRig learned skin prediction, then UniRig-style sampled-space -> canonical reskin/remap.",
        "- Optimization default: Torch + Adam per-frame fitting with warm start.",
        "",
        "## Paper-confirmed behavior (RigMo)",
        "- Table 1 evaluates Training Reconstruction and Cross-Motion Transfer on DT4D test split.",
        "- Auto-rigging baselines generate initial bone structures from canonical pose followed by per-sequence transform optimization.",
        "",
        "## Reproduction implementation",
        f"- Train reconstruction tasks: {len(train_tasks)} sampled from `/data_split/train`.",
        f"- Cross-motion tasks: {len(cross_tasks)} sampled as (source in train, target in val) for same character.",
        f"- Cross-motion enumeration: `{args.cross_enum}`.",
        f"- Cross-motion policy used: `{args.cross_policy}`.",
        f"- Optimizer backend used: `{args.optimizer_backend}`.",
    ]
    args.protocol_report.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_report.write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"Tasks done: {len(ok)}/{len(tasks)} successful")
    print(f"Results JSON: {args.results_json}")
    print(f"Results CSV:  {args.results_csv}")
    print(f"Table report: {args.table_report}")


if __name__ == "__main__":
    main()
