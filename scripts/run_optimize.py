#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str((ROOT / ".hf_cache").resolve()))
sys.path.insert(0, str(ROOT))

from src.dataset import DT4D  # noqa: E402
from src.fit_adam import AdamFitConfig, apply_motion_params_adam, fit_sequence_adam  # noqa: E402
from src.mesh_utils import safe_sample_id  # noqa: E402
from src.optimizer import OptimizeConfig, apply_motion_params, optimize_sequence  # noqa: E402
from src.rig_format import load_rig  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_rig_weight_reports(src_safe: str, pair_dir: Path) -> None:
    rig_dir = ROOT / "outputs" / "rigs" / src_safe
    for name in ["weight_transfer_config.json", "weight_transfer_report.json"]:
        p = rig_dir / name
        if p.exists():
            shutil.copy2(p, pair_dir / name)


def _reset_pair_artifacts(pair_dir: Path) -> None:
    for name in [
        "recon_vertices.npy",
        "motion_params_adam.npz",
        "motion_params.npz",
        "loss_trace.json",
        "fit_frame_summary.json",
        "opt_log.json",
        "fit_config.json",
        "meta.json",
        "weight_transfer_config.json",
        "weight_transfer_report.json",
    ]:
        p = pair_dir / name
        if p.exists():
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rig-conditioned sequence optimization")
    parser.add_argument("--h5", type=Path, default=ROOT.parent / "dt4d.hdf5")
    parser.add_argument("--source-sample-id", type=str, required=True)
    parser.add_argument("--target-sample-id", type=str, required=True)
    parser.add_argument("--cross-policy", choices=["fixed_rig_opt_test", "direct_transfer"], default="fixed_rig_opt_test")
    parser.add_argument("--rig-json", type=Path, default=None)
    parser.add_argument("--weights-npy", type=Path, default=None)

    parser.add_argument("--optimizer-backend", choices=["adam", "scipy_legacy"], default="adam")

    # Adam defaults (new hybrid default)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--trace-every", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vertex-sample-count", type=int, default=1500)
    parser.add_argument("--rot-l2", type=float, default=0.0)
    parser.add_argument("--root-l2", type=float, default=0.0)
    parser.add_argument("--temporal-smooth", type=float, default=0.0)

    # Legacy SciPy knobs
    parser.add_argument("--max-nfev", type=int, default=40)
    args = parser.parse_args()

    ds = DT4D(args.h5)
    src_id = ds.resolve_sample_id(args.source_sample_id)
    tgt_id = ds.resolve_sample_id(args.target_sample_id)
    src = ds.load_sample(src_id)
    tgt = ds.load_sample(tgt_id)

    src_safe = safe_sample_id(src_id)
    tgt_safe = safe_sample_id(tgt_id)
    pair_dir = ROOT / "outputs" / "opt" / f"{src_safe}__to__{tgt_safe}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    _reset_pair_artifacts(pair_dir)

    rig_json = args.rig_json or ROOT / "outputs" / "rigs" / src_safe / "rig.json"
    weights_npy = args.weights_npy or ROOT / "outputs" / "rigs" / src_safe / "weights.npy"
    if not Path(rig_json).exists() or not Path(weights_npy).exists():
        raise FileNotFoundError(
            f"Missing rig files for source {src_id}. Expected {rig_json} and {weights_npy}"
        )

    rig = load_rig(Path(rig_json), Path(weights_npy))

    if rig.weights.shape[0] != src.num_vertices:
        raise ValueError(
            f"Rig weights vertex count {rig.weights.shape[0]} does not match source canonical vertices {src.num_vertices}"
        )

    if src.num_vertices != tgt.num_vertices:
        raise ValueError(
            f"Source/target vertex count mismatch ({src.num_vertices} vs {tgt.num_vertices}); cannot run direct topology transfer"
        )

    _copy_rig_weight_reports(src_safe, pair_dir)

    fit_cfg: dict[str, object]
    if args.optimizer_backend == "adam":
        cfg = AdamFitConfig(
            n_iters=args.n_iters,
            lr=args.lr,
            trace_every=args.trace_every,
            vertex_sample_count=args.vertex_sample_count,
            random_seed=0,
            device=args.device,
            rot_l2=args.rot_l2,
            root_l2=args.root_l2,
            temporal_smooth=args.temporal_smooth,
        )
        fit_cfg = {"optimizer_backend": "adam", **cfg.__dict__}

        if args.cross_policy == "fixed_rig_opt_test":
            out = fit_sequence_adam(
                rig=rig,
                canonical_vertices=src.canonical_vertices,
                target_sequence=tgt.vertices,
                config=cfg,
                show_progress=True,
            )
            recon = out["recon_vertices"]
            axis_angle = out["axis_angle"]
            root_trans = out["root_trans"]
            loss_trace = out["loss_trace"]
            frame_summary = out["fit_frame_summary"]
        else:
            fit_src = fit_sequence_adam(
                rig=rig,
                canonical_vertices=src.canonical_vertices,
                target_sequence=src.vertices,
                config=cfg,
                show_progress=True,
            )
            t = min(fit_src["axis_angle"].shape[0], tgt.num_frames)
            axis_angle = fit_src["axis_angle"][:t]
            root_trans = fit_src["root_trans"][:t]
            recon = apply_motion_params_adam(
                rig=rig,
                canonical_vertices=src.canonical_vertices,
                axis_angle=axis_angle,
                root_trans=root_trans,
                device=args.device,
            )
            loss_trace = fit_src["loss_trace"]
            frame_summary = fit_src["fit_frame_summary"]

        np.save(pair_dir / "recon_vertices.npy", recon)
        np.savez_compressed(pair_dir / "motion_params_adam.npz", axis_angle=axis_angle, root_trans=root_trans)
        _write_json(pair_dir / "loss_trace.json", {"format": "sampled", "frames": loss_trace})
        _write_json(pair_dir / "fit_frame_summary.json", {"frames": frame_summary})

    else:
        cfg = OptimizeConfig(
            max_nfev=args.max_nfev,
            vertex_sample_count=args.vertex_sample_count,
            temporal_smooth=args.temporal_smooth,
        )
        fit_cfg = {"optimizer_backend": "scipy_legacy", **cfg.__dict__}

        if args.cross_policy == "fixed_rig_opt_test":
            out = optimize_sequence(
                rig=rig,
                canonical_vertices=src.canonical_vertices,
                target_sequence=tgt.vertices,
                config=cfg,
                show_progress=True,
            )
            recon = out["recon_vertices"]
            motion = out["motion_params"]
            logs = out["opt_log"]
        else:
            fit_src = optimize_sequence(
                rig=rig,
                canonical_vertices=src.canonical_vertices,
                target_sequence=src.vertices,
                config=cfg,
                show_progress=True,
            )
            motion_src = fit_src["motion_params"]
            t = min(len(motion_src), tgt.num_frames)
            motion = motion_src[:t]
            recon = apply_motion_params(rig=rig, canonical_vertices=src.canonical_vertices, motion_params=motion)
            logs = fit_src["opt_log"]

        np.save(pair_dir / "recon_vertices.npy", recon)
        np.savez_compressed(pair_dir / "motion_params.npz", motion_params=motion)
        _write_json(pair_dir / "opt_log.json", logs)

    _write_json(pair_dir / "fit_config.json", fit_cfg)

    meta = {
        "source_sample_id": src_id,
        "target_sample_id": tgt_id,
        "cross_policy": args.cross_policy,
        "optimizer_backend": args.optimizer_backend,
        "rig_json": str(Path(rig_json).resolve()),
        "weights_npy": str(Path(weights_npy).resolve()),
        "num_frames": int(recon.shape[0]),
    }
    _write_json(pair_dir / "meta.json", meta)

    print(f"Optimization finished: {src_id} -> {tgt_id}")
    print(f"Backend: {args.optimizer_backend}")
    print(f"Saved: {pair_dir}")


if __name__ == "__main__":
    main()
