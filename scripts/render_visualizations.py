#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DT4D  # noqa: E402
from src.mesh_utils import safe_sample_id  # noqa: E402
from src.rig_format import load_rig  # noqa: E402
from src.visualization import (  # noqa: E402
    compute_bounds,
    default_camera_pose,
    keyframe_indices,
    make_side_by_side,
    render_mesh_frames,
    render_rig_overlay,
    save_video,
    write_image,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_task_lookup(results_json: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(results_json)
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("results", []):
        src = row.get("source_sample_id")
        tgt = row.get("target_sample_id")
        if not src or not tgt:
            continue
        task_id = f"{safe_sample_id(src)}__to__{safe_sample_id(tgt)}"
        out[task_id] = row
    return out


def _resolve_task_row(task_id: str, results_json: Path) -> dict[str, Any]:
    lookup = _build_task_lookup(results_json)
    if task_id not in lookup:
        known = "\n".join(sorted(lookup.keys())[:20])
        raise ValueError(f"Task id not found: {task_id}\nKnown tasks (first 20):\n{known}")
    return lookup[task_id]


def _cap_frames(arr: np.ndarray, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames <= 0:
        return arr
    return arr[: min(len(arr), max_frames)]


def _center_for_viz(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if seq.ndim != 3 or seq.shape[-1] != 3:
        raise ValueError(f"Expected sequence (T,N,3), got {seq.shape}")
    center0 = np.mean(seq[0], axis=0, keepdims=True)
    centered = seq - center0[None, :, :]
    return centered.astype(np.float32), center0.reshape(3).astype(np.float32)


def _save_task_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_if_diff(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _task_file_name(task_type: str, base_name: str) -> str:
    return f"{task_type}__{base_name}"


def _flatten_loss_points(loss_trace_payload: dict[str, Any]) -> list[tuple[float, float]]:
    frames = loss_trace_payload.get("frames", [])
    points: list[tuple[float, float]] = []
    x = 0.0
    for fr in frames:
        p = fr.get("points", [])
        if not p:
            continue
        for item in p:
            y = float(item.get("loss_total", item.get("loss", 0.0)))
            points.append((x, y))
            x += 1.0
    return points


def _render_loss_curve(loss_trace_path: Path, out_image: Path, width: int = 960, height: int = 540) -> bool:
    if not loss_trace_path.exists():
        return False
    payload = json.loads(loss_trace_path.read_text(encoding="utf-8"))
    pts = _flatten_loss_points(payload)
    if len(pts) < 2:
        return False

    ys = np.array([p[1] for p in pts], dtype=np.float32)
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    pad_l, pad_r, pad_t, pad_b = 70, 30, 30, 55
    x0, y0 = pad_l, height - pad_b
    x1, y1 = width - pad_r, pad_t

    cv2.line(canvas, (x0, y0), (x1, y0), (80, 80, 80), 2)
    cv2.line(canvas, (x0, y0), (x0, y1), (80, 80, 80), 2)

    poly = []
    for i, (_x, y) in enumerate(pts):
        px = x0 + int((x1 - x0) * (i / max(1, len(pts) - 1)))
        py = y0 - int((y0 - y1) * ((y - y_min) / (y_max - y_min)))
        poly.append((px, py))
    cv2.polylines(canvas, [np.array(poly, dtype=np.int32)], False, (37, 99, 235), 2, lineType=cv2.LINE_AA)

    cv2.putText(canvas, "Optimization Loss Trace", (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"min={y_min:.4f} max={y_max:.4f}", (x0, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)

    write_image(out_image, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return True


def _render_task(
    *,
    ds: DT4D,
    task_id: str,
    row: dict[str, Any],
    out_dir: Path,
    per_task_dir: Path,
    fps: int,
    max_frames: int | None,
    width: int,
    height: int,
    turntable: bool,
) -> None:
    task_type = str(row.get("task_type", "unknown_task"))
    source_id = str(row["source_sample_id"])
    target_id = str(row["target_sample_id"])
    source_safe = safe_sample_id(source_id)

    recon_path = per_task_dir / task_id / "recon_vertices.npy"
    if not recon_path.exists():
        raise FileNotFoundError(f"Missing reconstruction for task: {recon_path}")

    source = ds.load_sample(source_id)
    target = ds.load_sample(target_id)
    recon = np.asarray(np.load(recon_path), dtype=np.float32)

    t = min(len(recon), target.num_frames)
    if max_frames is not None and max_frames > 0:
        t = min(t, max_frames)

    original_seq = target.vertices[:t]
    recon_seq = recon[:t]
    original_viz, original_center0 = _center_for_viz(original_seq)
    recon_viz, recon_center0 = _center_for_viz(recon_seq)

    _bmin, _bmax, center, scale = compute_bounds([original_viz, recon_viz])
    camera_pose = default_camera_pose(center=center, scale=scale)

    print(f"[viz] Rendering task {task_id} frames={t}")
    left_frames = render_mesh_frames(
        original_viz,
        target.faces,
        width=width,
        height=height,
        camera_pose=camera_pose,
        mesh_color=(110, 150, 230),
    )
    right_frames = render_mesh_frames(
        recon_viz,
        source.faces,
        width=width,
        height=height,
        camera_pose=camera_pose,
        mesh_color=(240, 155, 85),
    )

    original_video_typed = out_dir / _task_file_name(task_type, "original_mesh.mp4")
    recon_video_typed = out_dir / _task_file_name(task_type, "reconstruction_mesh.mp4")
    side_video_typed = out_dir / _task_file_name(task_type, "original_vs_reconstruction.mp4")
    loss_curve_typed = out_dir / _task_file_name(task_type, "loss_curve.png")
    rig_overlay_typed = out_dir / _task_file_name(task_type, "rig_overlay.png")
    rig_turntable_typed = out_dir / _task_file_name(task_type, "rig_turntable.mp4")

    save_video(original_video_typed, left_frames, fps=fps)
    save_video(recon_video_typed, right_frames, fps=fps)

    side = make_side_by_side(left_frames, right_frames)
    save_video(side_video_typed, side, fps=fps)

    for k in keyframe_indices(len(side)):
        if k == 0:
            name = "frame_000_compare.png"
        elif k == len(side) - 1:
            name = "frame_last_compare.png"
        else:
            name = "frame_mid_compare.png"
        write_image(out_dir / _task_file_name(task_type, name), side[k])

    loss_curve_written = _render_loss_curve(
        per_task_dir / task_id / "loss_trace.json",
        loss_curve_typed,
        width=width,
        height=height,
    )

    rig_json = ROOT / "outputs" / "rigs" / source_safe / "rig.json"
    weights_npy = ROOT / "outputs" / "rigs" / source_safe / "weights.npy"
    if rig_json.exists() and weights_npy.exists():
        rig = load_rig(rig_json, weights_npy)
        render_rig_overlay(
            canonical_vertices=source.canonical_vertices,
            canonical_faces=source.faces,
            joints=rig.joint_positions,
            parents=rig.parents,
            out_image=rig_overlay_typed,
            width=width,
            height=height,
            turntable_out=rig_turntable_typed if turntable else None,
            fps=fps,
            turntable_frames=72,
        )
    else:
        print(f"[viz] Skipping rig overlay, rig files missing for source: {source_safe}")

    # Backward-compatible unprefixed aliases.
    _copy_if_diff(original_video_typed, out_dir / "original_mesh.mp4")
    _copy_if_diff(recon_video_typed, out_dir / "reconstruction_mesh.mp4")
    _copy_if_diff(side_video_typed, out_dir / "original_vs_reconstruction.mp4")
    if loss_curve_written:
        _copy_if_diff(loss_curve_typed, out_dir / "loss_curve.png")
    if rig_overlay_typed.exists():
        _copy_if_diff(rig_overlay_typed, out_dir / "rig_overlay.png")
    if turntable and rig_turntable_typed.exists():
        _copy_if_diff(rig_turntable_typed, out_dir / "rig_turntable.mp4")
    for frame_name in ["frame_000_compare.png", "frame_mid_compare.png", "frame_last_compare.png"]:
        typed = out_dir / _task_file_name(task_type, frame_name)
        if typed.exists():
            _copy_if_diff(typed, out_dir / frame_name)

    task_semantics = {
        "task_type": task_type,
        "skeleton_source_sample_id": source_id,
        "target_gt_sample_id": target_id,
        "reconstruction_task_id": task_id,
        "skeleton_from_source": True,
        "gt_animation_from_target": True,
        "reconstruction_under_source_rig": True,
    }

    _save_task_meta(
        out_dir / "viz_meta.json",
        {
            "mode": "task",
            "task_type": task_type,
            "task_id": task_id,
            "source_sample_id": source_id,
            "target_sample_id": target_id,
            "semantics": task_semantics,
            "source_faces_from": f"h5:{source.sample_id}",
            "target_faces_from": f"h5:{target.sample_id}",
            "recon_vertices": str(recon_path.resolve()),
            "rig_json": str(rig_json.resolve()) if rig_json.exists() else None,
            "weights_npy": str(weights_npy.resolve()) if weights_npy.exists() else None,
            "frames_used": int(t),
            "fps": int(fps),
            "camera": "shared fixed camera derived from combined GT/recon bounds",
            "outputs_typed": {
                "original_mesh": str(original_video_typed.resolve()),
                "reconstruction_mesh": str(recon_video_typed.resolve()),
                "side_by_side": str(side_video_typed.resolve()),
                "rig_overlay": str(rig_overlay_typed.resolve()) if rig_overlay_typed.exists() else None,
                "rig_turntable": str(rig_turntable_typed.resolve()) if rig_turntable_typed.exists() else None,
                "loss_curve": str(loss_curve_typed.resolve()) if loss_curve_written else None,
                "frame_000_compare": str((out_dir / _task_file_name(task_type, "frame_000_compare.png")).resolve()),
                "frame_mid_compare": str((out_dir / _task_file_name(task_type, "frame_mid_compare.png")).resolve()),
                "frame_last_compare": str((out_dir / _task_file_name(task_type, "frame_last_compare.png")).resolve()),
            },
            "outputs_legacy_aliases": {
                "original_mesh": str((out_dir / "original_mesh.mp4").resolve()),
                "reconstruction_mesh": str((out_dir / "reconstruction_mesh.mp4").resolve()),
                "side_by_side": str((out_dir / "original_vs_reconstruction.mp4").resolve()),
                "rig_overlay": str((out_dir / "rig_overlay.png").resolve()) if (out_dir / "rig_overlay.png").exists() else None,
                "rig_turntable": str((out_dir / "rig_turntable.mp4").resolve()) if (out_dir / "rig_turntable.mp4").exists() else None,
                "loss_curve": str((out_dir / "loss_curve.png").resolve()) if (out_dir / "loss_curve.png").exists() else None,
            },
            "viz_centering": {
                "enabled": True,
                "rule": "subtract first-frame centroid per sequence for view framing",
                "original_center0": original_center0.tolist(),
                "recon_center0": recon_center0.tolist(),
            },
        },
    )


def _render_sample(
    *,
    ds: DT4D,
    sample_id: str,
    out_dir: Path,
    per_task_dir: Path,
    fps: int,
    max_frames: int | None,
    width: int,
    height: int,
    turntable: bool,
) -> None:
    sid = ds.resolve_sample_id(sample_id)
    safe_id = safe_sample_id(sid)
    sample = ds.load_sample(sid)

    seq = _cap_frames(sample.vertices, max_frames)
    seq_viz, seq_center0 = _center_for_viz(seq)
    _bmin, _bmax, center, scale = compute_bounds([seq_viz])
    camera_pose = default_camera_pose(center=center, scale=scale)

    print(f"[viz] Rendering sample {sid} frames={len(seq)}")
    orig_frames = render_mesh_frames(
        seq_viz,
        sample.faces,
        width=width,
        height=height,
        camera_pose=camera_pose,
        mesh_color=(110, 150, 230),
    )
    save_video(out_dir / "original_mesh.mp4", orig_frames, fps=fps)

    rig_json = ROOT / "outputs" / "rigs" / safe_id / "rig.json"
    weights_npy = ROOT / "outputs" / "rigs" / safe_id / "weights.npy"
    if rig_json.exists() and weights_npy.exists():
        rig = load_rig(rig_json, weights_npy)
        render_rig_overlay(
            canonical_vertices=sample.canonical_vertices,
            canonical_faces=sample.faces,
            joints=rig.joint_positions,
            parents=rig.parents,
            out_image=out_dir / "rig_overlay.png",
            width=width,
            height=height,
            turntable_out=(out_dir / "rig_turntable.mp4") if turntable else None,
            fps=fps,
            turntable_frames=72,
        )
    else:
        print(f"[viz] No rig found for {safe_id}; rig overlay skipped")

    self_task = f"{safe_id}__to__{safe_id}"
    recon_path = per_task_dir / self_task / "recon_vertices.npy"
    if recon_path.exists():
        recon = np.asarray(np.load(recon_path), dtype=np.float32)
        t = min(len(seq), len(recon))
        seq_t = seq[:t]
        recon_t = recon[:t]
        seq_t_viz, _seq_t_center = _center_for_viz(seq_t)
        recon_t_viz, _recon_t_center = _center_for_viz(recon_t)

        _bmin2, _bmax2, center2, scale2 = compute_bounds([seq_t_viz, recon_t_viz])
        camera_pose2 = default_camera_pose(center=center2, scale=scale2)

        recon_frames = render_mesh_frames(
            recon_t_viz,
            sample.faces,
            width=width,
            height=height,
            camera_pose=camera_pose2,
            mesh_color=(240, 155, 85),
        )
        save_video(out_dir / "reconstruction_mesh.mp4", recon_frames, fps=fps)

        side = make_side_by_side(orig_frames[:t], recon_frames)
        save_video(out_dir / "original_vs_reconstruction.mp4", side, fps=fps)
    else:
        print(f"[viz] No self-task reconstruction found at {recon_path}; skipping recon videos")

    _save_task_meta(
        out_dir / "viz_meta.json",
        {
            "mode": "sample",
            "sample_id": sid,
            "faces_from": f"h5:{sample.sample_id}",
            "rig_json": str(rig_json.resolve()) if rig_json.exists() else None,
            "weights_npy": str(weights_npy.resolve()) if weights_npy.exists() else None,
            "self_recon": str(recon_path.resolve()) if recon_path.exists() else None,
            "frames_used": int(len(seq)),
            "fps": int(fps),
            "viz_centering": {
                "enabled": True,
                "rule": "subtract first-frame centroid for view framing",
                "center0": seq_center0.tolist(),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render minimal presentation visualizations for DT4D repro tasks")
    parser.add_argument("--h5", type=Path, default=ROOT.parent / "dt4d.hdf5")
    parser.add_argument("--results-json", type=Path, default=ROOT / "outputs" / "eval" / "results.json")
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "visualizations")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-turntable", action="store_true", default=False)
    args = parser.parse_args()

    if bool(args.task_id) == bool(args.sample_id):
        raise ValueError("Provide exactly one of --task-id or --sample-id")

    ds = DT4D(args.h5)
    per_task_dir = ROOT / "outputs" / "eval" / "per_task"

    if args.task_id:
        row = _resolve_task_row(args.task_id, args.results_json)
        out_dir = args.output_dir / args.task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _render_task(
            ds=ds,
            task_id=args.task_id,
            row=row,
            out_dir=out_dir,
            per_task_dir=per_task_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            width=args.width,
            height=args.height,
            turntable=not args.no_turntable,
        )
    else:
        sid = ds.resolve_sample_id(args.sample_id)
        safe_id = safe_sample_id(sid)
        out_dir = args.output_dir / safe_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _render_sample(
            ds=ds,
            sample_id=sid,
            out_dir=out_dir,
            per_task_dir=per_task_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            width=args.width,
            height=args.height,
            turntable=not args.no_turntable,
        )

    print(f"[viz] Done. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
