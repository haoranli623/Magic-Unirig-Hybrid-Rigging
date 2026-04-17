from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from src.mesh_utils import read_obj
from src.weight_transfer import WeightTransferConfig, reskin_unirig_style


@dataclass
class UniRigSkinBridgeConfig:
    enabled: bool = True
    python_exe: str = "python3"
    unirig_repo: str = ""
    task_config: str = ""
    cls_label: str = "dt4d"
    seed: int = 123
    normalize_into_min: float = -1.0
    normalize_into_max: float = 1.0


def _trans_to_m(t: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = t.astype(np.float32)
    return m


def _scale_to_m(s: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = float(s)
    m[1, 1] = float(s)
    m[2, 2] = float(s)
    return m


def _apply_affine(points: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return points @ trans[:3, :3].T + trans[:3, 3]


def _compute_unirig_affine(
    vertices: np.ndarray,
    joints: np.ndarray,
    normalize_into_min: float,
    normalize_into_max: float,
) -> np.ndarray:
    bound_min = vertices.min(axis=0)
    bound_max = vertices.max(axis=0)
    if joints.size > 0:
        bound_min = np.minimum(bound_min, joints.min(axis=0))
        bound_max = np.maximum(bound_max, joints.max(axis=0))

    trans = np.eye(4, dtype=np.float32)
    trans = _trans_to_m(-(bound_max + bound_min) / 2.0) @ trans

    denom = max(normalize_into_max - normalize_into_min, 1e-6)
    scale = float(np.max((bound_max - bound_min) / denom))
    if scale < 1e-8:
        scale = 1.0
    trans = _scale_to_m(1.0 / scale) @ trans

    bias = (normalize_into_min + normalize_into_max) / 2.0
    trans = _trans_to_m(np.array([bias, bias, bias], dtype=np.float32)) @ trans
    return trans


def _parents_to_unirig_int(parents: np.ndarray) -> np.ndarray:
    out = np.asarray(parents, dtype=np.int32).copy()
    if out.size == 0:
        return out
    out[out < 0] = -1
    # Keep root in the first slot for UniRig data assumptions.
    out[0] = -1
    return out


def _build_tails(joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)

    k = joints.shape[0]
    children: list[list[int]] = [[] for _ in range(k)]
    for i in range(k):
        p = int(parents[i])
        if p >= 0:
            children[p].append(i)

    bbox = joints.max(axis=0) - joints.min(axis=0) if k > 0 else np.array([1.0, 1.0, 1.0], dtype=np.float32)
    eps = float(max(np.max(bbox) * 0.03, 1e-3))

    tails = np.zeros_like(joints)
    for i in range(k):
        ch = children[i]
        if ch:
            tails[i] = joints[np.asarray(ch, dtype=np.int32)].mean(axis=0)
            continue

        p = int(parents[i])
        if p >= 0:
            direction = joints[i] - joints[p]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.array([0.0, eps, 0.0], dtype=np.float32)
            else:
                direction = direction / norm * eps
            tails[i] = joints[i] + direction
        else:
            tails[i] = joints[i] + np.array([0.0, eps, 0.0], dtype=np.float32)
    return tails.astype(np.float32)


def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.asarray(mesh.vertex_normals, dtype=np.float32), np.asarray(mesh.face_normals, dtype=np.float32)


def _prepare_unirig_predict_npz(
    *,
    canonical_obj: Path,
    joints: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    out_path: Path,
) -> dict[str, Any]:
    vertices, faces = read_obj(canonical_obj)
    v_norm, f_norm = _compute_normals(vertices, faces)
    tails = _build_tails(joints, parents)

    payload = {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "vertex_normals": v_norm,
        "faces": np.asarray(faces, dtype=np.int64),
        "face_normals": f_norm,
        "joints": np.asarray(joints, dtype=np.float32),
        "skin": None,
        "parents": np.asarray(_parents_to_unirig_int(parents), dtype=np.int64),
        "names": np.asarray(list(names), dtype=object),
        "matrix_local": None,
        "tails": tails,
        "no_skin": None,
        "path": None,
        "cls": None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)

    return {
        "predict_npz": str(out_path.resolve()),
        "num_vertices": int(vertices.shape[0]),
        "num_faces": int(faces.shape[0]),
        "num_joints": int(joints.shape[0]),
    }


def _load_predicted_skin(raw_skin_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(raw_skin_npz, allow_pickle=True)
    if "skin" not in d or "vertices" not in d or "joints" not in d:
        raise ValueError(f"Unexpected UniRig skin output keys in {raw_skin_npz}: {list(d.keys())}")
    skin = np.asarray(d["skin"], dtype=np.float32)
    vertices = np.asarray(d["vertices"], dtype=np.float32)
    joints = np.asarray(d["joints"], dtype=np.float32)
    return skin, vertices, joints


def run_unirig_learned_skin_bridge(
    *,
    project_root: Path,
    sample_id: str,
    canonical_obj: Path,
    joints: np.ndarray,
    parents: np.ndarray,
    names: list[str],
    weight_transfer_cfg: WeightTransferConfig,
    unirig_cfg: UniRigSkinBridgeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not unirig_cfg.enabled:
        raise RuntimeError("UniRig learned skin bridge is disabled")

    unirig_repo = Path(unirig_cfg.unirig_repo).resolve()
    if not unirig_repo.exists():
        raise FileNotFoundError(f"UniRig repo not found: {unirig_repo}")

    task_cfg = Path(unirig_cfg.task_config).resolve()
    if not task_cfg.exists():
        raise FileNotFoundError(f"UniRig task config not found: {task_cfg}")

    bridge_root = project_root / "outputs" / "unirig_skin_bridge" / sample_id.replace("/", "__")
    input_dir = bridge_root / "input_meshes"
    npz_dir = bridge_root / "npz_cache"
    logs_dir = bridge_root / "logs"
    input_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_mesh = input_dir / "canonical.obj"
    shutil.copy2(canonical_obj, input_mesh)

    predict_npz = npz_dir / "canonical" / "predict_skeleton.npz"
    prep_meta = _prepare_unirig_predict_npz(
        canonical_obj=canonical_obj,
        joints=joints,
        parents=parents,
        names=names,
        out_path=predict_npz,
    )

    runner = project_root / "scripts" / "run_unirig_skin_predict_minimal.py"
    if not runner.exists():
        raise FileNotFoundError(f"UniRig bridge runner script missing: {runner}")

    cmd = [
        unirig_cfg.python_exe,
        str(runner.resolve()),
        "--unirig-repo",
        str(unirig_repo),
        "--task-config",
        str(task_cfg),
        "--seed",
        str(unirig_cfg.seed),
        "--sample-dir",
        str(npz_dir / "canonical"),
        "--output-dir",
        str(bridge_root / "results"),
        "--npz-dir",
        str(npz_dir),
        "--data-name",
        "predict_skeleton.npz",
        "--cls",
        str(unirig_cfg.cls_label),
    ]

    env = os.environ.copy()
    shim_path = str((project_root / "shims").resolve())
    prev_py = env.get("PYTHONPATH", "")
    path_parts: list[str] = [shim_path]
    if env.get("MAGIC_DISABLE_PKG_PYTHONPATH", "0") != "1":
        path_parts.append(str((project_root / ".pkg").resolve()))
    if prev_py:
        path_parts.append(prev_py)
    env["PYTHONPATH"] = ":".join(path_parts)
    env["HF_HOME"] = str((project_root / ".hf_cache").resolve())
    env["TRANSFORMERS_CACHE"] = str((project_root / ".hf_cache").resolve())

    proc = subprocess.run(
        cmd,
        cwd=str(unirig_repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    (logs_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (logs_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (logs_dir / "command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "cwd": str(unirig_repo),
                "pythonpath": env["PYTHONPATH"],
                "task_config": str(task_cfg),
                "runner_script": str(runner.resolve()),
                "typing_patch": "enabled_via_run_unirig_skin_predict_minimal",
                "returncode": int(proc.returncode),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_skin_npz = npz_dir / "canonical" / "predict_skin.npz"
    if proc.returncode != 0:
        raise RuntimeError(
            "UniRig skin inference failed; "
            f"returncode={proc.returncode}. See {logs_dir / 'stderr.txt'}"
        )
    if not raw_skin_npz.exists():
        raise FileNotFoundError(
            f"UniRig skin output missing: {raw_skin_npz}. See logs in {logs_dir}"
        )

    sampled_skin, sampled_vertices_norm, sampled_joints_norm = _load_predicted_skin(raw_skin_npz)

    canonical_vertices, canonical_faces = read_obj(canonical_obj)
    affine = _compute_unirig_affine(
        vertices=canonical_vertices,
        joints=np.asarray(joints, dtype=np.float32),
        normalize_into_min=unirig_cfg.normalize_into_min,
        normalize_into_max=unirig_cfg.normalize_into_max,
    )
    affine_inv = np.linalg.inv(affine)

    sampled_vertices = _apply_affine(sampled_vertices_norm, affine_inv).astype(np.float32)
    sampled_joints = _apply_affine(sampled_joints_norm, affine_inv).astype(np.float32)

    weights = reskin_unirig_style(
        sampled_vertices=sampled_vertices,
        vertices=np.asarray(canonical_vertices, dtype=np.float32),
        faces=np.asarray(canonical_faces, dtype=np.int64),
        sampled_skin=sampled_skin,
        parents=np.asarray(parents, dtype=np.int32),
        sample_method=weight_transfer_cfg.sample_method,
        nearest_samples=weight_transfer_cfg.nearest_samples,
        iter_steps=weight_transfer_cfg.iter_steps,
        threshold=weight_transfer_cfg.threshold,
        alpha=weight_transfer_cfg.alpha,
    )

    nnz = (weights > 0).sum(axis=1)
    report = {
        "weight_method": "unirig_learned_skin",
        "bridge_config": asdict(unirig_cfg),
        "reskin_config": {
            "sample_method": weight_transfer_cfg.sample_method,
            "nearest_samples": int(weight_transfer_cfg.nearest_samples),
            "iter_steps": int(weight_transfer_cfg.iter_steps),
            "threshold": float(weight_transfer_cfg.threshold),
            "alpha": float(weight_transfer_cfg.alpha),
        },
        "artifacts": {
            "bridge_root": str(bridge_root.resolve()),
            "predict_skeleton_npz": str(predict_npz.resolve()),
            "raw_predicted_skin_npz": str(raw_skin_npz.resolve()),
            "logs_dir": str(logs_dir.resolve()),
        },
        "prep_meta": prep_meta,
        "predicted_sampled_stats": {
            "sampled_vertices_shape": [int(x) for x in sampled_vertices.shape],
            "sampled_skin_shape": [int(x) for x in sampled_skin.shape],
            "sampled_joints_shape": [int(x) for x in sampled_joints.shape],
        },
        "result_stats": {
            "shape": [int(weights.shape[0]), int(weights.shape[1])],
            "row_sum_min": float(weights.sum(axis=1).min()),
            "row_sum_max": float(weights.sum(axis=1).max()),
            "nnz_per_vertex_min": int(nnz.min()),
            "nnz_per_vertex_max": int(nnz.max()),
            "nnz_per_vertex_mean": float(nnz.mean()),
            "weight_min_nonzero": float(weights[weights > 0].min()) if np.any(weights > 0) else 0.0,
            "weight_max": float(weights.max()),
        },
    }
    return weights.astype(np.float32), report
