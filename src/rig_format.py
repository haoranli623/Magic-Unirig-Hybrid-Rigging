from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RigData:
    joint_names: list[str]
    joint_positions: np.ndarray  # (K, 3)
    parents: np.ndarray  # (K,), -1 for root
    root_index: int
    weights: np.ndarray  # (N, K)
    source: str
    has_native_weights: bool
    assumptions: list[str]

    @property
    def num_joints(self) -> int:
        return int(self.joint_positions.shape[0])

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": "magic_dt4d_rig_v1",
            "joint_names": self.joint_names,
            "joint_positions": self.joint_positions.tolist(),
            "parents": self.parents.tolist(),
            "root_index": int(self.root_index),
            "source": self.source,
            "has_native_weights": bool(self.has_native_weights),
            "assumptions": self.assumptions,
        }


def _name_to_index(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def _infer_parent_array(
    names: list[str], root_name: str | None, hier_edges: list[tuple[str, str]]
) -> tuple[np.ndarray, int]:
    name2idx = _name_to_index(names)
    parents = np.full((len(names),), -1, dtype=np.int32)

    for p_name, c_name in hier_edges:
        if p_name not in name2idx or c_name not in name2idx:
            continue
        c_idx = name2idx[c_name]
        p_idx = name2idx[p_name]
        parents[c_idx] = p_idx

    if root_name and root_name in name2idx:
        root_idx = name2idx[root_name]
    else:
        roots = np.where(parents < 0)[0]
        root_idx = int(roots[0]) if len(roots) else 0

    parents[root_idx] = -1
    return parents, root_idx


def parse_magic_pred_txt(txt_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    names: list[str] = []
    pos: list[list[float]] = []
    root_name: str | None = None
    hier_edges: list[tuple[str, str]] = []

    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0].lower()
        if tag == "joints" and len(parts) >= 5:
            names.append(parts[1])
            pos.append([float(parts[2]), float(parts[3]), float(parts[4])])
        elif tag == "root" and len(parts) >= 2:
            root_name = parts[1]
        elif tag == "hier" and len(parts) >= 3:
            hier_edges.append((parts[1], parts[2]))

    if not names:
        raise ValueError(f"No joints found in MagicArticulate output: {txt_path}")

    joint_pos = np.asarray(pos, dtype=np.float32)
    parents, root_idx = _infer_parent_array(names, root_name, hier_edges)
    return names, joint_pos, parents, root_idx


def estimate_skinning_weights(
    vertices: np.ndarray,
    joint_positions: np.ndarray,
    top_k: int = 4,
) -> tuple[np.ndarray, list[str]]:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shape (N,3), got {vertices.shape}")
    if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
        raise ValueError(f"Expected joint positions shape (K,3), got {joint_positions.shape}")

    n_verts = vertices.shape[0]
    n_joints = joint_positions.shape[0]
    if n_joints <= 0:
        raise ValueError("Cannot estimate weights without joints")

    d2 = np.sum((vertices[:, None, :] - joint_positions[None, :, :]) ** 2, axis=-1)

    nearest = np.sqrt(np.clip(np.min(d2, axis=1), 0.0, None))
    sigma = float(max(np.median(nearest) * 2.0, 1e-4))
    logits = np.exp(-d2 / (2.0 * sigma * sigma))

    if top_k is not None and top_k > 0 and top_k < n_joints:
        idx = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        sparse = np.zeros_like(logits)
        row = np.arange(n_verts)[:, None]
        sparse[row, idx] = logits[row, idx]
        logits = sparse

    denom = np.sum(logits, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    weights = (logits / denom).astype(np.float32)

    notes = [
        "No native skinning weights were output by public MagicArticulate skeleton demo; weights estimated by joint-distance soft assignment.",
        f"Weight model: Gaussian over Euclidean distance to joints with sigma={sigma:.6f}, top_k={top_k}.",
    ]
    return weights, notes


def save_rig(rig: RigData, rig_json_path: Path, weights_npy_path: Path) -> None:
    rig_json_path.parent.mkdir(parents=True, exist_ok=True)
    weights_npy_path.parent.mkdir(parents=True, exist_ok=True)
    rig_json_path.write_text(json.dumps(rig.to_json_dict(), indent=2), encoding="utf-8")
    np.save(weights_npy_path, rig.weights)


def load_rig(rig_json_path: Path, weights_npy_path: Path) -> RigData:
    data = json.loads(rig_json_path.read_text(encoding="utf-8"))
    weights = np.load(weights_npy_path)
    return RigData(
        joint_names=list(data["joint_names"]),
        joint_positions=np.asarray(data["joint_positions"], dtype=np.float32),
        parents=np.asarray(data["parents"], dtype=np.int32),
        root_index=int(data["root_index"]),
        weights=np.asarray(weights, dtype=np.float32),
        source=str(data.get("source", "unknown")),
        has_native_weights=bool(data.get("has_native_weights", False)),
        assumptions=list(data.get("assumptions", [])),
    )
