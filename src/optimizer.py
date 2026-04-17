from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from src.lbs import deform_with_params
from src.rig_format import RigData


@dataclass
class OptimizeConfig:
    max_nfev: int = 40
    temporal_smooth: float = 1e-2
    local_rot_reg: float = 5e-4
    local_trans_reg: float = 5e-3
    root_rot_reg: float = 1e-4
    vertex_sample_count: int = 1500
    random_seed: int = 0


def _pack(
    root_rot: np.ndarray,
    root_trans: np.ndarray,
    local_rot: np.ndarray,
    local_trans: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            root_rot.reshape(-1),
            root_trans.reshape(-1),
            local_rot.reshape(-1),
            local_trans.reshape(-1),
        ],
        axis=0,
    )


def _unpack(x: np.ndarray, n_joints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = 0
    root_rot = x[i : i + 3]
    i += 3
    root_trans = x[i : i + 3]
    i += 3
    local_rot = x[i : i + 3 * n_joints].reshape(n_joints, 3)
    i += 3 * n_joints
    local_trans = x[i : i + 3 * n_joints].reshape(n_joints, 3)
    return root_rot, root_trans, local_rot, local_trans


def _initial_params(rig: RigData, target_vertices: np.ndarray) -> np.ndarray:
    root_rot = np.zeros(3, dtype=np.float64)
    root_trans = np.mean(target_vertices - rig.joint_positions[rig.root_index][None, :], axis=0)
    local_rot = np.zeros((rig.num_joints, 3), dtype=np.float64)
    local_trans = np.zeros((rig.num_joints, 3), dtype=np.float64)
    return _pack(root_rot, root_trans, local_rot, local_trans)


def optimize_sequence(
    *,
    rig: RigData,
    canonical_vertices: np.ndarray,
    target_sequence: np.ndarray,
    config: OptimizeConfig,
    show_progress: bool = True,
) -> dict[str, Any]:
    if target_sequence.ndim != 3 or target_sequence.shape[-1] != 3:
        raise ValueError(f"Expected target sequence (T,N,3), got {target_sequence.shape}")
    if canonical_vertices.shape[0] != target_sequence.shape[1]:
        raise ValueError(
            "Vertex count mismatch between canonical and target sequence: "
            f"{canonical_vertices.shape[0]} vs {target_sequence.shape[1]}"
        )
    if rig.weights.shape[0] != canonical_vertices.shape[0] or rig.weights.shape[1] != rig.num_joints:
        raise ValueError(
            f"Weight shape {rig.weights.shape} incompatible with canonical {canonical_vertices.shape} and joints {rig.num_joints}"
        )

    t_frames = target_sequence.shape[0]
    n_verts = canonical_vertices.shape[0]

    rng = np.random.default_rng(config.random_seed)
    sample_count = min(config.vertex_sample_count, n_verts)
    fit_idx = np.sort(rng.choice(n_verts, size=sample_count, replace=False))

    rest_fit = canonical_vertices[fit_idx]
    w_fit = rig.weights[fit_idx]

    params = np.zeros((t_frames, 6 + 6 * rig.num_joints), dtype=np.float32)
    recon = np.zeros_like(target_sequence, dtype=np.float32)
    logs: list[dict[str, Any]] = []

    prev_x: np.ndarray | None = None
    iterator = range(t_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Optimizing sequence", leave=False)

    for t in iterator:
        tgt_full = target_sequence[t]
        tgt_fit = tgt_full[fit_idx]

        x0 = _initial_params(rig, tgt_fit) if prev_x is None else prev_x.copy()

        def residual(x: np.ndarray) -> np.ndarray:
            root_rot, root_trans, local_rot, local_trans = _unpack(x, rig.num_joints)
            pred_fit = deform_with_params(
                rest_vertices=rest_fit,
                joint_positions=rig.joint_positions,
                parents=rig.parents,
                weights=w_fit,
                root_rotvec=root_rot,
                root_trans=root_trans,
                local_rotvec=local_rot,
                local_trans=local_trans,
            )
            r = [
                (pred_fit - tgt_fit).reshape(-1),
                np.sqrt(config.local_rot_reg) * local_rot.reshape(-1),
                np.sqrt(config.local_trans_reg) * local_trans.reshape(-1),
                np.sqrt(config.root_rot_reg) * root_rot.reshape(-1),
            ]
            if prev_x is not None and config.temporal_smooth > 0:
                r.append(np.sqrt(config.temporal_smooth) * (x - prev_x))
            return np.concatenate(r, axis=0)

        sol = least_squares(
            residual,
            x0=x0,
            method="trf",
            loss="soft_l1",
            max_nfev=config.max_nfev,
            verbose=0,
        )

        x_opt = sol.x.astype(np.float64)
        prev_x = x_opt
        params[t] = x_opt.astype(np.float32)

        root_rot, root_trans, local_rot, local_trans = _unpack(x_opt, rig.num_joints)
        recon[t] = deform_with_params(
            rest_vertices=canonical_vertices,
            joint_positions=rig.joint_positions,
            parents=rig.parents,
            weights=rig.weights,
            root_rotvec=root_rot,
            root_trans=root_trans,
            local_rotvec=local_rot,
            local_trans=local_trans,
        )

        logs.append(
            {
                "frame": t,
                "success": bool(sol.success),
                "status": int(sol.status),
                "message": str(sol.message),
                "nfev": int(sol.nfev),
                "cost": float(sol.cost),
                "fit_vertex_count": int(sample_count),
            }
        )

    return {
        "motion_params": params,
        "recon_vertices": recon,
        "fit_vertex_indices": fit_idx,
        "opt_log": logs,
    }


def apply_motion_params(
    *,
    rig: RigData,
    canonical_vertices: np.ndarray,
    motion_params: np.ndarray,
) -> np.ndarray:
    t_frames = motion_params.shape[0]
    out = np.zeros((t_frames, canonical_vertices.shape[0], 3), dtype=np.float32)
    for t in range(t_frames):
        root_rot, root_trans, local_rot, local_trans = _unpack(
            motion_params[t].astype(np.float64), rig.num_joints
        )
        out[t] = deform_with_params(
            rest_vertices=canonical_vertices,
            joint_positions=rig.joint_positions,
            parents=rig.parents,
            weights=rig.weights,
            root_rotvec=root_rot,
            root_trans=root_trans,
            local_rotvec=local_rot,
            local_trans=local_trans,
        )
    return out
