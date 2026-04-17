from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.rig_format import RigData


@dataclass
class AdamFitConfig:
    n_iters: int = 500
    lr: float = 1e-2
    trace_every: int = 25
    vertex_sample_count: int = 1500
    random_seed: int = 0
    device: str = "auto"
    rot_l2: float = 0.0
    root_l2: float = 0.0
    temporal_smooth: float = 0.0


def _choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / angle

    cos_a = torch.cos(angle).unsqueeze(-1)
    sin_a = torch.sin(angle).unsqueeze(-1)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)

    K = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    R = cos_a * I + sin_a * K + (1 - cos_a) * torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(-2))
    return R


def _topological_order(parents: np.ndarray) -> list[int]:
    parents = np.asarray(parents, dtype=np.int32)
    k = int(parents.shape[0])
    children: list[list[int]] = [[] for _ in range(k)]
    roots: list[int] = []
    for i in range(k):
        p = int(parents[i])
        if p < 0:
            roots.append(i)
        else:
            children[p].append(i)

    order: list[int] = []
    stack = list(reversed(roots))
    while stack:
        u = stack.pop()
        order.append(u)
        ch = children[u]
        for v in reversed(ch):
            stack.append(v)

    if len(order) < k:
        seen = set(order)
        for i in range(k):
            if i not in seen:
                order.append(i)
    return order


def forward_kinematics(
    joints: torch.Tensor,
    parents: np.ndarray,
    local_R: torch.Tensor,
    root_trans: torch.Tensor,
    topo_order: list[int],
) -> torch.Tensor:
    k = joints.shape[0]
    device = joints.device

    global_R: list[torch.Tensor | None] = [None] * k
    global_pos: list[torch.Tensor | None] = [None] * k

    for i in topo_order:
        p = int(parents[i])
        if p < 0:
            global_R[i] = local_R[i]
            global_pos[i] = joints[i] + root_trans
        else:
            assert global_R[p] is not None and global_pos[p] is not None
            global_R[i] = global_R[p] @ local_R[i]
            global_pos[i] = global_pos[p] + global_R[p] @ (joints[i] - joints[p])

    mats = torch.zeros((k, 4, 4), device=device, dtype=torch.float32)
    inv_bind = torch.zeros((k, 4, 4), device=device, dtype=torch.float32)

    eye4 = torch.eye(4, device=device, dtype=torch.float32)
    for i in range(k):
        assert global_R[i] is not None and global_pos[i] is not None
        mats[i] = eye4
        mats[i, :3, :3] = global_R[i]
        mats[i, :3, 3] = global_pos[i]

        inv_bind[i] = eye4
        inv_bind[i, :3, 3] = -joints[i]

    return mats @ inv_bind


def lbs(rest_vertices: torch.Tensor, weights: torch.Tensor, skin_mats: torch.Tensor) -> torch.Tensor:
    n = rest_vertices.shape[0]
    v_h = torch.cat([rest_vertices, torch.ones((n, 1), device=rest_vertices.device, dtype=torch.float32)], dim=1)
    transformed = torch.einsum("kij,nj->kni", skin_mats, v_h)[..., :3]
    deformed = torch.einsum("nk,kni->ni", weights, transformed)
    return deformed


def _deform_with_pose(
    rest_vertices: torch.Tensor,
    weights: torch.Tensor,
    joints: torch.Tensor,
    parents: np.ndarray,
    axis_angle: torch.Tensor,
    root_trans: torch.Tensor,
    topo_order: list[int],
) -> torch.Tensor:
    local_R = axis_angle_to_matrix(axis_angle)
    skin_mats = forward_kinematics(
        joints=joints,
        parents=parents,
        local_R=local_R,
        root_trans=root_trans,
        topo_order=topo_order,
    )
    return lbs(rest_vertices=rest_vertices, weights=weights, skin_mats=skin_mats)


def fit_sequence_adam(
    *,
    rig: RigData,
    canonical_vertices: np.ndarray,
    target_sequence: np.ndarray,
    config: AdamFitConfig,
    show_progress: bool = True,
) -> dict[str, Any]:
    if target_sequence.ndim != 3 or target_sequence.shape[-1] != 3:
        raise ValueError(f"Expected target sequence (T,N,3), got {target_sequence.shape}")

    canonical_vertices = np.asarray(canonical_vertices, dtype=np.float32)
    target_sequence = np.asarray(target_sequence, dtype=np.float32)

    if canonical_vertices.shape[0] != target_sequence.shape[1]:
        raise ValueError(
            f"Vertex count mismatch: canonical {canonical_vertices.shape[0]} vs target {target_sequence.shape[1]}"
        )

    if rig.weights.shape[0] != canonical_vertices.shape[0] or rig.weights.shape[1] != rig.num_joints:
        raise ValueError(
            f"Weight shape {rig.weights.shape} incompatible with canonical {canonical_vertices.shape} and joints {rig.num_joints}"
        )

    device = _choose_device(config.device)
    rng = np.random.default_rng(config.random_seed)

    n = canonical_vertices.shape[0]
    sample_count = min(config.vertex_sample_count, n) if config.vertex_sample_count > 0 else n
    fit_idx = np.sort(rng.choice(n, size=sample_count, replace=False))

    rest_full = torch.tensor(canonical_vertices, dtype=torch.float32, device=device)
    rest_fit = rest_full[fit_idx]

    target_full = torch.tensor(target_sequence, dtype=torch.float32, device=device)
    target_fit = target_full[:, fit_idx]

    weights_full = torch.tensor(rig.weights, dtype=torch.float32, device=device)
    weights_fit = weights_full[fit_idx]

    joints = torch.tensor(rig.joint_positions, dtype=torch.float32, device=device)
    parents = np.asarray(rig.parents, dtype=np.int32)
    topo_order = _topological_order(parents)

    t_frames = target_sequence.shape[0]
    j = rig.num_joints

    axis_all = np.zeros((t_frames, j, 3), dtype=np.float32)
    root_all = np.zeros((t_frames, 3), dtype=np.float32)
    recon = np.zeros_like(target_sequence, dtype=np.float32)

    loss_trace: list[dict[str, Any]] = []
    frame_summary: list[dict[str, Any]] = []

    prev_axis: torch.Tensor | None = None
    prev_root: torch.Tensor | None = None

    iterator = range(t_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Adam fitting", leave=False)

    for t in iterator:
        if prev_axis is None:
            axis = torch.zeros((j, 3), dtype=torch.float32, device=device, requires_grad=True)
            root = torch.zeros((3,), dtype=torch.float32, device=device, requires_grad=True)
            warm_start = False
        else:
            axis = prev_axis.detach().clone().requires_grad_(True)
            root = prev_root.detach().clone().requires_grad_(True)
            warm_start = True

        optimizer = torch.optim.Adam([axis, root], lr=config.lr)

        best_loss = float("inf")
        best_data_loss = float("inf")
        best_iter = -1
        best_axis: torch.Tensor | None = None
        best_root: torch.Tensor | None = None

        frame_points: list[dict[str, Any]] = []
        final_total = None
        final_data = None

        for it in range(int(config.n_iters)):
            optimizer.zero_grad()
            pred_fit = _deform_with_pose(
                rest_vertices=rest_fit,
                weights=weights_fit,
                joints=joints,
                parents=parents,
                axis_angle=axis,
                root_trans=root,
                topo_order=topo_order,
            )

            data_loss = torch.mean((pred_fit - target_fit[t]) ** 2)
            total = data_loss

            if config.rot_l2 > 0:
                total = total + config.rot_l2 * torch.mean(axis * axis)
            if config.root_l2 > 0:
                total = total + config.root_l2 * torch.mean(root * root)
            if config.temporal_smooth > 0 and prev_axis is not None and prev_root is not None:
                total = total + config.temporal_smooth * (
                    torch.mean((axis - prev_axis) ** 2) + torch.mean((root - prev_root) ** 2)
                )

            total.backward()
            optimizer.step()

            total_val = float(total.detach().item())
            data_val = float(data_loss.detach().item())
            final_total = total_val
            final_data = data_val

            should_trace = (it % max(1, int(config.trace_every)) == 0) or (it == int(config.n_iters) - 1)
            if should_trace:
                frame_points.append(
                    {
                        "iter": int(it),
                        "loss_total": total_val,
                        "loss_data": data_val,
                    }
                )

            if total_val < best_loss:
                best_loss = total_val
                best_data_loss = data_val
                best_iter = int(it)
                best_axis = axis.detach().clone()
                best_root = root.detach().clone()

        assert best_axis is not None and best_root is not None

        pred_full = _deform_with_pose(
            rest_vertices=rest_full,
            weights=weights_full,
            joints=joints,
            parents=parents,
            axis_angle=best_axis,
            root_trans=best_root,
            topo_order=topo_order,
        )

        recon[t] = pred_full.detach().cpu().numpy().astype(np.float32)
        axis_all[t] = best_axis.detach().cpu().numpy().astype(np.float32)
        root_all[t] = best_root.detach().cpu().numpy().astype(np.float32)

        prev_axis = best_axis
        prev_root = best_root

        loss_trace.append(
            {
                "frame": int(t),
                "points": frame_points,
                "best_iter": int(best_iter),
                "best_loss": float(best_loss),
                "best_data_loss": float(best_data_loss),
            }
        )

        frame_summary.append(
            {
                "frame": int(t),
                "warm_start": bool(warm_start),
                "n_iters": int(config.n_iters),
                "best_iter": int(best_iter),
                "best_loss": float(best_loss),
                "best_data_loss": float(best_data_loss),
                "final_loss": float(final_total if final_total is not None else best_loss),
                "final_data_loss": float(final_data if final_data is not None else best_data_loss),
                "fit_vertex_count": int(sample_count),
            }
        )

    return {
        "axis_angle": axis_all,
        "root_trans": root_all,
        "recon_vertices": recon,
        "fit_vertex_indices": fit_idx,
        "loss_trace": loss_trace,
        "fit_frame_summary": frame_summary,
        "fit_config": asdict(config),
    }


def apply_motion_params_adam(
    *,
    rig: RigData,
    canonical_vertices: np.ndarray,
    axis_angle: np.ndarray,
    root_trans: np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    canonical_vertices = np.asarray(canonical_vertices, dtype=np.float32)
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    root_trans = np.asarray(root_trans, dtype=np.float32)

    if axis_angle.ndim != 3 or axis_angle.shape[2] != 3:
        raise ValueError(f"Expected axis_angle (T,J,3), got {axis_angle.shape}")
    if root_trans.ndim != 2 or root_trans.shape[1] != 3:
        raise ValueError(f"Expected root_trans (T,3), got {root_trans.shape}")
    if axis_angle.shape[0] != root_trans.shape[0]:
        raise ValueError("axis_angle/root_trans frame mismatch")

    dev = _choose_device(device)
    rest = torch.tensor(canonical_vertices, dtype=torch.float32, device=dev)
    weights = torch.tensor(rig.weights, dtype=torch.float32, device=dev)
    joints = torch.tensor(rig.joint_positions, dtype=torch.float32, device=dev)
    parents = np.asarray(rig.parents, dtype=np.int32)
    topo_order = _topological_order(parents)

    t_frames = axis_angle.shape[0]
    out = np.zeros((t_frames, canonical_vertices.shape[0], 3), dtype=np.float32)
    for t in range(t_frames):
        aa = torch.tensor(axis_angle[t], dtype=torch.float32, device=dev)
        rt = torch.tensor(root_trans[t], dtype=torch.float32, device=dev)
        pred = _deform_with_pose(
            rest_vertices=rest,
            weights=weights,
            joints=joints,
            parents=parents,
            axis_angle=aa,
            root_trans=rt,
            topo_order=topo_order,
        )
        out[t] = pred.detach().cpu().numpy().astype(np.float32)
    return out
