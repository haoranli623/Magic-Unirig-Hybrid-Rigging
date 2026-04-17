from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class WeightTransferConfig:
    # Supported:
    # - unirig_learned_skin: learned UniRig skin prediction + UniRig-style reskin transfer
    # - unirig_reskin: heuristic seed + UniRig-style reskin transfer (legacy)
    # - joint_distance_legacy: older joint-distance heuristic (legacy)
    method: str = "unirig_learned_skin"
    seed_top_k: int = 4
    sigma_mode: str = "median"
    sample_method: str = "median"
    nearest_samples: int = 7
    iter_steps: int = 1
    threshold: float = 0.03
    alpha: float = 2.0


def _segment_distance_sq(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        d = points - a[None, :]
        return np.sum(d * d, axis=1)

    ap = points - a[None, :]
    t = np.sum(ap * ab[None, :], axis=1) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    d = points - proj
    return np.sum(d * d, axis=1)


def _sigma_from_d2(d2: np.ndarray, sigma_mode: str = "median") -> float:
    nearest = np.sqrt(np.clip(np.min(d2, axis=1), 0.0, None))
    if sigma_mode == "mean":
        base = float(np.mean(nearest))
    elif sigma_mode == "p75":
        base = float(np.percentile(nearest, 75.0))
    else:
        base = float(np.median(nearest))
    return max(base * 2.0, 1e-4)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.sum(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return (x / denom).astype(np.float32)


def seed_skin_bone_distance(
    vertices: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    top_k: int = 4,
    sigma_mode: str = "median",
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices (N,3), got {vertices.shape}")
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"Expected joints (K,3), got {joints.shape}")
    if parents.ndim != 1 or parents.shape[0] != joints.shape[0]:
        raise ValueError(f"Expected parents (K,), got {parents.shape}")

    n_verts = vertices.shape[0]
    n_joints = joints.shape[0]

    d2 = np.zeros((n_verts, n_joints), dtype=np.float32)
    for j in range(n_joints):
        p = int(parents[j])
        if p < 0:
            diff = vertices - joints[j][None, :]
            d2[:, j] = np.sum(diff * diff, axis=1)
        else:
            d2[:, j] = _segment_distance_sq(vertices, joints[p], joints[j]).astype(np.float32)

    sigma = _sigma_from_d2(d2, sigma_mode=sigma_mode)
    logits = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)

    if top_k is not None and top_k > 0 and top_k < n_joints:
        idx = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        sparse = np.zeros_like(logits)
        row = np.arange(n_verts)[:, None]
        sparse[row, idx] = logits[row, idx]
        logits = sparse

    weights = _normalize_rows(logits)

    stats = {
        "seed_method": "bone_distance_gaussian",
        "sigma_mode": sigma_mode,
        "sigma": float(sigma),
        "top_k": int(top_k),
        "n_vertices": int(n_verts),
        "n_joints": int(n_joints),
        "row_sum_min": float(weights.sum(axis=1).min()),
        "row_sum_max": float(weights.sum(axis=1).max()),
        "nnz_per_vertex_mean": float((weights > 0).sum(axis=1).mean()),
    }
    return weights, stats


def seed_skin_joint_distance_legacy(
    vertices: np.ndarray,
    joints: np.ndarray,
    top_k: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)

    d2 = np.sum((vertices[:, None, :] - joints[None, :, :]) ** 2, axis=-1)
    sigma = _sigma_from_d2(d2, sigma_mode="median")
    logits = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)

    n_verts, n_joints = logits.shape
    if top_k is not None and top_k > 0 and top_k < n_joints:
        idx = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
        sparse = np.zeros_like(logits)
        row = np.arange(n_verts)[:, None]
        sparse[row, idx] = logits[row, idx]
        logits = sparse

    weights = _normalize_rows(logits)
    stats = {
        "seed_method": "joint_distance_gaussian_legacy",
        "sigma": float(sigma),
        "top_k": int(top_k),
    }
    return weights, stats


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"Similarity solve needs equal shape, got {src.shape} vs {dst.shape}")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src[None, :]
    dst_c = dst - mu_dst[None, :]

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var_src = np.sum(src_c * src_c) / src.shape[0]
    s = float(np.sum(S) / max(var_src, 1e-12))
    t = mu_dst - s * (R @ mu_src)
    return s, R.astype(np.float64), t.astype(np.float64)


def _apply_similarity(x: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (R @ x.T)).T + t[None, :]


def _invert_similarity(s: float, R: np.ndarray, t: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    s_inv = 1.0 / max(s, 1e-12)
    R_inv = R.T
    t_inv = -s_inv * (R_inv @ t)
    return float(s_inv), R_inv, t_inv


def align_sampled_to_target(
    sampled_vertices: np.ndarray,
    target_vertices: np.ndarray,
    joints_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sampled_vertices = np.asarray(sampled_vertices, dtype=np.float32)
    target_vertices = np.asarray(target_vertices, dtype=np.float32)
    joints_target = np.asarray(joints_target, dtype=np.float32)

    report: dict[str, Any] = {
        "alignment": "identity",
        "matched_by_index": False,
    }

    if sampled_vertices.shape == target_vertices.shape:
        # target -> sampled (for seeding in sampled frame)
        s_t2s, R_t2s, t_t2s = _umeyama_similarity(target_vertices, sampled_vertices)
        joints_sampled = _apply_similarity(joints_target, s_t2s, R_t2s, t_t2s).astype(np.float32)

        # sampled -> target (for kNN transfer in target frame)
        s_s2t, R_s2t, t_s2t = _invert_similarity(s_t2s, R_t2s, t_t2s)
        sampled_aligned = _apply_similarity(sampled_vertices, s_s2t, R_s2t, t_s2t).astype(np.float32)

        err = np.linalg.norm(sampled_aligned - target_vertices, axis=1)
        report = {
            "alignment": "similarity_index_correspondence",
            "matched_by_index": True,
            "s_target_to_sampled": float(s_t2s),
            "s_sampled_to_target": float(s_s2t),
            "rmse_after_alignment": float(np.sqrt(np.mean(err * err))),
            "max_error_after_alignment": float(np.max(err)),
        }
        return sampled_aligned, joints_sampled, report

    report = {
        "alignment": "identity_fallback",
        "matched_by_index": False,
        "notes": "Sampled mesh topology differs from target; using raw sampled coordinates.",
    }
    return sampled_vertices, joints_target, report


def reskin_unirig_style(
    sampled_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    sampled_skin: np.ndarray,
    parents: np.ndarray,
    sample_method: str = "median",
    nearest_samples: int = 7,
    iter_steps: int = 1,
    threshold: float = 0.03,
    alpha: float = 2.0,
) -> np.ndarray:
    sampled_vertices = np.asarray(sampled_vertices, dtype=np.float32)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    sampled_skin = np.asarray(sampled_skin, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)

    if sample_method not in {"mean", "median"}:
        raise ValueError(f"sample_method must be mean or median, got {sample_method}")

    n = vertices.shape[0]
    j = sampled_skin.shape[1]
    k = int(max(1, min(nearest_samples, sampled_vertices.shape[0])))

    tree = cKDTree(sampled_vertices)
    dis, nearest = tree.query(vertices, k=k, p=2)
    if k == 1:
        dis = dis[:, None]
        nearest = nearest[:, None]

    if sample_method == "mean":
        weights = np.exp(-alpha * dis).astype(np.float32)
        weight_sum = np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
        sampled_skin_nearest = sampled_skin[nearest]
        skin = (sampled_skin_nearest * weights[..., None]).sum(axis=1) / weight_sum
    else:
        skin = np.median(sampled_skin[nearest], axis=1)

    skin = np.clip(skin, 0.0, None)
    skin = _normalize_rows(skin)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)

    for _ in range(max(0, int(iter_steps))):
        sum_skin = skin.copy()
        for i in reversed(range(j)):
            p = int(parents[i])
            if p < 0:
                continue
            sum_skin[:, p] += sum_skin[:, i]

        mask = sum_skin[edges[:, 1]] < sum_skin[edges[:, 0]]
        neighbor_skin = np.zeros_like(sum_skin)
        neighbor_co = np.zeros((n, j), dtype=np.float32)

        edge_dis = np.sqrt(np.sum((vertices[edges[:, 1]] - vertices[edges[:, 0]]) ** 2, axis=1, keepdims=True))
        co = np.exp(-edge_dis * alpha).astype(np.float32)

        neighbor_skin[edges[:, 1]] += sum_skin[edges[:, 0]] * co * mask
        neighbor_co[edges[:, 1]] += co * mask

        sum_skin = (sum_skin + neighbor_skin) / (1.0 + neighbor_co)
        for i in range(j):
            p = int(parents[i])
            if p < 0:
                continue
            sum_skin[:, p] -= sum_skin[:, i]

        skin = np.clip(sum_skin, 0.0, None)
        skin = _normalize_rows(skin)

    mask = (skin >= threshold).any(axis=1, keepdims=True)
    skin[(skin < threshold) & mask] = 0.0
    skin = _normalize_rows(skin)
    return skin.astype(np.float32)


def transfer_weights_hybrid(
    *,
    canonical_vertices: np.ndarray,
    canonical_faces: np.ndarray,
    sampled_vertices: np.ndarray,
    joints_target: np.ndarray,
    parents: np.ndarray,
    config: WeightTransferConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    c_v = np.asarray(canonical_vertices, dtype=np.float32)
    c_f = np.asarray(canonical_faces, dtype=np.int64)
    s_v = np.asarray(sampled_vertices, dtype=np.float32)
    joints_target = np.asarray(joints_target, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)

    sampled_aligned, joints_sampled, align_report = align_sampled_to_target(
        sampled_vertices=s_v,
        target_vertices=c_v,
        joints_target=joints_target,
    )

    if config.method == "joint_distance_legacy":
        sampled_skin, seed_stats = seed_skin_joint_distance_legacy(
            vertices=s_v,
            joints=joints_sampled,
            top_k=config.seed_top_k,
        )
        weights = reskin_unirig_style(
            sampled_vertices=sampled_aligned,
            vertices=c_v,
            faces=c_f,
            sampled_skin=sampled_skin,
            parents=parents,
            sample_method=config.sample_method,
            nearest_samples=config.nearest_samples,
            iter_steps=config.iter_steps,
            threshold=config.threshold,
            alpha=config.alpha,
        )
    elif config.method == "unirig_reskin":
        sampled_skin, seed_stats = seed_skin_bone_distance(
            vertices=s_v,
            joints=joints_sampled,
            parents=parents,
            top_k=config.seed_top_k,
            sigma_mode=config.sigma_mode,
        )
        weights = reskin_unirig_style(
            sampled_vertices=sampled_aligned,
            vertices=c_v,
            faces=c_f,
            sampled_skin=sampled_skin,
            parents=parents,
            sample_method=config.sample_method,
            nearest_samples=config.nearest_samples,
            iter_steps=config.iter_steps,
            threshold=config.threshold,
            alpha=config.alpha,
        )
    else:
        raise ValueError(f"Unknown weight transfer method: {config.method}")

    nnz = (weights > 0).sum(axis=1)
    report = {
        "weight_method": config.method,
        "config": asdict(config),
        "alignment": align_report,
        "seed_stats": seed_stats,
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
