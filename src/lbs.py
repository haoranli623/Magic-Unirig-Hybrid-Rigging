from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _make_transform(rotvec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    t[:3, 3] = trans
    return t


def compute_skinning_matrices(
    joint_positions: np.ndarray,
    parents: np.ndarray,
    root_rotvec: np.ndarray,
    root_trans: np.ndarray,
    local_rotvec: np.ndarray,
    local_trans: np.ndarray,
) -> np.ndarray:
    k = joint_positions.shape[0]
    mats = np.zeros((k, 4, 4), dtype=np.float64)

    root_global = _make_transform(root_rotvec, root_trans)

    done = np.zeros((k,), dtype=bool)

    def compute_one(i: int) -> np.ndarray:
        if done[i]:
            return mats[i]
        p = int(parents[i])
        if p < 0:
            offset = joint_positions[i]
            local = _make_transform(local_rotvec[i], offset + local_trans[i])
            mats[i] = root_global @ local
        else:
            parent_mat = compute_one(p)
            offset = joint_positions[i] - joint_positions[p]
            local = _make_transform(local_rotvec[i], offset + local_trans[i])
            mats[i] = parent_mat @ local
        done[i] = True
        return mats[i]

    for i in range(k):
        compute_one(i)

    inv_bind = np.zeros((k, 4, 4), dtype=np.float64)
    for i in range(k):
        bind = np.eye(4, dtype=np.float64)
        bind[:3, 3] = joint_positions[i]
        inv_bind[i] = np.linalg.inv(bind)

    return mats @ inv_bind


def lbs_deform(
    rest_vertices: np.ndarray,
    weights: np.ndarray,
    skin_mats: np.ndarray,
) -> np.ndarray:
    n = rest_vertices.shape[0]
    v_h = np.concatenate([rest_vertices, np.ones((n, 1), dtype=np.float64)], axis=1)
    transformed = np.einsum("kij,nj->kni", skin_mats, v_h)[..., :3]
    deformed = np.einsum("nk,kni->ni", weights, transformed)
    return deformed.astype(np.float32)


def deform_with_params(
    rest_vertices: np.ndarray,
    joint_positions: np.ndarray,
    parents: np.ndarray,
    weights: np.ndarray,
    root_rotvec: np.ndarray,
    root_trans: np.ndarray,
    local_rotvec: np.ndarray,
    local_trans: np.ndarray,
) -> np.ndarray:
    skin = compute_skinning_matrices(
        joint_positions=joint_positions,
        parents=parents,
        root_rotvec=root_rotvec,
        root_trans=root_trans,
        local_rotvec=local_rotvec,
        local_trans=local_trans,
    )
    return lbs_deform(rest_vertices=rest_vertices, weights=weights, skin_mats=skin)
