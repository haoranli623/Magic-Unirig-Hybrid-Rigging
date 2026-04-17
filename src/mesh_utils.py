from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    with open(path, "w", encoding="utf-8") as fh:
        for x, y, z in v:
            fh.write(f"v {float(x):.9f} {float(y):.9f} {float(z):.9f}\n")
        for a, b, c in f:
            fh.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed vertex line in {path}: {line}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if not line.startswith("f "):
                continue

            parts = line.split()[1:]
            if len(parts) < 3:
                raise ValueError(f"Malformed face line in {path}: {line}")

            idxs: list[int] = []
            for token in parts:
                base = token.split("/")[0]
                if base == "":
                    raise ValueError(f"Malformed face token in {path}: {token}")
                idx = int(base)
                if idx > 0:
                    idx0 = idx - 1
                else:
                    idx0 = len(vertices) + idx
                idxs.append(idx0)

            # Triangulate n-gons by fan.
            for i in range(1, len(idxs) - 1):
                faces.append([idxs[0], idxs[i], idxs[i + 1]])

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    if not faces:
        raise ValueError(f"OBJ has no faces: {path}")

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if f.min() < 0 or f.max() >= v.shape[0]:
        raise ValueError(f"Face index out of range in {path}: v={v.shape[0]}, face_min={int(f.min())}, face_max={int(f.max())}")
    return v, f
