from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from src.mesh_utils import ensure_dir, safe_sample_id, write_obj


@dataclass
class SequenceSample:
    sample_id: str
    character_id: str
    vertices: np.ndarray  # (T, N, 3)
    faces: np.ndarray  # (F, 3), zero-indexed

    @property
    def canonical_vertices(self) -> np.ndarray:
        return self.vertices[0]

    @property
    def num_frames(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[1])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def _decode_h5_str(x: Any) -> str:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def normalize_faces(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces (F,3), got {faces.shape}")

    min_idx = int(faces.min())
    max_idx = int(faces.max())
    if min_idx == 0:
        out = faces
    elif min_idx == 1 and max_idx == n_vertices:
        out = faces - 1
    else:
        raise ValueError(f"Unsupported face indexing [{min_idx}, {max_idx}] for N={n_vertices}")

    if int(out.min()) < 0 or int(out.max()) >= n_vertices:
        raise ValueError("Face indices out of range after normalization")
    return out.astype(np.int32)


class DT4D:
    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

    def list_all_sample_ids(self) -> list[str]:
        out: list[str] = []
        with h5py.File(self.h5_path, "r") as f:
            for ch in sorted(f.keys()):
                if ch == "data_split":
                    continue
                if not isinstance(f[ch], h5py.Group):
                    continue
                for motion in sorted(f[ch].keys()):
                    sid = f"{ch}/{motion}"
                    g = f[sid]
                    if isinstance(g, h5py.Group) and {"vertices", "faces"}.issubset(set(g.keys())):
                        out.append(sid)
        return out

    def list_split_sample_ids(self, split: str) -> list[str]:
        split = split.lower()
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val, or all")

        if split == "all":
            return self.list_all_sample_ids()

        with h5py.File(self.h5_path, "r") as f:
            if "data_split" not in f or split not in f["data_split"]:
                raise ValueError("Expected /data_split/{train,val} in HDF5")
            arr = f["data_split"][split][:]
            return [_decode_h5_str(x) for x in arr]

    def resolve_sample_id(self, sample_id: str) -> str:
        with h5py.File(self.h5_path, "r") as f:
            if sample_id in f:
                return sample_id

        all_ids = self.list_all_sample_ids()
        match = [sid for sid in all_ids if sid.endswith(f"/{sample_id}")]
        if len(match) == 1:
            return match[0]
        if not match:
            raise ValueError(f"Cannot resolve sample id: {sample_id}")
        raise ValueError(f"Ambiguous sample id '{sample_id}'; matches {match[:5]}")

    def load_sample(self, sample_id: str) -> SequenceSample:
        with h5py.File(self.h5_path, "r") as f:
            if sample_id not in f:
                raise KeyError(f"Sample path missing in HDF5: {sample_id}")
            g = f[sample_id]
            if not isinstance(g, h5py.Group):
                raise ValueError(f"Expected group at {sample_id}")

            if "vertices" not in g or "faces" not in g:
                raise ValueError(f"Missing vertices/faces in {sample_id}")

            vertices = np.asarray(g["vertices"][:], dtype=np.float32)
            if vertices.ndim == 2 and vertices.shape[-1] == 3:
                vertices = vertices[None, ...]
            if vertices.ndim != 3 or vertices.shape[-1] != 3:
                raise ValueError(f"Invalid vertices shape in {sample_id}: {vertices.shape}")

            faces = normalize_faces(np.asarray(g["faces"][:], dtype=np.int64), int(vertices.shape[1]))
            return SequenceSample(
                sample_id=sample_id,
                character_id=sample_id.split("/")[0],
                vertices=vertices,
                faces=faces,
            )


def prepare_dt4d_samples(
    h5_path: Path,
    outputs_dir: Path,
    report_path: Path,
    split: str,
    limit: int | None,
    sample_id: str | None,
    export_frames: int,
) -> dict[str, Any]:
    ds = DT4D(h5_path)
    if sample_id:
        selected = [ds.resolve_sample_id(sample_id)]
    else:
        selected = ds.list_split_sample_ids(split)

    if limit is not None and limit > 0:
        selected = selected[:limit]

    samples_dir = outputs_dir / "samples"
    ensure_dir(samples_dir)

    manifest_entries: list[dict[str, Any]] = []
    n_frames: list[int] = []
    n_verts: list[int] = []
    n_faces: list[int] = []

    for sid in tqdm(selected, desc="Preparing DT4D"):
        sample = ds.load_sample(sid)
        safe_id = safe_sample_id(sid)
        out_dir = samples_dir / safe_id
        ensure_dir(out_dir)

        canonical_obj = out_dir / "canonical.obj"
        write_obj(canonical_obj, sample.canonical_vertices, sample.faces)

        n_dump = min(max(export_frames, 0), sample.num_frames)
        for t in range(n_dump):
            write_obj(out_dir / f"frame_{t:03d}.obj", sample.vertices[t], sample.faces)

        meta = {
            "sample_id": sample.sample_id,
            "character_id": sample.character_id,
            "h5_group": sample.sample_id,
            "h5_path": str(Path(h5_path).resolve()),
            "num_frames": sample.num_frames,
            "num_vertices": sample.num_vertices,
            "num_faces": sample.num_faces,
            "canonical_source": "vertices[0]",
            "notes": "DT4D HDF5 does not expose explicit canonical/rest mesh field; frame 0 exported as canonical.",
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        manifest_entries.append(
            {
                "sample_id": sample.sample_id,
                "character_id": sample.character_id,
                "safe_id": safe_id,
                "sample_dir": str(out_dir.resolve()),
                "canonical_obj": str(canonical_obj.resolve()),
                "metadata_json": str((out_dir / "metadata.json").resolve()),
                "num_frames": sample.num_frames,
                "num_vertices": sample.num_vertices,
                "num_faces": sample.num_faces,
            }
        )
        n_frames.append(sample.num_frames)
        n_verts.append(sample.num_vertices)
        n_faces.append(sample.num_faces)

    manifest = {
        "h5_path": str(Path(h5_path).resolve()),
        "split": split,
        "sample_id_filter": sample_id,
        "limit": limit,
        "num_samples": len(manifest_entries),
        "samples": manifest_entries,
    }
    manifest_path = outputs_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = [
        "# Dataset Summary",
        "",
        f"- HDF5 input: `{Path(h5_path).resolve()}`",
        f"- Split: `{split}`",
        f"- Requested sample_id: `{sample_id}`",
        f"- Limit: `{limit}`",
        f"- Prepared samples: `{len(manifest_entries)}`",
    ]
    if n_frames:
        report += [
            f"- Frame count range: `{min(n_frames)}..{max(n_frames)}`",
            f"- Vertex count range: `{min(n_verts)}..{max(n_verts)}`",
            f"- Face count range: `{min(n_faces)}..{max(n_faces)}`",
        ]

    report += [
        "",
        "## Canonical Choice",
        "- Using frame 0 as canonical mesh for each sequence (assumption, documented in metadata).",
        "",
        "## Outputs",
        f"- Manifest: `{manifest_path.resolve()}`",
        "- Per-sample files in `outputs/samples/<sample_id>/` include `canonical.obj` and `metadata.json`.",
    ]

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    return {
        "manifest_path": str(manifest_path.resolve()),
        "report_path": str(report_path.resolve()),
        "num_samples": len(manifest_entries),
    }
