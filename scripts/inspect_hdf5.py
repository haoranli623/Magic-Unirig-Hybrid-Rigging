#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def preview_dataset(dset: h5py.Dataset) -> dict[str, Any]:
    meta: dict[str, Any] = {"shape": tuple(int(x) for x in dset.shape), "dtype": str(dset.dtype)}
    if dset.size == 0:
        meta["sample_head"] = []
        return meta

    try:
        if dset.ndim == 0:
            arr = np.asarray(dset[()])
        elif dset.ndim == 1:
            arr = np.asarray(dset[: min(8, dset.shape[0])])
        else:
            slices = [slice(0, 1)] * dset.ndim
            arr = np.asarray(dset[tuple(slices)])

        flat = arr.reshape(-1)
        meta["sample_head"] = [_py(x) for x in flat[:8]]
        if arr.dtype.kind in {"i", "u", "f"}:
            meta["sample_min"] = float(np.nanmin(arr))
            meta["sample_max"] = float(np.nanmax(arr))
    except Exception as exc:
        meta["preview_error"] = str(exc)
    return meta


def decode_str_array(arr: np.ndarray, n: int = 5) -> list[str]:
    out = []
    for x in arr[:n]:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        out.append(str(x))
    return out


def inspect_h5(h5_path: Path, structure_out: Path, summary_out: Path) -> None:
    structure_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"[FILE] {h5_path.resolve()}"]
    ds_meta: dict[str, dict[str, Any]] = {}

    with h5py.File(h5_path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            p = f"/{name}" if name else "/"
            if isinstance(obj, h5py.Group):
                if name:
                    lines.append(f"[GROUP] {p}")
                return
            if isinstance(obj, h5py.Dataset):
                m = preview_dataset(obj)
                ds_meta[p] = m
                lines.append(f"[DATASET] {p} shape={m['shape']} dtype={m['dtype']}")
                if "sample_head" in m:
                    lines.append(f"  sample_head={m['sample_head']}")
                if "sample_min" in m and "sample_max" in m:
                    lines.append(f"  sample_range=[{m['sample_min']:.6g}, {m['sample_max']:.6g}]")
                if "preview_error" in m:
                    lines.append(f"  preview_error={m['preview_error']}")

        f.visititems(visit)

        leaf_names = Counter(p.split("/")[-1] for p in ds_meta)
        sample_groups = []
        v_stats = []
        f_stats = []
        nonconforming = []

        for top in sorted(f.keys()):
            if top == "data_split":
                continue
            if not isinstance(f[top], h5py.Group):
                continue
            for motion in sorted(f[top].keys()):
                gpath = f"/{top}/{motion}"
                obj = f[gpath]
                if not isinstance(obj, h5py.Group):
                    nonconforming.append(gpath)
                    continue
                keys = set(obj.keys())
                if not {"vertices", "faces"}.issubset(keys):
                    nonconforming.append(gpath)
                    continue
                sample_groups.append(gpath)
                v = obj["vertices"]
                fa = obj["faces"]
                if v.ndim == 3 and v.shape[-1] == 3:
                    v_stats.append((int(v.shape[0]), int(v.shape[1])))
                else:
                    nonconforming.append(gpath)
                if fa.ndim == 2 and fa.shape[-1] == 3:
                    f_stats.append(int(fa.shape[0]))
                else:
                    nonconforming.append(gpath)

        split_info = {}
        split_examples = {}
        split_missing = {}
        if "data_split" in f and "train" in f["data_split"] and "val" in f["data_split"]:
            for split in ["train", "val"]:
                arr = f["data_split"][split][:]
                split_info[split] = int(arr.shape[0])
                split_examples[split] = decode_str_array(arr, n=5)
                missing = 0
                for x in arr:
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    if isinstance(x, bytes):
                        x = x.decode("utf-8", errors="replace")
                    if str(x) not in f:
                        missing += 1
                split_missing[split] = missing

    structure_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    frames = [a for a, _ in v_stats] if v_stats else []
    nverts = [b for _, b in v_stats] if v_stats else []

    md = [
        "# DT4D HDF5 Summary",
        "",
        "## Confirmed Facts",
        f"- File: `{h5_path.resolve()}`",
        f"- Total datasets: `{len(ds_meta)}`",
        f"- Leaf dataset counts: `{dict(leaf_names)}`",
        f"- Sample groups with `vertices` + `faces`: `{len(sample_groups)}`",
    ]

    if v_stats:
        md.append(
            f"- Vertices are stored as trajectories `(T, N, 3)`: `T` in `{min(frames)}..{max(frames)}`, `N` in `{min(nverts)}..{max(nverts)}`."
        )
    if f_stats:
        md.append(f"- Faces are triangles `(F, 3)`: `F` in `{min(f_stats)}..{max(f_stats)}`.")
    if split_info:
        md.append(f"- `/data_split/train`: `{split_info['train']}` items; `/data_split/val`: `{split_info['val']}` items.")
        md.append(
            f"- Split references missing in file: train=`{split_missing['train']}`, val=`{split_missing['val']}`."
        )

    md += [
        "",
        "## Likely Interpretations",
        "- `sample_id` is naturally represented as `<character>/<motion>`.",
        "- In absence of explicit canonical/rest dataset, `vertices[0]` is a practical canonical mesh for baseline preparation.",
        "- No direct rig/weights fields are present in DT4D HDF5; rigging must come from external auto-rigging (MagicArticulate).",
        "",
        "## Unresolved Ambiguity",
        "- The file schema alone does not prove `vertices[0]` is a neutral pose for every sequence.",
        "- Exact train/test interpretation for cross-motion transfer must be aligned with RigMo protocol text.",
    ]
    if nonconforming:
        md.append(f"- Found `{len(nonconforming)}` nonconforming groups (sample schema check).")

    summary_out.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inspect DT4D HDF5 structure")
    parser.add_argument("--h5", type=Path, default=root.parent / "dt4d.hdf5")
    parser.add_argument("--structure-out", type=Path, default=root / "reports" / "hdf5_structure.txt")
    parser.add_argument("--summary-out", type=Path, default=root / "reports" / "hdf5_summary.md")
    args = parser.parse_args()

    if not args.h5.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {args.h5}")

    inspect_h5(args.h5, args.structure_out, args.summary_out)
    print(f"Wrote: {args.structure_out}")
    print(f"Wrote: {args.summary_out}")


if __name__ == "__main__":
    main()
