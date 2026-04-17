#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import prepare_dt4d_samples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DT4D canonical meshes and metadata")
    parser.add_argument("--h5", type=Path, default=ROOT.parent / "dt4d.hdf5")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--export-frames", type=int, default=2)
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--report-path", type=Path, default=ROOT / "reports" / "dataset_summary.md")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    out = prepare_dt4d_samples(
        h5_path=args.h5,
        outputs_dir=args.outputs_dir,
        report_path=args.report_path,
        split=args.split,
        limit=args.limit,
        sample_id=args.sample_id,
        export_frames=args.export_frames,
    )
    print(f"Prepared samples: {out['num_samples']}")
    print(f"Manifest: {out['manifest_path']}")
    print(f"Report:   {out['report_path']}")


if __name__ == "__main__":
    main()
