#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.magic_runner import run_magic_for_manifest_entry  # noqa: E402
from src.unirig_skin_bridge import UniRigSkinBridgeConfig  # noqa: E402
from src.weight_transfer import WeightTransferConfig  # noqa: E402


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official MagicArticulate on prepared canonical meshes")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "sample_manifest.json")
    parser.add_argument(
        "--magic-repo",
        type=Path,
        default=ROOT / "third_party" / "MagicArticulate",
        help="Path to cloned MagicArticulate repo",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=ROOT / "third_party" / "MagicArticulate" / "skeleton_ckpt" / "checkpoint_trainonv2_hier.pth",
    )
    parser.add_argument("--python-exe", type=str, default="python3")
    parser.add_argument("--input-pc-num", type=int, default=4096)
    parser.add_argument("--hier-order", action="store_true", default=True)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--sample-id", type=str, default=None)

    # Hybrid weight-transfer defaults (Magic skeleton + UniRig-style reskin)
    parser.add_argument(
        "--weight-method",
        choices=["unirig_learned_skin", "unirig_reskin", "joint_distance_legacy"],
        default="unirig_learned_skin",
    )
    parser.add_argument("--seed-top-k", type=int, default=4)
    parser.add_argument("--sigma-mode", choices=["median", "mean", "p75"], default="median")
    parser.add_argument("--sample-method", choices=["median", "mean"], default="median")
    parser.add_argument("--nearest-samples", type=int, default=7)
    parser.add_argument("--iter-steps", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument(
        "--no-weight-fallback",
        action="store_true",
        default=False,
        help="Disable fallback to legacy heuristic weighting if UniRig learned skin inference fails.",
    )
    parser.add_argument(
        "--unirig-repo",
        type=Path,
        default=ROOT.parent / "UniRig-main",
        help="Path to UniRig-main repository used for learned skin prediction.",
    )
    parser.add_argument(
        "--unirig-task-config",
        type=Path,
        default=ROOT / "configs" / "unirig_skin_magic_bridge.yaml",
        help="Task config passed to UniRig run.py for learned skin inference.",
    )
    parser.add_argument("--unirig-python-exe", type=str, default=None)
    parser.add_argument("--unirig-cls-label", type=str, default="dt4d")
    parser.add_argument("--unirig-seed", type=int, default=123)

    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "reports" / "magicarticulate_integration.md",
    )
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.magic_repo.exists():
        raise FileNotFoundError(f"MagicArticulate repo path not found: {args.magic_repo}")
    if not args.pretrained_weights.exists():
        raise FileNotFoundError(f"MagicArticulate checkpoint not found: {args.pretrained_weights}")

    wt_cfg = WeightTransferConfig(
        method=args.weight_method,
        seed_top_k=args.seed_top_k,
        sigma_mode=args.sigma_mode,
        sample_method=args.sample_method,
        nearest_samples=args.nearest_samples,
        iter_steps=args.iter_steps,
        threshold=args.threshold,
        alpha=args.alpha,
    )
    unirig_cfg = UniRigSkinBridgeConfig(
        enabled=(args.weight_method == "unirig_learned_skin"),
        python_exe=args.unirig_python_exe or args.python_exe,
        unirig_repo=str(args.unirig_repo.resolve()),
        task_config=str(args.unirig_task_config.resolve()),
        cls_label=args.unirig_cls_label,
        seed=args.unirig_seed,
    )

    manifest = load_manifest(args.manifest)
    entries = list(manifest["samples"])

    if args.sample_id:
        entries = [e for e in entries if e["sample_id"] == args.sample_id or e["sample_id"].endswith(f"/{args.sample_id}")]
        if not entries:
            raise ValueError(f"sample_id not found in manifest: {args.sample_id}")

    if args.limit is not None and args.limit > 0:
        entries = entries[: args.limit]

    results = []
    for entry in tqdm(entries, desc="Running MagicArticulate"):
        try:
            result = run_magic_for_manifest_entry(
                project_root=ROOT,
                repo_dir=args.magic_repo,
                sample_entry=entry,
                pretrained_weights=args.pretrained_weights,
                input_pc_num=args.input_pc_num,
                hier_order=args.hier_order,
                python_exe=args.python_exe,
                cuda_visible_devices=args.cuda_visible_devices,
                weight_transfer_cfg=wt_cfg,
                unirig_skin_cfg=unirig_cfg if args.weight_method == "unirig_learned_skin" else None,
                allow_weight_fallback=not args.no_weight_fallback,
            )
        except Exception as e:
            result = {
                "sample_id": entry.get("sample_id"),
                "safe_id": entry.get("safe_id"),
                "status": "failed",
                "returncode": None,
                "rig_dir": str((ROOT / "outputs" / "rigs" / entry.get("safe_id", "unknown")).resolve()),
                "error": str(e),
            }
        results.append(result)

    out_json = ROOT / "outputs" / "rigs" / "magic_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "weight_transfer": wt_cfg.__dict__,
                "unirig_bridge": unirig_cfg.__dict__,
                "allow_weight_fallback": bool(not args.no_weight_fallback),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = [r for r in results if r.get("status") == "ok"]
    fail = [r for r in results if r.get("status") != "ok"]
    lines = [
        "# MagicArticulate Integration",
        "",
        "## Environment",
        f"- MagicArticulate repo: `{args.magic_repo.resolve()}`",
        f"- Checkpoint: `{args.pretrained_weights.resolve()}`",
        f"- CUDA_VISIBLE_DEVICES: `{args.cuda_visible_devices}`",
        "",
        "## Hybrid Default",
        "- Skeleton source: public MagicArticulate `demo.py`.",
        f"- Weight transfer: `{wt_cfg.method}`.",
        "- Learned skin source: UniRig-main `unirig_skin` predictor when `weight_method=unirig_learned_skin`.",
        "- Transfer stage: UniRig-style sampled-space -> canonical-mesh reskin.",
        f"- Weight transfer config: `{wt_cfg.__dict__}`",
        f"- UniRig bridge config: `{unirig_cfg.__dict__}`",
        f"- Allow fallback to heuristic path: `{bool(not args.no_weight_fallback)}`",
        "",
        "## Run Summary",
        f"- Requested meshes: `{len(entries)}`",
        f"- Successful rig conversions: `{len(ok)}`",
        f"- Failed runs: `{len(fail)}`",
        f"- Result JSON: `{out_json.resolve()}`",
        "",
        "## Public-code-confirmed behavior",
        "- Inference executed through official `demo.py` in public MagicArticulate repository.",
        "- Raw outputs include `*_pred.txt`, `*_skel.obj`, and mesh copies in `raw_magic_output/`.",
        "- UniRig learned skin inference is executed through UniRig-main `run.py` (skin task) when enabled.",
        "",
        "## Reproduction assumptions",
        "- Public MagicArticulate skeleton demo does not emit native skinning weights.",
        "- This project's default downstream now uses UniRig learned skin prediction plus UniRig-style reskin transfer.",
        "- This is a hybrid design and not a pure MagicArticulate downstream baseline.",
    ]

    if fail:
        lines += ["", "## Failure Cases"]
        for r in fail:
            lines.append(
                f"- `{r.get('sample_id')}`: returncode={r.get('returncode')} error={r.get('error')}"
            )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Magic runs complete: {len(ok)} succeeded, {len(fail)} failed")
    print(f"Details: {out_json}")
    print(f"Report:  {args.report_path}")


if __name__ == "__main__":
    main()
