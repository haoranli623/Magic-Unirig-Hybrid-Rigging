from __future__ import annotations

from dataclasses import replace
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.mesh_utils import read_obj, safe_sample_id
from src.rig_format import RigData, parse_magic_pred_txt, save_rig
from src.unirig_skin_bridge import UniRigSkinBridgeConfig, run_unirig_learned_skin_bridge
from src.weight_transfer import WeightTransferConfig, transfer_weights_hybrid


def run_magicarticulate_demo(
    *,
    repo_dir: Path,
    project_root: Path,
    canonical_obj: Path,
    run_name: str,
    pretrained_weights: Path,
    input_pc_num: int,
    hier_order: bool,
    python_exe: str = "python3",
    cuda_visible_devices: str | None = None,
) -> tuple[int, str, str, Path]:
    repo_dir = Path(repo_dir).resolve()
    output_root = project_root / "outputs" / "magic_tmp"
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        "demo.py",
        "--input_path",
        str(canonical_obj.resolve()),
        "--pretrained_weights",
        str(pretrained_weights.resolve()),
        "--save_name",
        run_name,
        "--input_pc_num",
        str(input_pc_num),
        "--output_dir",
        str(output_root.resolve()),
    ]
    if hier_order:
        cmd.append("--hier_order")

    env = os.environ.copy()
    env["HF_HOME"] = str((project_root / ".hf_cache").resolve())
    env["TRANSFORMERS_CACHE"] = str((project_root / ".hf_cache").resolve())
    prev_py = env.get("PYTHONPATH", "")
    if env.get("MAGIC_DISABLE_PKG_PYTHONPATH", "0") == "1":
        if prev_py:
            env["PYTHONPATH"] = prev_py
        elif "PYTHONPATH" in env:
            env.pop("PYTHONPATH")
    else:
        repo_py = str(repo_dir.resolve())
        project_py = str((project_root / ".pkg").resolve())
        path_parts: list[str] = [project_py, repo_py]
        if prev_py:
            path_parts.append(prev_py)
        env["PYTHONPATH"] = ":".join(path_parts)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    run_dir = output_root / run_name
    return proc.returncode, proc.stdout, proc.stderr, run_dir


def _load_sampled_mesh(raw_magic_run_dir: Path, canonical_obj: Path) -> tuple[np.ndarray, np.ndarray, str]:
    sampled_mesh_path = raw_magic_run_dir / "canonical_mesh.obj"
    if sampled_mesh_path.exists():
        sampled_vertices, sampled_faces = read_obj(sampled_mesh_path)
        return sampled_vertices, sampled_faces, str(sampled_mesh_path.resolve())

    sampled_vertices, sampled_faces = read_obj(canonical_obj)
    return sampled_vertices, sampled_faces, str(canonical_obj.resolve())


def convert_magic_output_to_rig(
    *,
    sample_id: str,
    canonical_obj: Path,
    raw_magic_run_dir: Path,
    rig_dir: Path,
    source_label: str,
    weight_transfer_cfg: WeightTransferConfig,
    project_root: Path,
    unirig_skin_cfg: UniRigSkinBridgeConfig | None = None,
    allow_weight_fallback: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    rig_dir.mkdir(parents=True, exist_ok=True)
    raw_out = rig_dir / "raw_magic_output"
    raw_out.mkdir(parents=True, exist_ok=True)

    mesh_base = Path(canonical_obj).stem
    pred_txt = raw_magic_run_dir / f"{mesh_base}_pred.txt"
    if not pred_txt.exists():
        candidates = sorted(raw_magic_run_dir.glob("*_pred.txt"))
        if not candidates:
            raise FileNotFoundError(
                f"MagicArticulate output missing *_pred.txt in {raw_magic_run_dir}"
            )
        pred_txt = candidates[0]

    for f in raw_magic_run_dir.glob("*"):
        if not f.is_file():
            continue
        dst = raw_out / f.name
        if f.resolve() == dst.resolve():
            continue
        shutil.copy2(f, dst)

    names, joints, parents, root_idx = parse_magic_pred_txt(pred_txt)
    vertices, faces = read_obj(canonical_obj)
    sampled_vertices, _sampled_faces, sampled_source = _load_sampled_mesh(raw_magic_run_dir, canonical_obj)

    fallback_used = False
    fallback_reason = None
    try:
        if weight_transfer_cfg.method == "unirig_learned_skin":
            if unirig_skin_cfg is None:
                raise RuntimeError("unirig_skin_cfg is required for method=unirig_learned_skin")
            weights, wt_report = run_unirig_learned_skin_bridge(
                project_root=project_root,
                sample_id=sample_id,
                canonical_obj=canonical_obj,
                joints=joints,
                parents=parents,
                names=names,
                weight_transfer_cfg=weight_transfer_cfg,
                unirig_cfg=unirig_skin_cfg,
            )
        else:
            weights, wt_report = transfer_weights_hybrid(
                canonical_vertices=vertices,
                canonical_faces=faces,
                sampled_vertices=sampled_vertices,
                joints_target=joints,
                parents=parents,
                config=weight_transfer_cfg,
            )
    except Exception as e:
        if not allow_weight_fallback or weight_transfer_cfg.method != "unirig_learned_skin":
            raise
        fallback_used = True
        fallback_reason = str(e)
        fallback_cfg = replace(weight_transfer_cfg, method="unirig_reskin")
        weights, wt_report = transfer_weights_hybrid(
            canonical_vertices=vertices,
            canonical_faces=faces,
            sampled_vertices=sampled_vertices,
            joints_target=joints,
            parents=parents,
            config=fallback_cfg,
        )
        wt_report["fallback"] = {
            "from_method": weight_transfer_cfg.method,
            "to_method": fallback_cfg.method,
            "reason": fallback_reason,
        }

    assumptions = [
        "MagicArticulate public demo provides skeleton structure; native skinning weights are not directly exposed.",
        (
            "Default hybrid weight transfer: method=unirig_learned_skin "
            "(UniRig learned skin predictor + UniRig-style reskin transfer)."
        ),
    ]
    if fallback_used:
        assumptions.append("UniRig learned skin prediction failed for this sample; used legacy heuristic+reskin fallback.")
    assumptions.append(f"Applied weight method: {wt_report.get('weight_method', weight_transfer_cfg.method)}.")

    source_value = source_label
    if weight_transfer_cfg.method == "unirig_learned_skin" and not fallback_used:
        source_value = "MagicArticulate skeleton + UniRig learned skin + UniRig reskin transfer"
    if fallback_used:
        source_value = "MagicArticulate skeleton + heuristic seed + UniRig reskin transfer (fallback)"

    rig = RigData(
        joint_names=names,
        joint_positions=joints,
        parents=parents,
        root_index=root_idx,
        weights=weights,
        source=source_value,
        has_native_weights=False,
        assumptions=assumptions,
    )

    rig_json = rig_dir / "rig.json"
    weights_npy = rig_dir / "weights.npy"
    save_rig(rig, rig_json, weights_npy)

    (rig_dir / "weight_transfer_config.json").write_text(
        json.dumps(weight_transfer_cfg.__dict__, indent=2), encoding="utf-8"
    )

    wt_report_payload = {
        "sample_id": sample_id,
        "canonical_obj": str(Path(canonical_obj).resolve()),
        "sampled_source_mesh": sampled_source,
        "num_joints": int(joints.shape[0]),
        "num_vertices": int(vertices.shape[0]),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        **wt_report,
    }
    (rig_dir / "weight_transfer_report.json").write_text(
        json.dumps(wt_report_payload, indent=2), encoding="utf-8"
    )

    meta = {
        "sample_id": sample_id,
        "raw_magic_pred_txt": str(pred_txt.resolve()),
        "num_joints": int(joints.shape[0]),
        "num_vertices": int(vertices.shape[0]),
        "has_native_weights": False,
        "weight_method_requested": weight_transfer_cfg.method,
        "weight_method_applied": wt_report.get("weight_method", weight_transfer_cfg.method),
        "fallback_used": bool(fallback_used),
        "conversion_notes": assumptions,
    }
    (rig_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rig_json, weights_npy, meta


def run_magic_for_manifest_entry(
    *,
    project_root: Path,
    repo_dir: Path,
    sample_entry: dict[str, Any],
    pretrained_weights: Path,
    input_pc_num: int,
    hier_order: bool,
    python_exe: str,
    cuda_visible_devices: str | None,
    weight_transfer_cfg: WeightTransferConfig,
    unirig_skin_cfg: UniRigSkinBridgeConfig | None = None,
    allow_weight_fallback: bool = True,
) -> dict[str, Any]:
    sample_id = sample_entry["sample_id"]
    safe_id = sample_entry.get("safe_id", safe_sample_id(sample_id))
    canonical_obj = Path(sample_entry["canonical_obj"])
    rig_dir = project_root / "outputs" / "rigs" / safe_id

    ret, out, err, run_dir = run_magicarticulate_demo(
        repo_dir=repo_dir,
        project_root=project_root,
        canonical_obj=canonical_obj,
        run_name=f"ma_{safe_id}",
        pretrained_weights=pretrained_weights,
        input_pc_num=input_pc_num,
        hier_order=hier_order,
        python_exe=python_exe,
        cuda_visible_devices=cuda_visible_devices,
    )

    (rig_dir / "logs").mkdir(parents=True, exist_ok=True)
    (rig_dir / "logs" / "stdout.txt").write_text(out, encoding="utf-8")
    (rig_dir / "logs" / "stderr.txt").write_text(err, encoding="utf-8")

    if ret != 0:
        return {
            "sample_id": sample_id,
            "safe_id": safe_id,
            "status": "failed",
            "returncode": ret,
            "rig_dir": str(rig_dir.resolve()),
            "error": "MagicArticulate demo command failed; see logs/stderr.txt",
        }

    rig_json, weights_npy, meta = convert_magic_output_to_rig(
        sample_id=sample_id,
        canonical_obj=canonical_obj,
        raw_magic_run_dir=run_dir,
        rig_dir=rig_dir,
        source_label="MagicArticulate skeleton + UniRig-style reskin",
        weight_transfer_cfg=weight_transfer_cfg,
        project_root=project_root,
        unirig_skin_cfg=unirig_skin_cfg,
        allow_weight_fallback=allow_weight_fallback,
    )

    return {
        "sample_id": sample_id,
        "safe_id": safe_id,
        "status": "ok",
        "rig_dir": str(rig_dir.resolve()),
        "rig_json": str(rig_json.resolve()),
        "weights_npy": str(weights_npy.resolve()),
        "num_joints": meta["num_joints"],
        "weight_method_requested": meta["weight_method_requested"],
        "weight_method_applied": meta["weight_method_applied"],
        "fallback_used": bool(meta["fallback_used"]),
    }
