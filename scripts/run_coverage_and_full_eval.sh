#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
H5_PATH="${H5_PATH:-$ROOT/../dt4d.hdf5}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SEED="${SEED:-0}"
N_ITERS="${N_ITERS:-60}"
VERTEX_SAMPLE_COUNT="${VERTEX_SAMPLE_COUNT:-600}"
TRACE_EVERY="${TRACE_EVERY:-20}"
LR="${LR:-1e-2}"

RUN_ID="${RUN_ID:-full_hybrid_learnedskin_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT/outputs/full_runs/$RUN_ID"
STATE_DIR="$RUN_DIR/state"
LOG_DIR="$RUN_DIR/logs"
REPORT_DIR="$RUN_DIR/reports"
EVAL_DIR="$RUN_DIR/eval"
PREP_DIR="$RUN_DIR/prep"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$REPORT_DIR" "$EVAL_DIR" "$PREP_DIR"
cd "$ROOT"

LOG_PATH="$LOG_DIR/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date -Iseconds)] Start coverage expansion + full eval"
echo "ROOT=$ROOT"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"

cat > "$STATE_DIR/launch_env.json" <<JSON
{
  "root": "$ROOT",
  "python_bin": "$PYTHON_BIN",
  "h5_path": "$H5_PATH",
  "cuda_device": "$CUDA_DEVICE",
  "seed": $SEED,
  "n_iters": $N_ITERS,
  "vertex_sample_count": $VERTEX_SAMPLE_COUNT,
  "trace_every": $TRACE_EVERY,
  "lr": $LR,
  "run_id": "$RUN_ID"
}
JSON

export MAGIC_DISABLE_PKG_PYTHONPATH=1
export MAGIC_ATTN_IMPL=eager
export HF_HOME="$ROOT/.hf_cache"
export TRANSFORMERS_CACHE="$ROOT/.hf_cache"
export MPLCONFIGDIR="$ROOT/.cache/mpl"

mkdir -p "$MPLCONFIGDIR"

echo "[$(date -Iseconds)] Step 1: coverage-before audit + missing candidate selection"
"$PYTHON_BIN" - <<PY
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from src.dataset import DT4D
from src.mesh_utils import safe_sample_id

root = Path("$ROOT")
state_dir = Path("$STATE_DIR")
ds = DT4D(Path("$H5_PATH"))

train_ids = ds.list_split_sample_ids("train")
val_ids = ds.list_split_sample_ids("val")
val_chars = sorted({sid.split('/')[0] for sid in val_ids})
train_by_char = defaultdict(list)
for sid in train_ids:
    train_by_char[sid.split('/')[0]].append(sid)

valid_sources = []
all_rigged = []
for sid in train_ids:
    safe = safe_sample_id(sid)
    rig_dir = root / "outputs" / "rigs" / safe
    meta_p = rig_dir / "conversion_meta.json"
    w_p = rig_dir / "weights.npy"
    if not (meta_p.exists() and w_p.exists()):
        continue
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        w_n = int(np.load(w_p, mmap_mode="r").shape[0])
        v_n = int(ds.load_sample(sid).num_vertices)
    except Exception:
        continue

    rec = {
        "sample_id": sid,
        "character_id": sid.split('/')[0],
        "weight_method_applied": meta.get("weight_method_applied"),
        "fallback_used": bool(meta.get("fallback_used")),
        "weights_vertices": w_n,
        "source_vertices": v_n,
        "vertex_match": bool(w_n == v_n),
    }
    all_rigged.append(rec)
    if rec["weight_method_applied"] == "unirig_learned_skin" and (not rec["fallback_used"]) and rec["vertex_match"]:
        valid_sources.append(rec)

covered_chars = sorted({r["character_id"] for r in valid_sources})
missing_chars = [c for c in val_chars if c not in covered_chars]

candidates = []
for ch in missing_chars:
    cands = sorted(train_by_char.get(ch, []))
    if cands:
        candidates.append({
            "character_id": ch,
            "candidate_source_id": cands[0],
            "num_train_samples_for_char": len(cands),
        })

summary = {
    "num_train_ids": len(train_ids),
    "num_val_ids": len(val_ids),
    "num_val_characters": len(val_chars),
    "num_rigged_train_sources_any": len(all_rigged),
    "num_learned_skin_nonfallback_sources": len(valid_sources),
    "num_covered_val_characters": len(covered_chars),
    "num_missing_val_characters": len(missing_chars),
    "missing_val_characters": missing_chars,
    "num_regen_candidates": len(candidates),
}

(state_dir / "coverage_before.json").write_text(json.dumps({
    "summary": summary,
    "covered_characters": covered_chars,
    "valid_sources": valid_sources,
    "all_rigged_sources": all_rigged,
}, indent=2), encoding="utf-8")
(state_dir / "coverage_regen_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[$(date -Iseconds)] Step 2: prepare canonical meshes for regen candidates"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

from src.dataset import DT4D
from src.mesh_utils import ensure_dir, safe_sample_id, write_obj

root = Path("$ROOT")
state_dir = Path("$STATE_DIR")
prep_dir = Path("$PREP_DIR")

candidates = json.loads((state_dir / "coverage_regen_candidates.json").read_text(encoding="utf-8"))
ds = DT4D(Path("$H5_PATH"))

samples_dir = prep_dir / "samples"
ensure_dir(samples_dir)
entries = []
for c in candidates:
    sid = c["candidate_source_id"]
    sample = ds.load_sample(sid)
    safe = safe_sample_id(sid)
    out_dir = samples_dir / safe
    ensure_dir(out_dir)
    canonical_obj = out_dir / "canonical.obj"
    write_obj(canonical_obj, sample.canonical_vertices, sample.faces)
    meta = {
        "sample_id": sample.sample_id,
        "character_id": sample.character_id,
        "num_frames": sample.num_frames,
        "num_vertices": sample.num_vertices,
        "num_faces": sample.num_faces,
        "canonical_source": "vertices[0]",
        "purpose": "learned_skin_coverage_expansion",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    entries.append({
        "sample_id": sample.sample_id,
        "character_id": sample.character_id,
        "safe_id": safe,
        "sample_dir": str(out_dir.resolve()),
        "canonical_obj": str(canonical_obj.resolve()),
        "metadata_json": str((out_dir / "metadata.json").resolve()),
        "num_frames": sample.num_frames,
        "num_vertices": sample.num_vertices,
        "num_faces": sample.num_faces,
    })

manifest = {
    "h5_path": str(Path("$H5_PATH").resolve()),
    "split": "train",
    "sample_id_filter": None,
    "limit": -1,
    "num_samples": len(entries),
    "samples": entries,
}
manifest_path = prep_dir / "coverage_manifest_for_magic.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"prepared_candidates": len(entries), "manifest_path": str(manifest_path)}, indent=2))
PY

echo "[$(date -Iseconds)] Step 3: run learned-skin regeneration for missing candidates (no fallback)"
cat > "$STATE_DIR/coverage_magic_command.sh" <<CMD
MAGIC_DISABLE_PKG_PYTHONPATH=1 \\
MAGIC_ATTN_IMPL=eager \\
HF_HOME="$ROOT/.hf_cache" \\
TRANSFORMERS_CACHE="$ROOT/.hf_cache" \\
MPLCONFIGDIR="$ROOT/.cache/mpl" \\
"$PYTHON_BIN" "$ROOT/scripts/run_magicarticulate.py" \\
  --manifest "$PREP_DIR/coverage_manifest_for_magic.json" \\
  --limit -1 \\
  --magic-repo "$ROOT/third_party/MagicArticulate" \\
  --pretrained-weights "$ROOT/third_party/MagicArticulate/skeleton_ckpt/checkpoint_trainonv2_hier.pth" \\
  --python-exe "$PYTHON_BIN" \\
  --cuda-visible-devices "$CUDA_DEVICE" \\
  --unirig-python-exe "$PYTHON_BIN" \\
  --weight-method unirig_learned_skin \\
  --no-weight-fallback \\
  --report-path "$REPORT_DIR/magicarticulate_integration_coverage.md"
CMD
bash "$STATE_DIR/coverage_magic_command.sh"

echo "[$(date -Iseconds)] Step 4: coverage-after audit + allowlist build"
"$PYTHON_BIN" - <<PY
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from src.dataset import DT4D
from src.mesh_utils import safe_sample_id
from src.protocol import build_cross_motion_tasks_all_val_targets

root = Path("$ROOT")
state_dir = Path("$STATE_DIR")
ds = DT4D(Path("$H5_PATH"))

before = json.loads((state_dir / "coverage_before.json").read_text(encoding="utf-8"))
candidates = json.loads((state_dir / "coverage_regen_candidates.json").read_text(encoding="utf-8"))

train_ids = ds.list_split_sample_ids("train")
val_ids = ds.list_split_sample_ids("val")
val_chars = sorted({sid.split('/')[0] for sid in val_ids})

valid_sources = []
valid_by_char = defaultdict(list)
for sid in train_ids:
    safe = safe_sample_id(sid)
    rig_dir = root / "outputs" / "rigs" / safe
    meta_p = rig_dir / "conversion_meta.json"
    w_p = rig_dir / "weights.npy"
    if not (meta_p.exists() and w_p.exists()):
        continue
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        w_n = int(np.load(w_p, mmap_mode="r").shape[0])
        v_n = int(ds.load_sample(sid).num_vertices)
    except Exception:
        continue
    if meta.get("weight_method_applied") == "unirig_learned_skin" and (not bool(meta.get("fallback_used"))) and w_n == v_n:
        rec = {
            "sample_id": sid,
            "character_id": sid.split('/')[0],
            "weights_vertices": w_n,
            "source_vertices": v_n,
            "weight_method_applied": meta.get("weight_method_applied"),
            "fallback_used": bool(meta.get("fallback_used")),
        }
        valid_sources.append(rec)
        valid_by_char[rec["character_id"]].append(rec)

allowlist = sorted([r["sample_id"] for r in valid_sources])
covered_chars = sorted(valid_by_char.keys())
missing_chars = [c for c in val_chars if c not in covered_chars]

candidate_status = []
for c in candidates:
    sid = c["candidate_source_id"]
    safe = safe_sample_id(sid)
    meta_p = root / "outputs" / "rigs" / safe / "conversion_meta.json"
    status = {
        **c,
        "status": "missing_conversion_meta",
        "weight_method_applied": None,
        "fallback_used": None,
        "vertex_match": None,
        "notes": None,
    }
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            w_n = int(np.load(root / "outputs" / "rigs" / safe / "weights.npy", mmap_mode="r").shape[0])
            v_n = int(ds.load_sample(sid).num_vertices)
            status.update({
                "weight_method_applied": meta.get("weight_method_applied"),
                "fallback_used": bool(meta.get("fallback_used")),
                "vertex_match": bool(w_n == v_n),
            })
            if meta.get("weight_method_applied") == "unirig_learned_skin" and (not bool(meta.get("fallback_used"))) and w_n == v_n:
                status["status"] = "learned_skin_ok"
            else:
                status["status"] = "not_qualified"
                status["notes"] = meta.get("conversion_notes", [])
        except Exception as e:
            status["status"] = "error"
            status["notes"] = str(e)
    candidate_status.append(status)

cross_tasks = build_cross_motion_tasks_all_val_targets(allowlist, val_ids, limit=-1, seed=int($SEED))

coverage_after = {
    "before_summary": before["summary"],
    "after_summary": {
        "num_learned_skin_nonfallback_sources": len(allowlist),
        "num_covered_val_characters": len(covered_chars),
        "num_missing_val_characters": len(missing_chars),
        "missing_val_characters": missing_chars,
    },
    "candidate_outcomes": candidate_status,
}

(state_dir / "coverage_after.json").write_text(json.dumps(coverage_after, indent=2), encoding="utf-8")

allow_payload = {
    "run_id": "$RUN_ID",
    "criteria": {
        "weight_method_applied": "unirig_learned_skin",
        "fallback_used": False,
        "weights_source_vertex_match": True,
    },
    "allowlist_count": len(allowlist),
    "allowlist_sample_ids": allowlist,
    "covered_val_characters": covered_chars,
    "missing_val_characters": missing_chars,
}
allow_path = state_dir / "learned_skin_source_allowlist.json"
allow_path.write_text(json.dumps(allow_payload, indent=2), encoding="utf-8")

provenance = {
    "allowlist_sample_ids": allowlist,
    "source_rig_provenance": valid_sources,
}
(state_dir / "source_rig_provenance_summary.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

policy = {
    "definition": "full eval over learned-skin-only source pool",
    "allowlist_count": len(allowlist),
    "train_recon_tasks": len(allowlist),
    "cross_motion_tasks": len(cross_tasks),
    "total_tasks": len(allowlist) + len(cross_tasks),
    "cross_enum": "all_val_targets",
    "cross_policy": "fixed_rig_opt_test",
    "optimizer_backend": "adam",
    "optimizer_config": {
        "n_iters": int($N_ITERS),
        "vertex_sample_count": int($VERTEX_SAMPLE_COUNT),
        "trace_every": int($TRACE_EVERY),
        "lr": float($LR),
    },
}
(state_dir / "task_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
print(json.dumps({
    "after_learned_skin_nonfallback_sources": len(allowlist),
    "after_covered_val_characters": len(covered_chars),
    "cross_motion_tasks": len(cross_tasks),
}, indent=2))
PY

echo "[$(date -Iseconds)] Step 5: launch full eval using explicit learned-skin allowlist"
cat > "$STATE_DIR/eval_command.sh" <<CMD
MAGIC_DISABLE_PKG_PYTHONPATH=1 \\
HF_HOME="$ROOT/.hf_cache" \\
TRANSFORMERS_CACHE="$ROOT/.hf_cache" \\
MPLCONFIGDIR="$ROOT/.cache/mpl" \\
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \\
"$PYTHON_BIN" "$ROOT/scripts/run_eval.py" \\
  --h5 "$H5_PATH" \\
  --source-allowlist "$STATE_DIR/learned_skin_source_allowlist.json" \\
  --allow-missing-rigs \\
  --train-limit -1 \\
  --cross-limit -1 \\
  --cross-enum all_val_targets \\
  --cross-policy fixed_rig_opt_test \\
  --optimizer-backend adam \\
  --n-iters "$N_ITERS" \\
  --lr "$LR" \\
  --vertex-sample-count "$VERTEX_SAMPLE_COUNT" \\
  --trace-every "$TRACE_EVERY" \\
  --device cuda \\
  --seed "$SEED" \\
  --outputs-dir "$EVAL_DIR" \\
  --results-json "$EVAL_DIR/results.json" \\
  --results-csv "$EVAL_DIR/results.csv" \\
  --table-report "$REPORT_DIR/table1_magicarticulate_repro.md" \\
  --protocol-report "$REPORT_DIR/protocol.md"
CMD

bash "$STATE_DIR/eval_command.sh"

echo "[$(date -Iseconds)] Full eval complete"
echo "results_json=$EVAL_DIR/results.json"
echo "results_csv=$EVAL_DIR/results.csv"
echo "table_report=$REPORT_DIR/table1_magicarticulate_repro.md"
