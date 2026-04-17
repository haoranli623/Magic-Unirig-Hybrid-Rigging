#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H5_PATH="${H5_PATH:-$ROOT/../dt4d.hdf5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAGIC_REPO="${MAGIC_REPO:-$ROOT/third_party/MagicArticulate}"
MAGIC_CKPT="${MAGIC_CKPT:-$ROOT/third_party/MagicArticulate/skeleton_ckpt/checkpoint_trainonv2_hier.pth}"
INPUT_PC_NUM="${INPUT_PC_NUM:-4096}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:-fullval_hybrid_$(date +%Y%m%d_%H%M%S)}"

RUN_DIR="$ROOT/outputs/fullval_runs/$RUN_ID"
STATE_DIR="$RUN_DIR/state"
REPORT_DIR="$RUN_DIR/reports"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$STATE_DIR" "$REPORT_DIR" "$EVAL_DIR"

LOG_PATH="$RUN_DIR/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date -Iseconds)] Start full-val hybrid run"
echo "ROOT=$ROOT"
echo "H5_PATH=$H5_PATH"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"

cat > "$STATE_DIR/launch_config.env" <<EOF
ROOT=$ROOT
H5_PATH=$H5_PATH
PYTHON_BIN=$PYTHON_BIN
MAGIC_REPO=$MAGIC_REPO
MAGIC_CKPT=$MAGIC_CKPT
INPUT_PC_NUM=$INPUT_PC_NUM
CUDA_VISIBLE_DEVICES_VALUE=$CUDA_VISIBLE_DEVICES_VALUE
SEED=$SEED
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
EOF

PYTHONPATH="$ROOT/.pkg:$ROOT" "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

from src.dataset import DT4D
from src.mesh_utils import safe_sample_id, write_obj

root = Path("$ROOT")
h5_path = Path("$H5_PATH")
state_dir = Path("$STATE_DIR")
outputs_dir = root / "outputs"
samples_dir = outputs_dir / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)

ds = DT4D(h5_path)
train_ids = ds.list_split_sample_ids("train")
val_ids = ds.list_split_sample_ids("val")

train_by_char = {}
for sid in sorted(train_ids):
    ch = sid.split("/")[0]
    train_by_char.setdefault(ch, []).append(sid)

required_sources = []
missing_chars = []
for sid in sorted(val_ids):
    ch = sid.split("/")[0]
    srcs = train_by_char.get(ch, [])
    if not srcs:
        missing_chars.append(ch)
        continue
    required_sources.append(srcs[0])

required_sources = sorted(set(required_sources))

manifest_entries = []
for sid in required_sources:
    sample = ds.load_sample(sid)
    safe = safe_sample_id(sid)
    out_dir = samples_dir / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_obj = out_dir / "canonical.obj"
    if not canonical_obj.exists():
        write_obj(canonical_obj, sample.canonical_vertices, sample.faces)

    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        meta = {
            "sample_id": sample.sample_id,
            "character_id": sample.character_id,
            "h5_group": sample.sample_id,
            "h5_path": str(h5_path.resolve()),
            "num_frames": sample.num_frames,
            "num_vertices": sample.num_vertices,
            "num_faces": sample.num_faces,
            "canonical_source": "vertices[0]",
            "notes": "Auto-generated for full-val hybrid run source rigging.",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    manifest_entries.append(
        {
            "sample_id": sid,
            "character_id": sample.character_id,
            "safe_id": safe,
            "sample_dir": str(out_dir.resolve()),
            "canonical_obj": str(canonical_obj.resolve()),
            "metadata_json": str(meta_path.resolve()),
            "num_frames": sample.num_frames,
            "num_vertices": sample.num_vertices,
            "num_faces": sample.num_faces,
        }
    )

required_manifest = {
    "h5_path": str(h5_path.resolve()),
    "split": "train_for_val_sources",
    "num_samples": len(manifest_entries),
    "samples": manifest_entries,
}
(state_dir / "required_sources_manifest.json").write_text(json.dumps(required_manifest, indent=2), encoding="utf-8")

policy = {
    "split_counts": {"train": len(train_ids), "val": len(val_ids)},
    "task_policy": {
        "train_recon": "disabled (train-limit=0)",
        "cross_motion": "all val targets (cross-enum=all_val_targets)",
        "source_selection": "first train sequence per character (for rig generation pool)",
    },
    "required_source_count": len(required_sources),
    "required_sources": required_sources,
    "missing_characters_without_train_source": sorted(set(missing_chars)),
}
(state_dir / "task_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
print(json.dumps(policy, indent=2))
PY

PYTHONPATH="$ROOT/.pkg:$ROOT" "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

from src.mesh_utils import safe_sample_id

root = Path("$ROOT")
state_dir = Path("$STATE_DIR")
required = json.loads((state_dir / "required_sources_manifest.json").read_text(encoding="utf-8"))

missing_entries = []
for e in required["samples"]:
    sid = e["sample_id"]
    safe = safe_sample_id(sid)
    rig_json = root / "outputs" / "rigs" / safe / "rig.json"
    weights_npy = root / "outputs" / "rigs" / safe / "weights.npy"
    if not (rig_json.exists() and weights_npy.exists()):
        missing_entries.append(e)

missing_manifest = {
    "h5_path": required.get("h5_path"),
    "split": "missing_rig_sources_only",
    "num_samples": len(missing_entries),
    "samples": missing_entries,
}
(state_dir / "missing_rig_manifest.json").write_text(json.dumps(missing_manifest, indent=2), encoding="utf-8")
print(f"missing_sources_for_magic={len(missing_entries)}")
PY

MISSING_COUNT="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
state_dir = Path("$STATE_DIR")
data = json.loads((state_dir / "missing_rig_manifest.json").read_text(encoding="utf-8"))
print(data["num_samples"])
PY
)"

if [[ "$MISSING_COUNT" != "0" ]]; then
  echo "[$(date -Iseconds)] Running MagicArticulate for missing sources: $MISSING_COUNT"
  PYTHONPATH="$ROOT/.pkg:$ROOT" "$PYTHON_BIN" "$ROOT/scripts/run_magicarticulate.py" \
    --manifest "$STATE_DIR/missing_rig_manifest.json" \
    --magic-repo "$MAGIC_REPO" \
    --pretrained-weights "$MAGIC_CKPT" \
    --python-exe "$PYTHON_BIN" \
    --input-pc-num "$INPUT_PC_NUM" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE" \
    --weight-method unirig_reskin \
    --limit -1 \
    --report-path "$REPORT_DIR/magicarticulate_integration.md"
else
  echo "[$(date -Iseconds)] All required source rigs already exist; skipping Magic stage."
fi

PYTHONPATH="$ROOT/.pkg:$ROOT" "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

from src.dataset import DT4D
from src.mesh_utils import safe_sample_id
from src.protocol import build_cross_motion_tasks_all_val_targets

root = Path("$ROOT")
h5_path = Path("$H5_PATH")
state_dir = Path("$STATE_DIR")

ds = DT4D(h5_path)
train_ids = ds.list_split_sample_ids("train")
val_ids = ds.list_split_sample_ids("val")

rigged_train = []
for sid in train_ids:
    safe = safe_sample_id(sid)
    if (root / "outputs" / "rigs" / safe / "rig.json").exists() and (root / "outputs" / "rigs" / safe / "weights.npy").exists():
        rigged_train.append(sid)

tasks = build_cross_motion_tasks_all_val_targets(rigged_train, val_ids, limit=-1, seed=$SEED)
summary = {
    "rigged_train_count": len(rigged_train),
    "val_count": len(val_ids),
    "expected_cross_tasks": len(tasks),
    "cross_enum": "all_val_targets",
    "seed": $SEED,
}
(state_dir / "expected_eval_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

cat > "$STATE_DIR/eval_command.sh" <<EOF
PYTHONPATH="$ROOT/.pkg:$ROOT" "$PYTHON_BIN" "$ROOT/scripts/run_eval.py" \\
  --h5 "$H5_PATH" \\
  --train-limit 0 \\
  --cross-limit -1 \\
  --cross-enum all_val_targets \\
  --seed "$SEED" \\
  --optimizer-backend adam \\
  --outputs-dir "$EVAL_DIR" \\
  --results-json "$EVAL_DIR/results.json" \\
  --results-csv "$EVAL_DIR/results.csv" \\
  --table-report "$REPORT_DIR/table1_magicarticulate_repro.md" \\
  --protocol-report "$REPORT_DIR/protocol.md"
EOF

echo "[$(date -Iseconds)] Running full-val eval..."
bash "$STATE_DIR/eval_command.sh"

echo "[$(date -Iseconds)] Full-val hybrid run completed."
echo "Results JSON: $EVAL_DIR/results.json"
echo "Results CSV:  $EVAL_DIR/results.csv"
echo "Table report: $REPORT_DIR/table1_magicarticulate_repro.md"
