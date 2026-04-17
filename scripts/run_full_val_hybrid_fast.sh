#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H5_PATH="${H5_PATH:-$ROOT/../dt4d.hdf5}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Fast relaunch knobs (lighter than previous full-val run)
N_ITERS="${N_ITERS:-120}"
VERTEX_SAMPLE_COUNT="${VERTEX_SAMPLE_COUNT:-800}"
TRACE_EVERY="${TRACE_EVERY:-40}"
SEED="${SEED:-0}"

RUN_ID="${RUN_ID:-fullval_hybrid_fast_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$ROOT/outputs/fullval_runs/$RUN_ID"
STATE_DIR="$RUN_DIR/state"
REPORT_DIR="$RUN_DIR/reports"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$STATE_DIR" "$REPORT_DIR" "$EVAL_DIR"

LOG_PATH="$RUN_DIR/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date -Iseconds)] Start fast full-val hybrid eval-only run"
echo "ROOT=$ROOT"
echo "H5_PATH=$H5_PATH"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"

cat > "$STATE_DIR/launch_config.env" <<EOF
ROOT=$ROOT
H5_PATH=$H5_PATH
PYTHON_BIN=$PYTHON_BIN
N_ITERS=$N_ITERS
VERTEX_SAMPLE_COUNT=$VERTEX_SAMPLE_COUNT
TRACE_EVERY=$TRACE_EVERY
SEED=$SEED
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
EOF

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
    "split_counts": {"train": len(train_ids), "val": len(val_ids)},
    "task_policy": {
        "train_recon": "disabled (train-limit=0)",
        "cross_motion": "all val targets (cross-enum=all_val_targets)",
    },
    "rigged_train_count": len(rigged_train),
    "expected_cross_tasks": len(tasks),
    "config": {
        "optimizer_backend": "adam",
        "n_iters": int($N_ITERS),
        "vertex_sample_count": int($VERTEX_SAMPLE_COUNT),
        "trace_every": int($TRACE_EVERY),
        "seed": int($SEED),
    },
}
(state_dir / "task_policy_and_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
  --n-iters "$N_ITERS" \\
  --vertex-sample-count "$VERTEX_SAMPLE_COUNT" \\
  --trace-every "$TRACE_EVERY" \\
  --outputs-dir "$EVAL_DIR" \\
  --results-json "$EVAL_DIR/results.json" \\
  --results-csv "$EVAL_DIR/results.csv" \\
  --table-report "$REPORT_DIR/table1_magicarticulate_repro.md" \\
  --protocol-report "$REPORT_DIR/protocol.md"
EOF

echo "[$(date -Iseconds)] Running full-val eval with lighter config..."
bash "$STATE_DIR/eval_command.sh"

echo "[$(date -Iseconds)] Fast full-val run completed."
echo "Results JSON: $EVAL_DIR/results.json"
echo "Results CSV:  $EVAL_DIR/results.csv"
echo "Table report: $REPORT_DIR/table1_magicarticulate_repro.md"
