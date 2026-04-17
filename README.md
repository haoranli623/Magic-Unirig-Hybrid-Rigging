# magicarticulate_dt4d_repro

Research pipeline for DT4D that now defaults to a **hybrid experiment**:

1. **Skeleton source:** MagicArticulate (public demo output)
2. **Weight stage:** UniRig learned skin prediction + UniRig-style reskin/remap transfer
3. **Optimization:** UniRig_on_dt4d-style Torch + Adam per-frame fitting

This is intentionally **not** a pure MagicArticulate downstream baseline.

## Scope and Inputs

- Local dataset input: `../dt4d.hdf5`
- No dependency on unrelated local projects
- Public MagicArticulate repo is under `third_party/MagicArticulate`

## Main Layout

- `scripts/inspect_hdf5.py`
- `scripts/prepare_dt4d.py`
- `scripts/run_magicarticulate.py`
- `scripts/run_optimize.py`
- `scripts/run_eval.py`
- `scripts/render_visualizations.py`
- `src/magic_runner.py` (Magic inference + conversion)
- `src/unirig_skin_bridge.py` (Magic->UniRig learned-skin bridge + transfer)
- `src/weight_transfer.py` (UniRig-style remap/reskin + legacy fallback paths)
- `src/fit_adam.py` (default Adam fitting backend)
- `src/optimizer.py` (legacy SciPy fallback)
- `src/reporting.py`, `src/metrics.py`, `src/protocol.py`
- `outputs/`, `reports/`

## Install

```bash
cd magicarticulate_dt4d_repro
python3 -m pip install -r requirements.txt --target .pkg
```

Run scripts with:

```bash
PYTHONPATH=.pkg:. python3 <script>
```

## Quick Run Order

1. Inspect DT4D schema:

```bash
PYTHONPATH=.pkg:. python3 scripts/inspect_hdf5.py --h5 ../dt4d.hdf5
```

2. Prepare a small subset:

```bash
PYTHONPATH=.pkg:. python3 scripts/prepare_dt4d.py --h5 ../dt4d.hdf5 --split train --limit 4 --export-frames 1
```

3. Run Magic skeleton inference + hybrid weight transfer:

```bash
PYTHONPATH=.pkg:. python3 scripts/run_magicarticulate.py \
  --manifest outputs/sample_manifest.json \
  --magic-repo third_party/MagicArticulate \
  --pretrained-weights third_party/MagicArticulate/skeleton_ckpt/checkpoint_trainonv2_hier.pth \
  --limit 2
```

4. Run evaluation (default backend is Adam):

```bash
PYTHONPATH=.pkg:. python3 scripts/run_eval.py \
  --h5 ../dt4d.hdf5 \
  --train-limit 1 \
  --cross-limit 1 \
  --optimizer-backend adam
```

5. Optional legacy comparison (non-default):

```bash
PYTHONPATH=.pkg:. python3 scripts/run_eval.py \
  --h5 ../dt4d.hdf5 \
  --train-limit 1 \
  --cross-limit 0 \
  --optimizer-backend scipy_legacy
```

6. Render qualitative outputs for a task:

```bash
PYTHONPATH=.pkg:. python3 scripts/render_visualizations.py \
  --task-id <source_safe>__to__<target_safe>
```

## Default Pipeline Details

### Weight transfer default

Default is `method=unirig_learned_skin`:
- run Magic skeleton inference on canonical mesh
- build UniRig `predict_skeleton.npz` bridge input from Magic joints/parents + canonical mesh
- run UniRig learned skin prediction on sampled space
- transfer sampled-space skin to canonical mesh via UniRig-style reskin:
  - kNN remap
  - median/mean aggregation (default: median)
  - edge diffusion
  - threshold pruning + renormalization

Legacy fallback paths remain available:
- `unirig_reskin` (heuristic seed + transfer)
- `joint_distance_legacy`

Saved per rig:
- `outputs/rigs/<safe_id>/weight_transfer_config.json`
- `outputs/rigs/<safe_id>/weight_transfer_report.json`

### Optimization default

`src/fit_adam.py::fit_sequence_adam` optimizes per frame:
- `axis_angle[T, J, 3]`
- `root_trans[T, 3]`

Using FK + LBS on fixed rig/weights, with warm-start from previous frame.

Saved per eval task:
- `fit_config.json`
- `loss_trace.json`
- `fit_frame_summary.json`
- `motion_params_adam.npz`
- `recon_vertices.npy`
- `metrics.json`
- `task_summary.json`

## Evaluation Outputs

- `outputs/eval/results.json`
- `outputs/eval/results.csv`
- `reports/table1_magicarticulate_repro.md`

CSV includes:
- `optimizer_backend`
- `weight_method`
- `n_iters`
- `lr`
- `trace_every`

## Visualization Outputs

Per task folder under `outputs/visualizations/<task_id>/`:
- `original_mesh.mp4`
- `reconstruction_mesh.mp4`
- `original_vs_reconstruction.mp4`
- `rig_overlay.png`
- `rig_turntable.mp4`
- `loss_curve.png` (from `loss_trace.json`)
- keyframe comparison PNGs

Task-typed presentation variants are also written:
- `train_recon__original_mesh.mp4`, `train_recon__reconstruction_mesh.mp4`, ...
- `cross_motion__original_mesh.mp4`, `cross_motion__reconstruction_mesh.mp4`, ...

`viz_meta.json` now explicitly records:
- `task_type`
- `source_sample_id` / `target_sample_id`
- semantics for cross-motion (`skeleton_from_source`, `gt_animation_from_target`, `reconstruction_under_source_rig`)
- typed output paths under `outputs_typed`

## Known Limits

- Exact equivalence to RigMo authors' internal implementation is not guaranteed.
- Public MagicArticulate release is used for skeleton inference; full native Magic weights are not exposed here.
- Hybrid downstream (UniRig learned skin + UniRig transfer + Adam fitting) is intentional and documented as a separate experimental setting.
- UniRig learned-skin bridge relies on additional runtime dependencies (`lightning`, `python-box`, `timm`, `addict`, `spconv`), and local shims for missing scatter/cluster ops in this environment.
- If MagicArticulate runtime tries to fetch external models/checkpoints in offline environments, inference may fail until those assets are cached locally.
