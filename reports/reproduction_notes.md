# Reproduction Notes

Target context: RigMo Table 1 baseline family on DT4D.

Current default in this repo is a **hybrid experiment**:
- MagicArticulate skeleton inference
- UniRig learned skin prediction + UniRig-style transfer/remap
- UniRig_on_dt4d-style Adam fitting

## 1) Paper-Confirmed

- RigMo Table 1 evaluates `Training Reconstruction` and `Cross-Motion Transfer` with CD-L1/CD-L2.
- Baseline description indicates automatic rigging from canonical pose, then per-sequence transformation optimization.

## 2) Public-Code-Confirmed

- Public MagicArticulate `demo.py` provides skeleton prediction artifacts (`*_pred.txt`, `*_skel.obj`) from mesh input.
- Public UniRig-main provides learned skin prediction model checkpoints and code, plus reusable reskin/remap logic (`kNN -> diffusion -> threshold/renorm`).
- UniRig_on_dt4d includes a per-frame Adam fitting style with warm-start and FK+LBS deformation.

## 3) Implemented Hybrid Default (this repo)

- `src/magic_runner.py`: keeps Magic skeleton stage as source of joints/parents.
- `src/unirig_skin_bridge.py`: runs UniRig learned skin prediction on Magic skeleton/canonical mesh inputs and writes bridge diagnostics.
- `src/weight_transfer.py`: applies UniRig-style reskin transfer to canonical/original benchmark mesh.
- `src/fit_adam.py`: default optimization backend (`axis_angle + root_trans`, per frame).
- `scripts/run_eval.py`: default `--optimizer-backend adam`.

## 4) Run Artifacts Added for Audit

Per rig:
- `weight_transfer_config.json`
- `weight_transfer_report.json`

Per eval task:
- `fit_config.json`
- `loss_trace.json`
- `fit_frame_summary.json`
- `motion_params_adam.npz`
- `recon_vertices.npy`
- `metrics.json`
- `task_summary.json`

## 5) Remaining Uncertainty

- RigMo does not fully specify all baseline hyperparameters/details needed for exact numerical equivalence.
- This hybrid setting is not pure Magic baseline reproduction because downstream skin prediction is from UniRig.
- Therefore this project is an auditable hybrid reproduction experiment, not a claim of exact private baseline parity.
