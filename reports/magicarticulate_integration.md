# MagicArticulate Integration

## Environment
- MagicArticulate repo: `${PROJECT_ROOT}/third_party/MagicArticulate`
- Checkpoint: `${PROJECT_ROOT}/third_party/MagicArticulate/skeleton_ckpt/checkpoint_trainonv2_hier.pth`
- CUDA_VISIBLE_DEVICES: `1`

## Hybrid Default
- Skeleton source: public MagicArticulate `demo.py`.
- Weight transfer: `unirig_learned_skin`.
- Learned skin source: UniRig-main `unirig_skin` predictor when `weight_method=unirig_learned_skin`.
- Transfer stage: UniRig-style sampled-space -> canonical-mesh reskin.
- Weight transfer config: `{'method': 'unirig_learned_skin', 'seed_top_k': 4, 'sigma_mode': 'median', 'sample_method': 'median', 'nearest_samples': 7, 'iter_steps': 1, 'threshold': 0.03, 'alpha': 2.0}`
- UniRig bridge config: `{'enabled': True, 'python_exe': '${WORKSPACE_ROOT}/conda_envs/dpo_env/bin/python', 'unirig_repo': '${WORKSPACE_ROOT}/UniRig-main', 'task_config': '${PROJECT_ROOT}/configs/unirig_skin_magic_bridge.yaml', 'cls_label': 'dt4d', 'seed': 123, 'normalize_into_min': -1.0, 'normalize_into_max': 1.0}`
- Allow fallback to heuristic path: `False`

## Run Summary
- Requested meshes: `1`
- Successful rig conversions: `1`
- Failed runs: `0`
- Result JSON: `${PROJECT_ROOT}/outputs/rigs/magic_results.json`

## Public-code-confirmed behavior
- Inference executed through official `demo.py` in public MagicArticulate repository.
- Raw outputs include `*_pred.txt`, `*_skel.obj`, and mesh copies in `raw_magic_output/`.
- UniRig learned skin inference is executed through UniRig-main `run.py` (skin task) when enabled.

## Reproduction assumptions
- Public MagicArticulate skeleton demo does not emit native skinning weights.
- This project's default downstream now uses UniRig learned skin prediction plus UniRig-style reskin transfer.
- This is a hybrid design and not a pure MagicArticulate downstream baseline.
