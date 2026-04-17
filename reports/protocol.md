# Protocol

## Hybrid Default (This Repo)
- Skeleton source: MagicArticulate public demo output.
- Weight stage: UniRig-style reskin/remap logic applied after Magic skeleton inference.
- Optimization default: Torch + Adam per-frame fitting with warm start.

## Paper-confirmed behavior (RigMo)
- Table 1 evaluates Training Reconstruction and Cross-Motion Transfer on DT4D test split.
- Auto-rigging baselines generate initial bone structures from canonical pose followed by per-sequence transform optimization.

## Reproduction implementation
- Train reconstruction tasks: 1 sampled from `/data_split/train`.
- Cross-motion tasks: 1 sampled as (source in train, target in val) for same character.
- Cross-motion policy used: `fixed_rig_opt_test`.
- Optimizer backend used: `adam`.
