# Deviations From Paper

This file records non-trivial differences between RigMo Table-1 baseline text and this repository's executable pipeline.

## A. Intentional Default Change (Hybrid)

Default route in this repo is now:
- MagicArticulate skeleton source
- UniRig learned skin prediction + UniRig-style remap/reskin transfer
- Torch+Adam fitting backend

This is intentionally a hybrid experiment and is not presented as pure Magic downstream behavior.

## B. Known Gaps vs Exact Paper Equivalence

1. Skinning weights
- Paper does not provide full public details for how Magic baseline weights are produced in their internal setup.
- Public Magic demo outputs skeletons, not complete native animation-ready weights in this repo.
- We therefore use UniRig learned skin prediction and documented transfer/remap as a hybrid substitute.

2. Canonical/rest definition
- DT4D file used here does not expose an explicit canonical field in this pipeline.
- This repo uses frame-0 canonical extraction from prepared samples.

3. Optimization schedule details
- Paper-level description supports per-sequence transform optimization but does not fully lock all optimizer hyperparameters.
- We use an auditable Adam schedule and report it per task.

## C. What Is Still Needed for Closer Equivalence

- Public release of full Magic baseline downstream weighting/optimization details from authors.
- Reference implementation and exact hyperparameters for RigMo Table-1 baseline runs.
- A pure-Magic downstream weighting path (without UniRig learned-skin substitution) for strict baseline identity.
