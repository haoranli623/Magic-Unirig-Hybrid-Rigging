# Table 1 Style (Sanity Explicit): Hybrid (Magic Skeleton + UniRig Learned Skin + Adam Opt.)

Values below are reported as Chamfer metrics in units of ×10^-3 (raw metric × 1000).

| Method | Training Reconstruction CD-L1 | Training Reconstruction CD-L2 | Cross-Motion Transfer CD-L1 | Cross-Motion Transfer CD-L2 | Mean CD-L1 | Mean CD-L2 |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid Magic+UniRig learned-skin downstream (sanity explicit) | 24.699 | 0.746 | 27.328 | 0.762 | 26.014 | 0.754 |

## Notes
- Train and cross numbers are each from one explicit sanity task.
- Task IDs:
- train_recon: `bearVGG/bearVGG_Actions0 -> bearVGG/bearVGG_Actions0`
- cross_motion: `bearVGG/bearVGG_Actions0 -> bearVGG/bearVGG_Turn3`
- Full metrics JSON: `outputs/eval/sanity_unirigskin_explicit/results.json`
