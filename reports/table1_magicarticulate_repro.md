# Table 1 Style: Hybrid (Magic Skeleton + UniRig-Style Reskin + Adam Opt.)

Values below are reported as Chamfer metrics in units of ×10^-3 (i.e., raw metric × 1000).

| Method | Training Reconstruction CD-L1 | Training Reconstruction CD-L2 | Cross-Motion Transfer CD-L1 | Cross-Motion Transfer CD-L2 | Mean CD-L1 | Mean CD-L2 |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid Magic+UniRig-style downstream (this reproduction) | 51.548 | 7.261 | 52.481 | 3.394 | 52.015 | 5.328 |

## Notes
- Training and cross-motion means are computed over successful tasks only.
- See `outputs/eval/results.json` for per-task details and skipped/failed cases.
