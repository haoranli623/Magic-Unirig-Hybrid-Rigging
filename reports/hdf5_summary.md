# DT4D HDF5 Summary

## Confirmed Facts
- File: `${WORKSPACE_ROOT}/dt4d.hdf5`
- Total datasets: `3546`
- Leaf dataset counts: `{'faces': 1772, 'vertices': 1772, 'train': 1, 'val': 1}`
- Sample groups with `vertices` + `faces`: `1772`
- Vertices are stored as trajectories `(T, N, 3)`: `T` in `6..695`, `N` in `4505..29330`.
- Faces are triangles `(F, 3)`: `F` in `8388..57248`.
- `/data_split/train`: `1418` items; `/data_split/val`: `354` items.
- Split references missing in file: train=`0`, val=`0`.

## Likely Interpretations
- `sample_id` is naturally represented as `<character>/<motion>`.
- In absence of explicit canonical/rest dataset, `vertices[0]` is a practical canonical mesh for baseline preparation.
- No direct rig/weights fields are present in DT4D HDF5; rigging must come from external auto-rigging (MagicArticulate).

## Unresolved Ambiguity
- The file schema alone does not prove `vertices[0]` is a neutral pose for every sequence.
- Exact train/test interpretation for cross-motion transfer must be aligned with RigMo protocol text.
