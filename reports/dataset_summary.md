# Dataset Summary

- HDF5 input: `${WORKSPACE_ROOT}/dt4d.hdf5`
- Split: `train`
- Requested sample_id: `bearVGG/bearVGG_Actions0`
- Limit: `-1`
- Prepared samples: `1`
- Frame count range: `121..121`
- Vertex count range: `22471..22471`
- Face count range: `44160..44160`

## Canonical Choice
- Using frame 0 as canonical mesh for each sequence (assumption, documented in metadata).

## Outputs
- Manifest: `${PROJECT_ROOT}/outputs/sample_manifest.json`
- Per-sample files in `outputs/samples/<sample_id>/` include `canonical.obj` and `metadata.json`.
