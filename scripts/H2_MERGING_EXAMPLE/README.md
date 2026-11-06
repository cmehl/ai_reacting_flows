# Merging of H2 databases

Merges the stochastic databases of `H2_IGN_KERNEL` (ignition kernel) and `H2_IGN_KERNEL_RESIDENCE_TIME` (same case with
renewal of the particles) into `MERGED_DTB_H2_MERGED/solutions.h5`, then processes it as in `H2_IGN_KERNEL`
(`database_type: merged`). Both databases have the same time step (5e-7 s), which the merging does not check.

# Usage

Generate first the two databases in their case folders (`mpirun -n X python generate_stoch_dtb.py`), then:

```bash
python merge_databases.py
python dtb_processing.py
```

Training and testing are not part of this example: they are done as in `H2_IGN_KERNEL`, with
`database_path: MERGED_DTB_H2_MERGED/dtb_resampled` in `networks_params.yaml`.
