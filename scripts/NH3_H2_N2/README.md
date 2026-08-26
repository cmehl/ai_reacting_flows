# NH3_H2_N2

Working directories for the NH3/H2/N2 ANN surrogate, covering two data
sources (CFD field snapshots vs. 0D stochastic-reactor cloud) each run
against two chemical mechanisms (detailed vs. reduced), for four cases
total:

| Folder | Data source | Mechanism | Raw data pipeline |
|---|---|---|---|
| `CFD_DETAILED/` | Real CFD snapshots (`DATA/output/*.h5`) | detailed (`detailed_no_Ar.yaml`, 32 species) | `build_raw_database.py` |
| `CFD_REDUCED/` | Real CFD snapshots (`DATA/output/*.h5`) | reduced (`STEC_A_noAR.yaml`, 18 species) | `build_raw_database.py` |
| `STOCH_DETAILED/` | 0D stochastic-particle cloud (Wan et al.) | detailed (32 species) | `generate_stoch_dtb.py` |
| `STOCH_REDUCED/` | 0D stochastic-particle cloud (Wan et al.) | reduced (18 species) | `generate_stoch_dtb.py` |

`CFD_*` and `STOCH_*` are genuinely different data sources for the *same*
combustor/mechanism pair, not just a naming split — `STOCH_DETAILED` is not
derived from `CFD_DETAILED`. They're grouped side by side here (and by the
CFD/STOCH prefix, rather than e.g. `DETAILED_CFD`/`DETAILED_STOCH`) so the
two data-generation strategies stay visually distinct at a glance, matching
the `CFD — .../STOCH — ...` grouping used in this project's model-explorer
tooling elsewhere.

## Per-case layout

Each of the four case folders bundles the full pipeline for that case:

```
<CASE>/
  build_raw_params.yaml + build_raw_database.py   (CFD_* only)
  dtb_params.yaml + generate_stoch_dtb.py          (STOCH_* only)
  dtb_processing.yaml + dtb_processing.py
  networks_params_mlp_percluster.yaml
  networks_params_perspecies.yaml
  networks_params_resnet.yaml
  ann_model_learning.py
  test_ann_vs_cvode.yaml + test_ann_vs_cvode.py
```

## Three architectures, one training script

Each case ships three `networks_params_<variant>.yaml` files, all trained
on the *same* processed database (`dtb_processing.yaml` -> `database_name`)
so they're directly comparable:

- **`mlp_percluster`** — one plain `MLP` per cluster (`networks_types: [MLP, MLP]`).
- **`perspecies`** — one `PerSpeciesMLP` per cluster, i.e. one sub-network
  per output species, **all sub-networks the same size** within a model
  (as opposed to `PerSpeciesMLPSized`, which sizes each sub-network to its
  species' difficulty — not used here).
- **`resnet`** — same `MLP`-per-cluster shape as `mlp_percluster`, but the
  hidden layers use `layers_type: resnet` (residual blocks,
  `Custom_layers.ResidualBlock`) instead of `dense`.

`ai_reacting_flows.ann_model_generation.NN_manager.NN_manager` always reads
exactly `networks_params.yaml` in the run folder — there's no way to point
it at a differently-named file. `ann_model_learning.py` in each case folder
works around that: it copies the variant you ask for over the fixed
filename, then trains:

```bash
python ann_model_learning.py mlp_percluster   # or: perspecies | resnet
```

The trained model lands in `MODELS/MODEL_<model_name_suffix>`, where the
suffix (set inside each `networks_params_<variant>.yaml`) already encodes
the variant name, so training all three into the same case folder doesn't
collide.

## Full workflow (one case)

```bash
ml_arf && source .venv/bin/activate   # from the repo root, see main README

cd scripts/NH3_H2_N2/CFD_DETAILED   # or CFD_REDUCED / STOCH_DETAILED / STOCH_REDUCED

# 1. Raw database (CFD_* cases: MPI; STOCH_* cases: MPI, particle cloud)
mpirun -n <nb_procs> python build_raw_database.py      # CFD_*
mpirun -n <nb_procs> python generate_stoch_dtb.py       # STOCH_*

# 2. Process into train/val CSVs (clustering, log-transform, T_threshold, ...)
python dtb_processing.py

# 3. Train whichever architecture(s) you want
python ann_model_learning.py mlp_percluster
python ann_model_learning.py perspecies
python ann_model_learning.py resnet

# 4. Test against held-out CFD slices (see test_ann_vs_cvode.yaml comments
#    for how to point it at the variant you just trained)
python test_ann_vs_cvode.py
```

## Testing against real CFD ground truth (`test_ann_vs_cvode.yaml`)

All four cases test through
`ai_reacting_flows.ann_model_generation.cfd_snapshot_testing.CFDSnapshotTester`,
against the exact same held-out ParaView CSV slices — `CFD_*` cases test
against their own `DATA/testing/*.csv`; `STOCH_*` cases test against
`../CFD_DETAILED/DATA/testing/*.csv` or `../CFD_REDUCED/DATA/testing/*.csv`
(the sibling case's data, built once and shared), so a 0D-stochastic-trained
model and a CFD-snapshot-trained model can be compared on identical ground
truth.

Set `mask_cvode_below_threshold: true` (already the default in every
`test_ann_vs_cvode.yaml` here) so CVODE itself is forced to identity below
`T_threshold` on both sides of the comparison — without it, RMSE is
inflated by a masking artifact where the ANN skips cold cells but CVODE
doesn't. `models_folder` and `output_file` in `test_ann_vs_cvode.yaml`
default to the `mlp_percluster` variant; edit both to whichever
`MODEL_<...>` / result filename you actually want to test.

## Notes

- `dtb_processing.yaml` / `build_raw_params.yaml` / `dtb_params.yaml` are
  run **once per case**, shared by all three architectures — don't re-run
  them per variant.
- `build_raw_database.py` writes its output to `CFD_DTB_<suffix>/` (matching
  `database_type: cfd` in `dtb_processing.yaml`); don't confuse this with
  the `STOCH_DTB_<suffix>/` naming `generate_stoch_dtb.py`/`database_type:
  stoch` cases use.
- These four folders are templates in the same sense as `scripts/H2_IGN_KERNEL/`
  etc. (see the top-level project `CLAUDE.md`): copy one to start a new
  case, don't edit these in place for one-off experiments — use `.idea/`
  (gitignored scratch space) for exploratory work instead.
