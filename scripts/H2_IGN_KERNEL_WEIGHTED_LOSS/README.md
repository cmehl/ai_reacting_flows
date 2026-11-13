# H2 ignition kernel, loss weighted by the increments

**Work in progress:** loss weighting is experimental and has not improved the tests of the networks so far (see
[theory](../../docs/theory.md#loss-weighting)). Compare with `H2_IGN_KERNEL` over several seeds before drawing conclusions.

Same case as `H2_IGN_KERNEL` (database and testing inputs identical), with a database undersampled on the norm of
the scaled increments $\Omega$ (`undersample_Omega` in `dtb_processing.py`, instead of the HRR undersampling) and the
loss weighted by the increments of the species (`loss_weighting_p`, `loss_weighting_eps` in `networks_params.yaml`, see
[theory](../../docs/theory.md#loss-weighting)): small increments, as in burnt gases, count more than with the usual
loss. The model is written in `MODELS/MODEL_H2_WEIGHTED_LOSS/`. Compare its 0D tests (`Temperature_stability.png`,
`log_changes/`) with those of `H2_IGN_KERNEL`, and vary `loss_weighting_p` (0, 0.5, 1), `loss_weighting_eps` and `loss_weighting_per_species`.

The database (`STOCH_DTB_H2_POF_2023/`) can be generated here, or copied from `H2_IGN_KERNEL` to skip the
generation; `dtb_processing.py` must be run here (learning database `dtb_omega_resampled`, sampled on $\Omega$).

# Usage

After installing the package (see the main `README.md`), you can run the following commands to generate and process the database. Replace `X` with the number of cores you want to use for parallel execution:

```bash
mpirun -n X python generate_stoch_dtb.py
python dtb_processing.py
python ann_model_learning.py
```

To test the resulting neural network on a 0D case, use:
```bash
python ann_model_testing.py
```
