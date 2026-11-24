# H2 ignition kernel, learnable species weights (uncertainty weighting)

**Work in progress, for tests only:** the option is not validated (no significant gain so far), see
[theory](../../docs/theory.md#uncertainty-weighting).

Same case as `H2_IGN_KERNEL` (database and networks identical), with learnable weights of the species in the loss
(`uncertainty_weighting: true` in `networks_params.yaml`) and the element correction `yk_renormalization: 3` in
`nn_testing.yaml`. The model is written in `MODELS/MODEL_H2_UNCERTAINTY/`; the learned weights are in the training log
(`UNCERTAINTY_WEIGHTS`). Compare with `H2_IGN_KERNEL` tested with `yk_renormalization: 3`, over several trainings.

The database (`STOCH_DTB_H2_POF_2023/`) can be generated here, or copied from `H2_IGN_KERNEL` to skip the generation.

# Usage

After installing the package (see the main `README.md`), you can run the following commands to generate and process the database. Replace `X` with the number of cores you want to use for parallel execution:

```bash
mpirun -n X python generate_stoch_dtb.py
python dtb_processing.py
python ann_model_learning.py
```

To test the resulting neural network on a 0D case, use:
```bash
mpirun -n 1 python ann_model_testing.py
```
