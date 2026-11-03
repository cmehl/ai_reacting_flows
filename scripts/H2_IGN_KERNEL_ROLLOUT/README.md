# H2 ignition kernel, rollout training

Same case as `H2_IGN_KERNEL`, with networks trained on rollouts: the network is applied 3 times in a row on its
own predictions (`rollout_steps` in `dtb_params.yaml`), with `output_omegas: false`. See the
[rollout theory](../../docs/theory.md#rollout-training). The model is written in `MODELS/MODEL_H2_ROLLOUT/`.

The stochastic reactors database (`STOCH_DTB_H2_POF_2023/solutions.h5`) can be generated here, or copied from
`H2_IGN_KERNEL` to skip the first step.

# Usage

After installing the package (see the main `README.md`), you can run the following commands. Replace `X` with the number of cores you want to use for parallel execution:

```bash
mpirun -n X python generate_stoch_dtb.py    # stochastic reactors database
mpirun -n X python generate_multi_dt.py     # reference trajectories of rollout: solutions_rollout.h5
python dtb_processing.py
python ann_model_learning.py
```

To test the resulting neural network on a 0D case, use:
```bash
python ann_model_testing.py
```
