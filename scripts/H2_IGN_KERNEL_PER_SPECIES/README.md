# H2 ignition kernel, one network per species

Same case as `H2_IGN_KERNEL` (database, processing and testing inputs identical), with `PerSpeciesMLP`
networks: in each cluster, one MLP per species (8 species, N2 excluded), each predicting the output of its
species from the whole input state. Each network is smaller than the single network of `H2_IGN_KERNEL`
(2 x 10 and 2 x 20 neurons instead of 2 x 20 and 2 x 40). The model is written in `MODELS/MODEL_H2_PER_SPECIES/`.

The database (`STOCH_DTB_H2_POF_2023/`) can be generated here, or copied from `H2_IGN_KERNEL` to skip the
first two steps.

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