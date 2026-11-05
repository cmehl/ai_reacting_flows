# H2 ignition kernel, renewal of the particles

Same case as `H2_IGN_KERNEL`, with particles renewed to their inlet state after the ignition transient: from
`renewal_start_time` = 1 ms, with a mean residence time `residence_time` = 0.5 ms (partially stirred reactor), so that
fresh gases keep mixing with hot products. See the
[renewal theory](../../docs/theory.md#renewal-of-the-particles-residence_time). The model is written in
`MODELS/MODEL_H2_RESIDENCE_TIME/`.

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
