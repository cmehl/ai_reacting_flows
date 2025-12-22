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