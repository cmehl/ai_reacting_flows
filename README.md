[![DOI](https://zenodo.org/badge/531591043.svg)](https://zenodo.org/badge/latestdoi/531591043)

AI FOR REACTING FLOWS (ARF)
===========================

The project *ai_for_reacting_flows* is a tool enabling to replace complex chemical kinetics calculations in a CFD code by a machine learning model. The idea is that typical exact solvers used to resolve such problems (such as [CVODE](https://computing.llnl.gov/projects/sundials/cvode)) are very expensive. Using fitted models, such as Artificial Neural Networks (ANN), enables to speed-up calculations by a significant factor. 

This package features the following functionalities:

+ A module to build databases from stochastic reactors or canonical flames (0-D reactors, 1-D premixed and diffusion flames)

+ A module to generate machine learning models to advance the chemical state of a mixture


# Package installation

The project itself must be installed as a package, which can be done by going in the root folder and using the following command:

```
python -m pip install -e .
```

Additional details on how to install a home-made package can be found [here](https://realpython.com/python-import/#create-and-install-a-local-package).

In order to run the database generation, an MPI distribution is also necessary. In the current version of the code, the preconized version is OpenMPI/4.1.1.

# Articles

Scripts required to reproduce studies conducted with ARF are gathered in the `articles` folder. A list of the articles is provided in the README file within that folder.

# Notebooks

Notebooks used as examples are in the "notebooks" folder. They are versionned in Git as .py files to ease their tracking. To create notebooks from the .py file, simply prompt the following command from the root folder:

```
jupytext --to notebook notebooks/*.py
```

Note 1: these .py files are in a format where cells can be executed by some IDE (VScode for instance).
Note 2: if you want to commit modifications from your work on some notebooks, you can use:
```
jupytext --set-formats ipynb,py:percent notebooks/*.ipynb
```

# Workflow

Each case is a folder `scripts/<CASE>/` containing YAML inputs and short launcher scripts. All steps are run from inside this folder, as each step reads its YAML from the current directory and finds the results of the previous steps there:

```
mpirun -n X python generate_stoch_dtb.py   # 1. database generation      (dtb_params.yaml)
python dtb_processing.py                   # 2. learning database        (dtb_processing.yaml)
python ann_model_learning.py               # 3. training                 (networks_params.yaml)
python ann_model_testing.py                # 4. testing on simple cases
```

## 1. Database generation

+ **Stochastic reactors**: 0-D stochastic particles following [Wan *et al.*](https://www.sciencedirect.com/science/article/pii/S0010218020302170). Each particle is a chemical state; particles mix (CURL, CURL_MODIFIED, CURL_MODIFIED_DD (work in progress) or EMST models) to mimic diffusion.
+ **Canonical flames**: 0-D reactors, 1-D premixed and 1-D diffusion flames, launched with `generate_0D_dtb.py` and `dtb_params_flmts.yaml`.
+ **Multiple time steps** (optional): `generate_multi_dt.py` re-integrates the database states with several $dt$, so that $dt$ becomes a network input.

**Warning:** an existing `STOCH_DTB_<suffix>` folder is deleted when relaunching generation.

Generated files:

```
STOCH_DTB_<suffix>/                 (FLAMELETS_DTB_<suffix>/ for canonical flames)
├── solutions.h5                    one group ITERATION_<n> per time step
├── <new_file_name>.h5              multi-dt database (optional)
├── dtb_params.yaml, <mechanism>    copies of the inputs
├── pairs_list.pkl                  mixing pairs, one array per time step (+ CURL_rate_list.pkl for
│                                   CURL_MODIFIED: one random mixing extent per pair);
│                                   reused with read_mixing: true if copied to the case folder
└── statistics/                     temperature statistics plots
```

In `solutions.h5`, each group `ITERATION_<n>` contains:

+ `all_states`: state of all active particles: `Temperature` [K], `Pressure` [Pa], species mass fractions $Y_k$, `Mix_frac`, `Equiv_ratio`, `Prog_var`, `HRR` [W/m³], `Time` [s], `Particle_number`, `Inlet_number`, atomic mass fractions `Y_C`, `Y_H`, `Y_O`, `Y_N`, `Mass`.
+ `X` and `Y` (if `build_ml_dtb: true`): states before and after the chemistry step $dt$ (`Temperature`, `Pressure`, $Y_k$, `Prog_var`, `HRR`). These are the ML input/output pairs.

Column names are stored in the `cols` attribute of each dataset. For canonical flames, the file has a single group `ITERATION_00000` with `X` and `Y`, and the simulated cases are listed in `sim_train_*.csv` / `sim_test_*.csv`. The multi-dt file has one group per MPI rank with `X`, `Y` (one set of outputs per $dt$) and `DT` (the $dt$ values [s]).

Post-processing of a stochastic database is shown in `post_processing_stoch_reac.py`.

## 2. Learning database

The class `LearningDatabase` builds the training data: temperature threshold, undersampling based on the heat release rate, transformation (log or Box-Cox), output as $Y_k(t+dt)-Y_k(t)$ if `output_omegas: true`, clustering (k-means or progress variable), train/validation split and standardization. The transforms and the network and k-means features are defined once in `tools/transforms.py`, shared by processing, training and inference.

The script `dtb_processing.py` creates a `LearningDatabase` (which reads `dtb_processing.yaml` and the h5 database) and calls its methods. Typical order:

```python
database = LearningDatabase()
database.apply_temperature_threshold()                     # 1. filtering
database.undersample_2D("PC1", "PC2", min_count=300)       # 2. undersampling (optional)
database.clusterize_dataset()                              # 3. clustering (optional)
database.process_database(plot_distributions=True,         # 4. writing training_data.h5 (always last)
                          distribution_species=["Temperature", "H2O"])
```

Variables passed to the methods are columns of the database (`Temperature`, species names, `HRR`, `Prog_var`) or `PC1`/`PC2`, the first two principal components of the standardized (T, $Y_k$) states (computed automatically when needed).

Data manipulation:

| Method | Description |
|--------|-------------|
| `apply_temperature_threshold()` | Remove states below `T_threshold`. Must be called before clustering |
| `undersample_HRR(var1, var2, hrr_func, keep_low_c, n_samples, n_bins=100)` | Draw `n_samples` points, with a probability favouring bins of the (`var1`, `var2`) space with high HRR. `hrr_func` is applied to the normalized HRR in [0, 1] (e.g. `1 + 5x`). If `keep_low_c`, points with `Prog_var < 0.2` and `T > 1000 K` are kept (needed for ignition) |
| `undersample_1D(var, min_count=1)` | Bin `var` and keep the same number of points in each bin (the size of the smallest bin with at least `min_count` points). Bins with fewer than `min_count` points are kept entirely |
| `undersample_2D(var1, var2, min_count=1)` | Same in 2D |
| `clusterize_dataset(c_bounds=None)` | Split the data in `nb_clusters` clusters. `kmeans`: k-means on the transformed and standardized states. `progvar`: clusters bounded by the progress variable values `c_bounds` (list of `nb_clusters + 1` values from 0 to 1; the progress variable is clipped to [0, 1]) |
| `process_database(plot_distributions=False, distribution_species=[])` | Transform, split and standardize each cluster, write `training_data.h5` and statistics. Optionally plot train/validation distributions of `distribution_species` |

Visualization:

| Method | Description |
|--------|-------------|
| `density_scatter(var_x, var_y)` | Scatter plot colored by point density |
| `compare_resampled_pdfs(var)` | PDF of `var` before and after undersampling |
| `plot_pdf_var(var)` | PDF of `var` |
| `visualize_clusters(var1, var2)` | Scatter plot colored by cluster |
| `print_data_size()` | Number of points in each cluster |

Generated files:

```
STOCH_DTB_<suffix>/
├── resampling_analysis.png               distributions before/after undersampling
└── <database_name>/
    ├── training_data.h5                  training data, one group CLUSTER_<i> per cluster
    ├── kmeans_model.pkl, Xscaler_kmeans.pkl,
    │   kmeans_norm.dat, km_centroids.dat  k-means clustering (c_bounds.pkl for progress variable)
    ├── dtb_statistics_report_<i>.txt     statistics of each cluster
    ├── distrib_<species>_*.png           train/validation distributions (optional)
    └── dtb_processing.yaml, mech_detailed.yaml
```

In `training_data.h5`, each group `CLUSTER_<i>` contains:

+ `X_train`, `X_val`: inputs (`Temperature`, $Y_k$, and $dt$ for multi-dt), transformed and standardized. Columns end with `_X`.
+ `Y_train`, `Y_val`: outputs ($Y_k$ or their differences), transformed and standardized. Columns end with `_Y`.
+ `Xscaler`, `Yscaler`: mean and variance of each column, used for standardization.

`Pressure`, `Prog_var` and `HRR` are not kept, nor `N2` unless `with_N_chemistry: true`.

The processing parameters are stored as attributes of `training_data.h5` (`dt_var`, `fuel`, transforms, `threshold`, `lambda_bct`, `output_omegas`, `with_N_chemistry`, `log_excluded_species`, clustering, number of clusters, `pv_species`). Training and testing read them there, and use the mechanism copy `mech_detailed.yaml`, so the YAML files of database processing and generation are not read again. Databases processed before this was introduced are still read through their YAML files.

## 3. ANN model generation

The class `NN_manager` (PyTorch) trains one network per cluster (MLP or DeepONet) to find $Y_k(t+dt)=ANN(T(t), Y_k(t))$. GPU is used if available.

Generated files:

```
nn_training_<timestamp>.log               training log (in the case folder)
MODELS/MODEL_<suffix>/
├── cluster<i>_model.pth                  PyTorch model
├── cluster<i>_model.h5                   weights and biases per layer (kernel:0, bias:0), for use outside PyTorch
├── norm_param_X_cluster<i>.dat           mean and variance of inputs
├── norm_param_Y_cluster<i>.dat           mean and variance of outputs
├── kmeans_model.pkl, ...                 copy of the clustering files
├── networks_params.yaml, <mechanism>     copies of the inputs
└── training/cluster_<i>/                 loss.png, mass_conservation.png, elements_conservation.png
```

## 4. Model testing

The class `NNTesting` compares the ANN to CVODE on:

+ 0-D homogeneous ignition or pyrolysis
+ 1-D premixed flame reaction rates
+ the validation dataset (scatter plots)

`NNTesting` (`NN_testing.py`) runs the reference and ANN cases; the error metrics are in `NN_testing_metrics.py` and the plots in `NN_testing_plots.py`. The ANN chemistry step itself (clustering, transforms, networks) is in `NN_inference.py` (`NNInference`), shared with the stochastic reactors.

The stochastic reactors can also be run with the ANN instead of CVODE for the chemistry (`ML_inference_flag: true` in `dtb_params.yaml`), to test the networks in mixing conditions. The model is read from `MODELS/<ML_model_folder>` in the case folder; use another `results_folder_suffix` than the database used for training, as the results folder is overwritten.

# Input YAML files

Examples of all files can be found in `scripts/<CASE>/`. Units are SI.

## `dtb_params.yaml` (stochastic reactors)

| Key | Description |
|-----|-------------|
| `mech_file` | Path to the Cantera mechanism (`.yaml`). Species must contain only C, H, O and N atoms: an error is raised for inert species such as Ar or He (use e.g. `mech_nh3_noArHe.yaml`) |
| `inlets` | Dict of inlets (see below) |
| `results_folder_suffix` | Results are written in `STOCH_DTB_<suffix>/` |
| `dtb_file` | Name of the output h5 file (e.g. `solutions.h5`) |
| `build_ml_dtb` | If `true`, store `X`/`Y` pairs for ML |
| `time_step` | Constant $dt$ [s], or dict `{t_start: dt}` to change $dt$ over time (first key must be 0). Write `1.0e-5`, not `1e-5` (read as a string by YAML) |
| `time_max` | End time [s] |
| `automatic_termination` | Stop when temperature variance is low (optional, default `true`) |
| `mixing_model` | `CURL`, `CURL_MODIFIED`, `CURL_MODIFIED_DD` (work in progress) or `EMST` |
| `mixing_time` | Mixing time scale [s] |
| `read_mixing` | If `true`, read mixing pairs from `pairs_list.pkl` in the case folder |
| `T_threshold` | Chemistry is not computed below this temperature [K] |
| `pv_species` | Species defining the progress variable $c = (Y_c - Y_{c,u})/(Y_{c,eq} - Y_{c,u})$, with $Y_c$ the sum of their mass fractions, $u$ the unburnt mixture and $eq$ the equilibrium (e.g. `[H2O]`) |
| `fuel` | List of fuels (e.g. `[H2]`) |
| `calc_mean_traj` | If `true`, write mean trajectories of each inlet |
| `ML_inference_flag` | If `true`, chemistry is computed with the ANN instead of CVODE (optional keys below are then required) |
| `ML_model_folder` | Model folder in `MODELS/` of the case folder (e.g. `MODEL_H2_POF_2023`). Its mechanism and `pv_species` must be those of this file |
| `ML_yk_renormalization` | If `true`, ANN mass fractions are corrected to conserve the elements |
| `ML_hybrid_ann_cvode` | If `true`, particles whose ANN mass fractions sum differs from 1 by more than `ML_hybrid_ann_cvode_tol` are advanced with CVODE |
| `ML_hybrid_ann_cvode_tol` | Tolerance on the sum of mass fractions [-] (e.g. `5.0e-05`) |

Each inlet has a `type`, a number of particles `nb_particles`, `T` [K], `P` [Pa] and an optional `activation_time` [s] (time at which its particles enter the simulation, default 0; at least one inlet must start at 0). Additional keys depend on the type:

+ `cold_premixed`, `burnt_premixed`: `fuel`, `phi` (equivalence ratio)
+ `fuel`: `fuel`
+ `air`, `blank`: none

Multi-dt generation (`generate_multi_dt.py`) reads the same file, with the additional keys:

| Key | Description |
|-----|-------------|
| `time_step_type` | `set`: use the $dt$ values listed in `time_step_range`; `random`: draw `nb_dt` values log-uniformly in `time_step_range` |
| `time_step_range` | List of $dt$ [s] (`set`) or `[dt_min, dt_max]` (`random`) |
| `nb_dt` | Number of $dt$ per state (`random` only) |
| `new_file_name` | Name of the output h5 file |

## `dtb_params_flmts.yaml` (canonical flames)

| Key | Description |
|-----|-------------|
| `mech_file`, `fuel`, `pv_species`, `results_folder_suffix`, `dtb_file` | Same as above (results in `FLAMELETS_DTB_<suffix>/`) |
| `pressure` | Pressure [Pa] |
| `time_step` | $dt$ [s] between `X` and `Y` |
| `include_zerod`, `include_oned_prem`, `include_oned_diff` | Which canonical cases to compute |
| `zerod_params` | `phi_bounds`, `T0_bounds` [K], `n_samples`, `max_sim_time` [s], `solve_mode` (`dt_cfd`: constant $dt$, `dt_cvode`: CVODE internal steps), `cvode_stride` (`dt_cvode` only: one CVODE step out of `cvode_stride` is stored, optional, default 1) |
| `oned_prem_params` | `phi_bounds`, `T0_bounds` [K], `n_samples` |
| `oned_diff_params` | `strain_bounds` [1/s], `T0_bounds` [K], `n_samples`, `width` [m] |
| `augment_dataset` | If `true`, add randomly perturbed states (enthalpy and $Y_k$), following Ding *et al.* |

Cases are sampled in the given bounds with Latin hypercube sampling.

## `dtb_processing.yaml` (learning database)

| Key | Description |
|-----|-------------|
| `database_params.database_type` | `stoch` or `flamelets` |
| `database_params.dtb_folder_suffix` | Suffix of the database folder to read |
| `database_params.dtb_file` | h5 file to read (`solutions.h5` or the multi-dt file) |
| `database_params.database_name` | Name of the output subfolder |
| `database_params.dt_var` | `true` for a multi-dt database ($dt$ is an input) |
| `database_params.fuel`, `database_params.mech_file` | Fuel and mechanism |
| `data_processing.log_transform_X`, `log_transform_Y` | 0: none, 1: log, 2: Box-Cox (with `lambda_bct`, optional, default 0.1) |
| `data_processing.threshold` | Mass fractions are clipped to this value before log transform (to 0 before Box-Cox) |
| `data_processing.T_threshold` | States below this temperature [K] are removed |
| `data_processing.output_omegas` | If `true`, outputs are $Y_k(t+dt)-Y_k(t)$ instead of $Y_k(t+dt)$ |
| `data_processing.with_N_chemistry` | If `false`, N2 is removed from inputs and outputs |
| `data_processing.log_excluded_species` | Species kept without log transform (optional) |
| `data_clustering.clustering_method` | `kmeans` or `progvar` (bounds given to `clusterize_dataset`) |
| `data_clustering.clusterize_on` | Multi-dt only: k-means on `phys` (T, $Y_k$), `dt`, or `all` |
| `data_clustering.nb_clusters` | Number of clusters |
| `train_set_size` | Fraction of data used for training (rest is validation) |

## `networks_params.yaml` (training)

| Key | Description |
|-----|-------------|
| `database_path` | Path to the learning database (`STOCH_DTB_<suffix>/<database_name>`) |
| `model_name_suffix` | Model is written in `MODELS/MODEL_<suffix>/` |
| `new_model_folder` | If `true`, (re)create the model folder |
| `networks_types` | One per cluster: `MLP`, `DeepONet` or `DeepONetShift` |
| `networks_def` | One per cluster: key of its definition in `clusters` |
| `clusters` | Network definitions (see below) |
| `learning.optimizer` | `adam` (default) or `sgd` |
| `learning.loss` | `MSE` (default), `L1` or `Huber` |
| `learning.initial_learning_rate` | Default 0.001 |
| `learning.scheduler_option` | `ExpLR` (default), `ReduceLROnPlateau` or `None` |
| `learning.decay_rate` | Decay factor per epoch for `ExpLR` |
| `learning.batch_size` | Default 2048 |
| `learning.epochs_list` | Number of epochs, one per cluster |
| `learning.val_every` | Validation is computed every `val_every` epochs (default 10) |

An MLP definition contains:

+ `nb_units_in_layers_list`: neurons in each hidden layer
+ `layers_activation_list`: activation of each layer, including output (`relu`, `gelu`, `tanh`, `id`; case-insensitive)
+ `layers_type`: type of each layer, including output (`dense` or `resnet`)

For `DeepONet`, these three keys are dicts with `branch` and `trunk` entries (plus `shift` for `DeepONetShift`), and `n_neurons` is added. The trunk takes $dt$ as input.

# Tests

```
pytest tests/
```

Tests 01 to 04 must run in order: the first one generates `tests/STOCH_DTB_TEST`, reused by the others, and results are compared to reference histograms. Tests 05 to 23 are independent and run on small cases in temporary folders (test 19 runs `mpiexec -n 2` and is skipped without `mpiexec`) (`h2_learning_database.py` builds a small H2 learning database for tests 08, 09, 13, 22 and 23). Each test file starts with a description of what it tests.
