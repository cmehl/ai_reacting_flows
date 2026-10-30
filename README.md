[![DOI](https://zenodo.org/badge/531591043.svg)](https://zenodo.org/badge/latestdoi/531591043)

AI FOR REACTING FLOWS (ARF)
===========================

The project *ai_for_reacting_flows* is a tool enabling to replace complex chemical kinetics calculations in a CFD code by a machine learning model. The idea is that typical exact solvers used to resolve such problems (such as [CVODE](https://computing.llnl.gov/projects/sundials/cvode)) are very expensive. Using fitted models, such as Artificial Neural Networks (ANN), enables to speed-up calculations by a significant factor. 

This package features the following functionalities:

+ A module to build databases from stochastic reactors or canonical flames (0-D reactors, 1-D premixed and diffusion flames)

+ A module to generate machine learning models to advance the chemical state of a mixture

The models and algorithms (mixing model, element conservation, error metrics) are detailed in [docs/theory.md](docs/theory.md), and the MPI parallelism in [docs/mpi.md](docs/mpi.md).


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
python ann_model_testing.py                # 4. testing on simple cases  (nn_testing.yaml)
```

## 1. Database generation

+ **Stochastic reactors**: 0-D stochastic particles following [Wan *et al.*](https://www.sciencedirect.com/science/article/pii/S0010218020302170). Each particle is a chemical state; particles mix (CURL, CURL_MODIFIED, CURL_MODIFIED_DD (work in progress), EMST_SIMPLIFIED or EMST_POPE models, see [theory](docs/theory.md#emst-mixing-model-emst_pope)) to mimic diffusion.
+ **Canonical flames**: 0-D reactors, 1-D premixed and 1-D diffusion flames, launched with `generate_0D_dtb.py` and `dtb_params_flmts.yaml`.
+ **Multiple time steps** (optional): `generate_multi_dt.py` re-integrates the database states with several $dt$, so that $dt$ becomes a network input.

**Warning:** an existing `STOCH_DTB_<suffix>` folder is deleted when relaunching generation.

With MPI, the chemistry and the derived variables (progress variable, which needs an equilibrium, mixture fraction, equivalence ratio, HRR) are computed by each rank on its share of the particles, dealt in turn (round-robin: rank $r$ of $N$ takes particles $r$, $r+N$, $r+2N$, ...), so that the particles of an inlet (e.g. the hot ones, the only ones reacting at first) are spread over all ranks; mixing and writing are done by rank 0. Each rank keeps its own copy of the particles, and only the attributes that change are exchanged, as arrays of floats (`Allgatherv` after each distributed step, `Bcast` after mixing). The exchanges are detailed in [docs/mpi.md](docs/mpi.md).

**Warning:** a particle attribute that starts changing during the simulation (e.g. in a new mixing model) must be added to `EXCHANGED_SCALARS` or `EXCHANGED_ARRAYS` at the top of `particles_cloud.py`. Otherwise the other ranks keep its old value, without any error (test 19, 1 against 2 ranks, only detects it if the attribute changes what is written in `solutions.h5`). Attributes used only by rank 0 (e.g. `Le_k` of `CURL_MODIFIED_DD`, used in mixing) need not be exchanged.

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

+ `all_states`: state of all active particles: `Temperature` [K], `Pressure` [Pa], species mass fractions $Y_k$, `Mix_frac`, `Equiv_ratio`, `Prog_var`, `HRR` [W/m³], `Time` [s], `Particle_number`, `Inlet_number`, atomic mass fractions `Y_C`, `Y_H`, `Y_O`, `Y_N`, `Mass`, `EMST_age` (age property of `EMST_POPE`, NaN for other mixing models).
+ `X` and `Y` (if `build_ml_dtb: true`): states before and after the chemistry step $dt$ (`Temperature`, `Pressure`, $Y_k$, `Prog_var`, `HRR`). These are the ML input/output pairs.

Column names are stored in the `cols` attribute of each dataset. For canonical flames, the file has a single group `ITERATION_00000` with `X` and `Y`, and the simulated cases are listed in `sim_train_*.csv` / `sim_test_*.csv`. The multi-dt file has one group per MPI rank with `X`, `Y` (one set of outputs per $dt$) and `DT` (the $dt$ values [s]).

Post-processing of a stochastic database is shown in `post_processing_stoch_reac.py`.

The `EMST_POPE` mixing model (Subramaniam & Pope, with an age property of the particles, written as `EMST_age`) is described in [theory: EMST mixing model](docs/theory.md#emst-mixing-model-emst_pope).

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
│                                         (PerSpeciesMLP: one group species_<i> per output, with the layers of its MLP)
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
+ 1-D counterflow diffusion flame reaction rates
+ the validation dataset (scatter plots)

The script `ann_model_testing.py` reads `nn_testing.yaml` and runs the selected series of cases (`NNTesting.run_testing`): for each case type, all combinations of the parameters given as lists (e.g. several `phi` and `T0`). Outputs are written in `TESTING_<models_folder>/`:

```
TESTING_<models_folder>/
├── nn_testing.yaml                       copy of the inputs
├── <case type>_<parameters>/             plots of each case (if plots: true)
├── summary_<case type>.yaml              errors of each case, mean and std over the series
├── simulation_errors_<case type>.csv     one row per case: parameters and errors
├── simulation_errors_<case type>_by_dt.csv   E(dt), std and number of simulations per dt, mean E_<variable> and speed-up
├── simulation_errors_<case type>_by_dt.png   E(dt) with std bars, and box plot of E_i per dt
├── simulation_errors_<case type>_by_variable.png   mean E_<variable> of T and each network species, per dt
├── simulation_errors_<case type>_by_cluster.csv   errors per cluster and dt
├── speedup_benchmark.csv                 speed-up of the ANN on batches of states
├── plots_validation/                     scatter plots of validation errors, and a priori check of the hybrid
│                                         criteria (hybrid_e_sum.png, hybrid_e_elements.png)
└── nn_testing_<timestamp>.log
```

Errors: 0-D cases give the simulation error $\mathcal{E}_i$ (`E_sim`), the error on the final temperature [%], and for ignition the ignition delays (time at which $T$ exceeds $T_0 + 200$ K) and their error [%]; 1-D flames give an a priori simulation error $\mathcal{E}_i$ (`E_sim`) and the L2 error of the reaction rates of `spec_to_plot`. All cases also give the numbers of ANN and CVODE calls, and the speed-up `speedup` = `time_cvode` / `time_ann`, ratio of the CPU times [s] of the chemistry steps (CVODE integration; ANN clustering, network and CVODE fallback of the hybrid mode; the progress variable computation, which needs an equilibrium, and the diagnostics are not counted). As in a CFD code, CVODE is restarted from a new reactor at each time step (also in 0-D). The ANN advances one state per call, without batching, after an untimed warm-up call (initialization of the networks and k-means), so this speed-up is a lower bound of what a CFD code with batched calls would get: see the benchmark below. 0-D cases also give the drifts after equilibrium: the ANN alone is continued for `stability_factor` times the number of steps, and `T_drift` [K] (from the ANN temperature at the end of the simulation) and `sumYk_drift` (from 1) are the maximum deviations over these steps. A network that does not stay at equilibrium would make burnt gases drift in a CFD code.

`E_sim` is the normalized simulation error of Mehl & Aubagnac (eq. 6), per case, per `dt`, per variable and per cluster ([theory: simulation error](docs/theory.md#simulation-error)). Its tables and plots are listed above. Each 0-D simulation stops when the CVODE temperature is within `T_eq_tolerance` of the equilibrium (HP) temperature of the initial mixture, and the ANN is run for the same number of steps; `max_simulation_time` only caps the simulations that do not reach equilibrium (`equilibrium_reached: false`). 1-D errors are a priori (one step $dt$ from each grid point in `x_window`).

`speedup_benchmark.csv` gives, for each batch size, the ANN time [s], the ANN and CVODE times per state [s] and the speed-up ([theory: speed-up benchmark](docs/theory.md#speed-up-benchmark)).

Single cases can also be run from Python (`test_0D_ignition`, `test_0D_pyrolysis`, `test_1D_premixed`, `test_1D_diffusion`), as in `notebooks/ann_model_testing.py`.

`NNTesting` (`NN_testing.py`) runs the reference and ANN cases; the error metrics are in `NN_testing_metrics.py` and the plots in `NN_testing_plots.py`. The ANN chemistry step itself (clustering, transforms, networks) is in `NN_inference.py` (`NNInference`), shared with the stochastic reactors. The new temperature follows from enthalpy conservation ([theory: temperature update](docs/theory.md#temperature-update-of-the-ann-chemistry-step)), and the predicted mass fractions are corrected to conserve the elements (`yk_renormalization`, [theory: element conservation](docs/theory.md#element-conservation-of-the-ann-mass-fractions-yk_renormalization)).

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
| `mixing_model` | `CURL`, `CURL_MODIFIED`, `CURL_MODIFIED_DD` (work in progress), `EMST_SIMPLIFIED` (minimum spanning tree on $Y_k$, each edge mixed in turn with the coefficient $\min(dt/\tau, 0.5)$) or `EMST_POPE` (model of Subramaniam & Pope, see [theory](docs/theory.md#emst-mixing-model-emst_pope)) |
| `mixing_time` | Mixing time scale $\tau$ [s] (for `EMST_POPE`, the variance function decays as $\exp(-t/\tau)$) |
| `read_mixing` | If `true`, read mixing pairs from `pairs_list.pkl` in the case folder |
| `T_threshold` | Chemistry is not computed below this temperature [K] |
| `pv_species` | Species defining the progress variable $c = (Y_c - Y_{c,u})/(Y_{c,eq} - Y_{c,u})$, with $Y_c$ the sum of their mass fractions, $u$ the unburnt mixture and $eq$ the equilibrium (e.g. `[H2O]`) |
| `fuel` | List of fuels (e.g. `[H2]`) |
| `calc_mean_traj` | If `true`, write mean trajectories of each inlet |
| `ML_inference_flag` | If `true`, chemistry is computed with the ANN instead of CVODE (optional keys below are then required) |
| `ML_model_folder` | Model folder in `MODELS/` of the case folder (e.g. `MODEL_H2_POF_2023`). Its mechanism and `pv_species` must be those of this file |
| `ML_yk_renormalization` | Correction of the ANN mass fractions to conserve the elements, as `yk_renormalization` of `nn_testing.yaml` |
| `ML_hybrid_ann_cvode` | Hybrid ANN/CVODE model, as `hybrid_ann_cvode` of `nn_testing.yaml`: particles whose ANN prediction has a conservation error larger than `ML_hybrid_ann_cvode_tol` are advanced with CVODE |
| `ML_hybrid_ann_cvode_tol` | Tolerance on the conservation error [-] (e.g. `2.0e-04`) |

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
| `database_params.dt_var` | `true` for a multi-dt database ($dt$ is an input, always as $\log(dt)$, whatever `log_transform_X`) |
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
| `networks_types` | One per cluster: `MLP`, `PerSpeciesMLP` (one MLP per species, each predicting the output of its species from the whole input state, all with the definition of the cluster), `DeepONet` or `DeepONetShift` |
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

For `PerSpeciesMLP`, this definition is that of the network of each species (one output), with dense layers only. The networks of all species are computed together (weights of a layer stacked for all species, one batched product per layer), as in the inference of NNICE, which is much faster than one network after the other. Example: `scripts/H2_IGN_KERNEL_PER_SPECIES`.

For `DeepONet`, these three keys are dicts with `branch` and `trunk` entries (plus `shift` for `DeepONetShift`), and `n_neurons` is added. The trunk takes $dt$ as input.

## `nn_testing.yaml` (testing)

| Key | Description |
|-----|-------------|
| `models_folder` | Model folder in `MODELS/` (e.g. `MODEL_H2_POF_2023`) |
| `spec_to_plot` | Species plotted, and on which 1-D errors are computed |
| `yk_renormalization` | Correction of the ANN mass fractions to conserve the elements: `0` none, `1` one species per element, `2` Wan *et al.* (2020), `3` Wan *et al.* extended to O (see [theory](docs/theory.md#element-conservation-of-the-ann-mass-fractions-yk_renormalization)) |
| `hybrid_ann_cvode` | Hybrid ANN/CVODE model of Mehl & Aubagnac-Karkar (2023): CVODE is used instead of the ANN when the conservation error of its prediction exceeds `hybrid_ann_cvode_tol`. `0`: never. `1`: error on the sum of mass fractions $\|\sum_k Y_k - 1\|$. `2`: largest relative error on the element mass fractions (see [theory](docs/theory.md#hybrid-anncvode-model-hybrid_ann_cvode)) |
| `hybrid_ann_cvode_tol` | Tolerance on the conservation error [-] (in the paper, about `2.0e-04` for `1` and `2.0e-03` for `2`) |
| `device` | Optional, device of the networks: `cpu` (as in a first CFD use) or `cuda`; absent or `null`: `cuda` if available, else `cpu` |
| `torch_threads` | Optional, number of CPU threads of torch: 1 as one MPI rank per core in a CFD code (small batches are also faster); absent or `null`: torch default (one per core) |
| `0D_ignition` | Series of 0-D ignitions of fuel/air: `phi`, `T0` [K], `pressure` [Pa], `dt` [s], `T_eq_tolerance` [K] (CVODE and ANN stop when the CVODE temperature is this close to equilibrium), `max_simulation_time` [s] (end time if equilibrium is not reached), `stability_factor` [-] (ANN continued after equilibrium for this times the number of steps, 0 to skip) |
| `0D_pyrolysis` | Series of 0-D pyrolysis: `T0` [K], `pressure` [Pa], `dt` [s], `T_eq_tolerance` [K], `max_simulation_time` [s], `stability_factor` [-], `diluent` (species or `null`), `X_diluent` [-] |
| `1D_premixed` | Series of 1-D premixed flames: `phi`, `T0` [K], `pressure` [Pa], `dt` [s], `T_threshold` [K] (ANN not applied below), `x_window` |
| `1D_diffusion` | Series of 1-D counterflow flames of pure fuel against air: `strain` [1/s], `T_ox` [K], `T_fuel` [K], `pressure` [Pa], `dt` [s], `width` [m], `T_threshold` [K], `x_window` |
| `validation_set` | `run`, `max_points_scatter` (maximum number of points plotted per cluster). Also checks a priori the criteria of the hybrid ANN/CVODE model on the validation states: $e_\Sigma$ and $e_e$ of one ANN step against its mean normalized error (see [theory](docs/theory.md#hybrid-anncvode-model-hybrid_ann_cvode)) |
| `speedup_benchmark` | `run`, `batch_sizes` (numbers of states per network call), `dt` [s] (training time step for constant time step models), `pressure` [Pa], `nb_states_cvode` |

Each series section has `run` (run the series or not) and `plots` (plots of each case). Parameters given as lists are swept (all combinations are run), others are the same for all cases. `x_window` is `[x_min, x_max]` [m] for the 1-D errors and plots, or `null` for the whole flame. The 1-D diffusion flame has the setup of the flamelets database (mixture-averaged transport). `conditions_file` (0-D ignition, 1-D premixed and diffusion) runs the test simulations set aside by a flamelets database (`FLAMELETS_DTB_<suffix>/sim_test_0D.csv`, `sim_test_1D_prem.csv`, `sim_test_1D_diff.csv`): each row gives one simulation (`Phi`, `T0` → `phi`, `T0`; for diffusion flames `Strain`, `T0` → `strain`, `T_ox`), run with every combination of the other parameters of the section (e.g. each `dt`); `null` to use the parameters of the section.

# Tests

```
pytest tests/
```

Tests 01 to 04 check a small H2 stochastic reactors database, its resampling, its multi-dt version and its processing against reference histograms; the database is built once per session in a temporary folder by the fixtures of `conftest.py` (`h2_stoch_database.py`). All tests are independent: they can run in any order, alone, or in several sessions at the same time. Tests 05 to 23 run on small cases in temporary folders (test 19 runs `mpiexec -n 2` and is skipped without `mpiexec`) (`h2_learning_database.py` builds a small H2 learning database for tests 08, 09, 13, 22 and 23). Each test file starts with a description of what it tests.
