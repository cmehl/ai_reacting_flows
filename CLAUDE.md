# Coding style

In an if/elif chain, only test one question: when conditions are about different things (e.g. `output_omegas`, then `log_transform_Y`), nest them instead of mixing them in the same chain, and comment each branch. Prefer nested ifs when it makes the code clearer, even if it adds lines or indentation.


# Project overview

`ai_reacting_flows` (ARF) builds neural-network surrogates for chemistry integration:
find a network such that `Y_k(t+dt) = ANN(T(t), Y_k(t))`. Package lives in `src/ai_reacting_flows/`
(installed with `pip install -e .`, Python >= 3.11, Cantera 3.0.1, PyTorch, mpi4py, h5py).

The workflow has four steps, each driven by a YAML file read from the **current working directory**:

| Step | Entry point | Input YAML | Output |
|------|-------------|------------|--------|
| 1. Database generation (MPI) | `stochastic_reactors_data_gen.main.generate_stochastic_database(comm)` or `flmts_reactors_data_gen.main.generate_flmts_database(comm)` | `dtb_params.yaml` / `dtb_params_flmts.yaml` | `STOCH_DTB_<suffix>/` (or `FLAMELETS_DTB_<suffix>/`) with `solutions.h5` |
| 1b. Optional multi-dt resampling (MPI) | `databases_processing.generate_var_dt_dtb.GenerateVariable_dt` | `dtb_params.yaml` (`time_step_type`, `time_step_range`, `nb_dt` or `rollout_dt`, `rollout_steps`, `new_file_name`) | new h5 file in the DTB folder |
| 1c. Optional merging of databases (before 1b) | `databases_processing.merge_databases.merge_databases()` | `dtb_merging.yaml` (`input_databases`, `results_folder_suffix`, `dtb_file`, `pv_species`) | `MERGED_DTB_<suffix>/` with merged h5 file, read like `stoch` with `database_type: merged` |
| 2. Learning database processing | `databases_processing.database_processing.LearningDatabase` (threshold, HRR or Omega undersampling (1D, or in 2D bins with a factor per bin), clustering, train/val split) | `dtb_processing.yaml` | `<DTB>/<database_name>/training_data.h5`, `kmeans_model.pkl`, copies of mechanism and yaml |
| 3. Training (PyTorch, one network per cluster) | `ann_model_generation.NN_manager.NN_manager().train_all_clusters()` | `networks_params.yaml` | `MODELS/MODEL_<suffix>/` + `nn_training_<timestamp>.log` |
| 4. Testing (series of 0D ignition/pyrolysis, 1D premixed/diffusion, validation scatter) | `ann_model_generation.NN_testing.NNTesting(params).run_testing()` | `nn_testing.yaml` (read by `ann_model_testing.py`) | `TESTING_<models_folder>/` (plots, `summary_<case type>.yaml`) |

Steps read results of previous steps through folder-name conventions (`STOCH_DTB_<results_folder_suffix>`,
`database_path`, `MODEL_<model_name_suffix>`), so all steps of a case are run from the same folder.
Each step copies its input YAML into its output folder to document the run.

## Source layout (`src/ai_reacting_flows/`)

- `stochastic_reactors_data_gen/`: 0-D stochastic particles method (Wan et al., CNF 2020).
  `particles_cloud.py` (time marching, mixing models CURL / CURL_MODIFIED / CURL_MODIFIED_DD / EMST_SIMPLIFIED / EMST_POPE, MPI distribution, h5 output,
  optional ANN chemistry with `ML_inference_flag`),
  `emst_pope.py` (EMST model of Subramaniam & Pope with particle ages, from Ren et al. emst.f; loops compiled with numba),
  `particle.py` (one particle: reaction with Cantera, progress variable, HRR), `pre_processing.py` (inlet types:
  `cold_premixed`, `burnt_premixed`, `air`, `fuel`, `blank`), `post_processing.py`.
- `flmts_reactors_data_gen/`: database from canonical flames (0D reactors, 1D premixed, 1D diffusion) — `database_flmts.py`.
- `databases_processing/`: `database_processing.py` (`LearningDatabase`), `generate_var_dt_dtb.py` (multi-dt and rollout databases),
  `merge_databases.py` (merging of solution files into `MERGED_DTB_<suffix>`, `database_type: merged`),
  `read_solutions.py` (reading of X, Y, DT of all iterations of a solution file: shared by `LearningDatabase` and multi-dt),
  `processing_parameters.py` (processing parameters written as attributes of `training_data.h5`, read by training and
  testing; YAML fallback for older databases).
- `ann_model_generation/` (PyTorch): `NN_manager.py` (training, optionally on rollouts, with the element projection and with increment weights in the loss; losses MSE/L1/Huber,
  schedulers ExpLR/ReduceLROnPlateau), `NN_models.py` (MLP, PerSpeciesMLP: one MLP per species, DeepONet, DeepONet_shift), `Custom_layers.py`,
  `NN_inference.py` (`NNInference`: ANN chemistry step on a batch of states, shared by `NNTesting` and the stochastic
  reactors), `NN_testing.py` (`NNTesting`: runs the reference/ANN test cases), `NN_testing_metrics.py` (error metrics),
  `NN_testing_plots.py` (plots of the tests).
- `tools/`: `utilities.py` (phi <-> Y_k, element/atomic matrices from Cantera, adiabatic flame, 0D reactor, `PRINT`),
  `transforms.py` (log/Box-Cox transforms, scaling, network output to Y_k, k-means features: shared by
  `LearningDatabase`, `NN_manager` and `NNTesting`), `element_projection.py` (element correction `yk_renormalization: 3`,
  numpy and torch: shared by `NNInference` and `NN_manager`), `loss_weighting.py` (increment weights of the training loss, one per point or per point and species; work in progress), `pca_tools.py`.

## Other folders

- `scripts/<CASE>/`: one folder per case (H2_IGN_KERNEL, H2_IGN_KERNEL_PER_SPECIES, H2_IGN_KERNEL_ROLLOUT, H2_IGN_KERNEL_RESIDENCE_TIME, H2_IGN_KERNEL_MULTI_DT, H2_IGN_KERNEL_WEIGHTED_LOSS, NH3_0D_REACTORS, NH3_PYRO_MULTI_DT, H2_MERGING_EXAMPLE),
  each with the YAML inputs and thin launcher scripts (`generate_stoch_dtb.py`, `dtb_processing.py`,
  `ann_model_learning.py`, `ann_model_testing.py`). Launch from inside the case folder, e.g.
  `mpirun -n X python generate_stoch_dtb.py`. Results are written there (git-ignored).
- `docs/theory.md`: models and algorithms with their equations (EMST_POPE mixing, renewal of the particles, temperature update, element
  conservation, rollout training, element projection in training, loss weighting, simulation error, speed-up benchmark). README.md keeps the workflow, files and inputs, and links to it:
  update both when a model changes.
- `docs/mpi.md`: MPI parallelism of the three generation steps (exchanges of particle states of the stochastic reactors,
  canonical flames, multi-dt): update it when the distribution or the exchanged attributes change.
- `articles/PoF_2023/`: scripts reproducing Mehl & Aubagnac-Karkar, PoF 35, 067115 (2023).
- `notebooks/`: jupytext `.py` (percent format) notebooks; `jupytext --to notebook notebooks/*.py`.
- `data/chemical_mechanisms/`: Cantera mechanisms (use `.yaml`; `.cti` are legacy). Referenced by relative path
  `../../data/chemical_mechanisms/...` from case folders.
- `tests/`: `pytest tests/` (CI: GitHub Actions, Ubuntu + OpenMPI, Python 3.11). Tests are numbered and
  independent: the H2 stochastic database of tests 01-04 is built once per session in a temporary folder by the
  fixtures of `conftest.py` (`h2_stoch_database.py`); `DummyComm.py` replaces MPI;
  results are compared to reference histograms (`H2_*_histograms/`, `H2_TEST_training_data/`) with a
  Wasserstein distance (`misc.py`).

## Notes

- MPI is required even for serial runs (scripts pass `MPI.COMM_WORLD`).
- Stochastic reactors with MPI: each rank keeps its own particles; only the attributes listed in `EXCHANGED_SCALARS` /
  `EXCHANGED_ARRAYS` (top of `particles_cloud.py`) are exchanged. Any new attribute changed during the simulation and
  used by other ranks than 0 must be added there, otherwise the ranks silently diverge.
- Running a generation step **deletes** an existing `STOCH_DTB_<suffix>` folder; `new_model_folder: true` deletes an existing model folder.
- Training uses `cuda:0` if available, else CPU.
- See `README.md` for installation (incl. offline install on cluster with `pip download`).
