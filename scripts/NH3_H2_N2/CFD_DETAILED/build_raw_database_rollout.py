"""Build a MULTI-STEP rollout database from CFD field snapshots (DATA/output/*.h5).

Same idea as build_raw_database.py (reacts every CFD cell state with Cantera,
same mechanism as the CFD solver -- see build_raw_params_rollout.yaml ->
mech_file), but instead of a single sim.advance(dt) call it keeps advancing
the SAME reactor for nb_steps consecutive steps, recording the state after
each one. This gives real chained ground truth (state at t, t+dt, t+2dt, ...,
t+nb_steps*dt) for training a model to predict autoregressively (predict,
feed the prediction back in, predict again) instead of a single fixed dt.

Output layout, written to the same CFD_DTB_<results_folder_suffix> folder
convention as build_raw_database.py but a distinct file (dtb_file), so it
never collides with a single-step run of the same case:
    ITERATION_XXXXX/X        [n_cells, n_species+4]              -- state at t
    ITERATION_XXXXX/Y        [n_cells, n_species+4]               -- state at t+dt (step 1), kept
                                                                     so this file also works with the
                                                                     existing single-step LearningDatabase
    ITERATION_XXXXX/Y_multi  [n_cells, nb_steps, n_species+4]     -- state at t+dt, t+2dt, ..., t+nb_steps*dt
Both X, Y and Y_multi carry a "cols" attr; Y_multi also carries a "dt" attr
(the single-step dt) and a "steps" attr (1..nb_steps).

This is a standalone script deliberately kept separate from
build_raw_database.py: it must never change behavior for existing
single-step cases.

Run with MPI:
    mpirun -n <nb_procs> python build_raw_database_rollout.py
"""

import os
import gc
import glob

import numpy as np
import h5py
import oyaml as yaml
import cantera as ct

from mpi4py import MPI

CONFIG_FILE = "build_raw_params_rollout.yaml"


def _validate_params(params: dict) -> None:

    required_keys = [
        "cfd_data_dir",
        "mech_file",
        "results_folder_suffix",
        "dtb_file",
        "time_step",
        "T_threshold",
        "nb_steps",
    ]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing required parameter '{key}' in {CONFIG_FILE}")

    if int(params["nb_steps"]) < 1:
        raise ValueError("nb_steps must be >= 1")


def _list_cfd_files(cfd_data_dir: str) -> list:

    files = sorted(glob.glob(os.path.join(cfd_data_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {cfd_data_dir}")
    return files


def _read_cfd_file(file_path: str, species_names: list) -> np.ndarray:
    """Read one CFD snapshot and return a raw state array [T, P, Y_1..Y_ns]."""

    with h5py.File(file_path, "r") as f:
        cell_data = f["STREAM_00/CELL_CENTER_DATA"]

        T = cell_data["TEMPERATURE"][()].astype(np.float64)
        P = cell_data["PRESSURE"][()].astype(np.float64)

        Y = np.empty((T.shape[0], len(species_names)), dtype=np.float64)
        for i_sp, name in enumerate(species_names):
            key = f"MASSFRAC_{name}"
            if key not in cell_data:
                raise KeyError(f"Species '{name}' (dataset '{key}') not found in {file_path}")
            Y[:, i_sp] = cell_data[key][()].astype(np.float64)

    return np.column_stack([T, P, Y])


def _react_state_multi(state: np.ndarray, gas: ct.Solution, dt: float, nb_steps: int) -> np.ndarray:
    """Advance one state [T, P, Y_1..Y_ns] for nb_steps consecutive dt's with a
    single constant-pressure Cantera reactor, returning every intermediate
    state (shape [nb_steps, state.shape[0]]).

    ReactorNet.advance(t) advances to the absolute time t, so calling it with
    k*dt for k = 1..nb_steps on the SAME sim continues the same integration
    instead of restarting a fresh reactor each step.
    """

    T0, P0 = state[0], state[1]
    Y0 = state[2:] / state[2:].sum()

    gas.TPY = T0, P0, Y0

    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])

    out = np.empty((nb_steps, state.shape[0]))
    for k in range(1, nb_steps + 1):
        sim.advance(k * dt)
        out[k - 1, 0] = gas.T
        out[k - 1, 1] = gas.P
        out[k - 1, 2:] = gas.Y

    return out


def build_raw_database_rollout(params: dict, comm: "MPI.Comm") -> None:

    rank = comm.Get_rank()
    size = comm.Get_size()

    _validate_params(params)

    mech_file = params["mech_file"]
    if not mech_file.endswith((".yaml", ".yml")):
        raise ValueError("mech_file must be a Cantera YAML mechanism")

    gas = ct.Solution(mech_file)
    species_names = gas.species_names
    n_species = len(species_names)

    dt = float(params["time_step"])
    T_threshold = float(params["T_threshold"])
    nb_steps = int(params["nb_steps"])

    # NB: must be CFD_DTB_ (not STOCH_DTB_) to match database_type: cfd in
    # dtb_processing.yaml -- LearningDatabase/NN_manager derive this same
    # prefix from database_type when looking the folder back up.
    output_folder = "CFD_DTB_" + params["results_folder_suffix"]
    rank_file_path = os.path.join(output_folder, f".rank_{rank:05d}.h5")

    all_files = _list_cfd_files(params["cfd_data_dir"])
    my_files = all_files[rank::size]

    if rank == 0:
        os.makedirs(output_folder, exist_ok=True)
        print(
            f">> {len(all_files)} CFD files found, dispatched across {size} rank(s), "
            f"nb_steps={nb_steps}",
            flush=True,
        )

    comm.Barrier()

    cols = ["Temperature", "Pressure"] + species_names + ["Prog_var", "HRR"]
    n_cols = n_species + 4

    with h5py.File(rank_file_path, "w") as h5f:
        for file_idx, f in enumerate(my_files):
            print(f"[rank {rank}] reading {os.path.basename(f)} ({file_idx + 1}/{len(my_files)})", flush=True)
            X_raw = _read_cfd_file(f, species_names)
            X_raw = X_raw[X_raw[:, 0] > T_threshold]
            n_local = X_raw.shape[0]
            print(
                f"[rank {rank}] reacting {n_local} states from {os.path.basename(f)} "
                f"with dt={dt:g}s, nb_steps={nb_steps}",
                flush=True,
            )

            X_out = np.empty((n_local, n_cols))
            X_out[:, :2 + n_species] = X_raw
            X_out[:, -2:] = -1.0
            del X_raw

            Y_multi = np.empty((n_local, nb_steps, n_cols))
            for i in range(n_local):
                if rank == 0 and i % 5000 == 0:
                    print(f"[rank 0] {i} / {n_local}", flush=True)
                Y_multi[i, :, :2 + n_species] = _react_state_multi(
                    X_out[i, :2 + n_species], gas, dt, nb_steps
                )
                Y_multi[i, :, -2:] = -1.0

            grp = h5f.create_group(f"ITERATION_{file_idx:05d}")
            dset_X = grp.create_dataset("X", data=X_out)
            dset_Y = grp.create_dataset("Y", data=Y_multi[:, 0, :])
            dset_Ym = grp.create_dataset("Y_multi", data=Y_multi)
            dset_X.attrs["cols"] = cols
            dset_Y.attrs["cols"] = cols
            dset_Ym.attrs["cols"] = cols
            dset_Ym.attrs["dt"] = dt
            dset_Ym.attrs["steps"] = np.arange(1, nb_steps + 1)

            del X_out, Y_multi
            gc.collect()
            print(f"[rank {rank}] wrote {os.path.basename(f)} -> ITERATION_{file_idx:05d}", flush=True)

    comm.Barrier()

    if rank == 0:
        output_path = os.path.join(output_folder, params["dtb_file"])
        print(f">> Merging {size} per-rank file(s) into {output_path}", flush=True)
        # LearningDatabase.get_database_from_h5 expects strictly sequential
        # ITERATION_00000, ITERATION_00001, ... group names (it counts top-
        # level groups and indexes them by range(nb_solutions)) -- NOT the
        # ITERATION_<rank>_<file_idx> compound names used per-rank above, so
        # groups must be renumbered sequentially while merging, not just
        # copied as-is.
        next_idx = 0
        with h5py.File(output_path, "w") as h5_out:
            for r in range(size):
                rp = os.path.join(output_folder, f".rank_{r:05d}.h5")
                with h5py.File(rp, "r") as h5_in:
                    for group_name in sorted(h5_in.keys()):
                        h5_in.copy(group_name, h5_out, name=f"ITERATION_{next_idx:05d}")
                        next_idx += 1
                os.remove(rp)
        print(f">> Raw rollout database written to {output_path} ({next_idx} ITERATION groups)", flush=True)


comm = MPI.COMM_WORLD

with open(CONFIG_FILE, "r") as file:
    data_gen_parameters = yaml.safe_load(file)

build_raw_database_rollout(data_gen_parameters, comm)
