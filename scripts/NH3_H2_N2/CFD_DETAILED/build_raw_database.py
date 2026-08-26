"""Build a raw ML database from CFD field snapshots (DATA/output/*.h5).

Concatenates (T, P, Y) over all cells of all snapshots, then reacts every
state at T + dt with Cantera (same mechanism as the CFD solver, see
build_raw_params.yaml -> mech_file) to build X (state at t) / Y (state at
t+dt). Output is written with the ITERATION_XXXXX/X, ITERATION_XXXXX/Y
HDF5 layout expected by
ai_reacting_flows.databases_processing.database_processing.LearningDatabase
(database_type: "cfd").

Memory-bounded: each rank processes ONE FILE AT A TIME, writing it to its
own per-rank HDF5 file immediately (no cross-rank write coordination
needed) and freeing the arrays before starting the next file. After all
ranks finish, rank 0 merges every per-rank file into a single solutions.h5.
Size n_ranks to the RAM actually available on the machine you run this on
-- a detailed (~30+ species) mechanism can need several GB per rank for a
single CFD snapshot's cell count.

Run with MPI:
    mpirun -n <nb_procs> python build_raw_database.py
"""

import os
import gc
import glob

import numpy as np
import h5py
import oyaml as yaml
import cantera as ct

from mpi4py import MPI

CONFIG_FILE = "build_raw_params.yaml"


def _validate_params(params: dict) -> None:

    required_keys = [
        "cfd_data_dir",
        "mech_file",
        "results_folder_suffix",
        "dtb_file",
        "time_step",
        "T_threshold",
    ]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing required parameter '{key}' in {CONFIG_FILE}")


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


def _react_state(state: np.ndarray, gas: ct.Solution, dt: float) -> np.ndarray:
    """Advance one state [T, P, Y_1..Y_ns] by dt with a constant-pressure Cantera reactor."""

    T0, P0 = state[0], state[1]
    Y0 = state[2:] / state[2:].sum()

    gas.TPY = T0, P0, Y0

    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    sim.advance(dt)

    new_state = np.empty(state.shape[0])
    new_state[0] = gas.T
    new_state[1] = gas.P
    new_state[2:] = gas.Y

    return new_state


def build_raw_database(params: dict, comm: "MPI.Comm") -> None:

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

    # NB: must be CFD_DTB_ (not STOCH_DTB_) to match database_type: cfd in
    # dtb_processing.yaml -- LearningDatabase/NN_manager derive this same
    # prefix from database_type when looking the folder back up.
    output_folder = "CFD_DTB_" + params["results_folder_suffix"]
    rank_file_path = os.path.join(output_folder, f".rank_{rank:05d}.h5")

    all_files = _list_cfd_files(params["cfd_data_dir"])
    my_files = all_files[rank::size]

    if rank == 0:
        os.makedirs(output_folder, exist_ok=True)
        print(f">> {len(all_files)} CFD files found, dispatched across {size} rank(s)", flush=True)

    comm.Barrier()

    cols = ["Temperature", "Pressure"] + species_names + ["Prog_var", "HRR"]

    with h5py.File(rank_file_path, "w") as h5f:
        for file_idx, f in enumerate(my_files):
            print(f"[rank {rank}] reading {os.path.basename(f)} ({file_idx + 1}/{len(my_files)})", flush=True)
            X_raw = _read_cfd_file(f, species_names)
            X_raw = X_raw[X_raw[:, 0] > T_threshold]
            n_local = X_raw.shape[0]
            print(f"[rank {rank}] reacting {n_local} states from {os.path.basename(f)} with dt={dt:g}s", flush=True)

            X_out = np.empty((n_local, n_species + 4))
            X_out[:, :2 + n_species] = X_raw
            X_out[:, -2:] = -1.0
            del X_raw

            Y_local = np.empty((n_local, n_species + 4))
            for i in range(n_local):
                if rank == 0 and i % 5000 == 0:
                    print(f"[rank 0] {i} / {n_local}", flush=True)
                Y_local[i, :2 + n_species] = _react_state(X_out[i, :2 + n_species], gas, dt)
                Y_local[i, -2:] = -1.0

            grp = h5f.create_group(f"ITERATION_{file_idx:05d}")
            dset_X = grp.create_dataset("X", data=X_out)
            dset_Y = grp.create_dataset("Y", data=Y_local)
            dset_X.attrs["cols"] = cols
            dset_Y.attrs["cols"] = cols

            del X_out, Y_local
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
        print(f">> Raw database written to {output_path} ({next_idx} ITERATION groups)", flush=True)


comm = MPI.COMM_WORLD

with open(CONFIG_FILE, "r") as file:
    data_gen_parameters = yaml.safe_load(file)

build_raw_database(data_gen_parameters, comm)
