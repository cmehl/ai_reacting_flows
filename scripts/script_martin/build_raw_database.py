"""Build a raw ML database from CFD field snapshots (DATA/REDUCED/output/*.h5).

Concatenates (T, P, Y) over all cells of all snapshots, then reacts every
state at T + dt with Cantera (same STEC_A_noAR reduced mechanism as the CFD
solver) to build X (state at t) / Y (state at t+dt). Output is written with
the ITERATION_XXXXX/X, ITERATION_XXXXX/Y HDF5 layout expected by
ai_reacting_flows.databases_processing.database_processing.LearningDatabase
(database_type: "stoch").

Run with MPI:
    mpirun -n <nb_procs> python build_raw_database.py
"""

import os
import glob

import numpy as np
import h5py
import oyaml as yaml
import cantera as ct

from mpi4py import MPI

# Chemin du YAML de config (en dur, comme demande)
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
    # CFD post-processing mass fractions rarely sum exactly to 1; Cantera needs
    # (near-)normalized input.
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


def build_raw_database(params: dict, comm: 'MPI.Comm') -> None:

    rank = comm.Get_rank()
    size = comm.Get_size()

    _validate_params(params)

    mech_file = params["mech_file"]
    if not mech_file.endswith((".yaml", ".yml")):
        raise ValueError("mech_file must be a Cantera YAML mechanism")

    gas = ct.Solution(mech_file)
    species_names = gas.species_names

    dt = float(params["time_step"])
    T_threshold = float(params["T_threshold"])

    output_folder = "STOCH_DTB_" + params["results_folder_suffix"]
    output_path = os.path.join(output_folder, params["dtb_file"])

    # Each rank reads and reacts its own subset of files: no single-rank
    # read + mpi4py scatter (which is capped around 2GB).
    all_files = _list_cfd_files(params["cfd_data_dir"])
    my_files = all_files[rank::size]

    if rank == 0:
        os.makedirs(output_folder, exist_ok=True)
        print(f">> {len(all_files)} CFD files found, dispatched across {size} rank(s)", flush=True)

    comm.Barrier()

    local_states = [_read_cfd_file(f, species_names) for f in my_files]
    for f in my_files:
        print(f"[rank {rank}] read {os.path.basename(f)}", flush=True)

    n_species = len(species_names)

    if local_states:
        X_local = np.concatenate(local_states, axis=0)
    else:
        X_local = np.empty((0, 2 + n_species))

    # Drop non-reacting / near-ambient cells
    X_local = X_local[X_local[:, 0] > T_threshold]

    cols = ["Temperature", "Pressure"] + species_names + ["Prog_var", "HRR"]

    n_local = X_local.shape[0]
    print(f"[rank {rank}] reacting {n_local} states with dt={dt:g}s", flush=True)

    # Prog_var/HRR are not computed here and set to dummy values, matching the
    # precedent in databases_processing.generate_var_dt_dtb.react_multi_dt
    # (process_database() drops these two columns before training anyway).
    X_out = np.empty((n_local, n_species + 4))
    X_out[:, :2 + n_species] = X_local
    X_out[:, -2:] = -1.0

    Y_local = np.empty((n_local, n_species + 4))
    for i in range(n_local):
        if rank == 0 and i % 5000 == 0:
            print(f"[rank 0] {i} / {n_local}", flush=True)
        Y_local[i, :2 + n_species] = _react_state(X_local[i], gas, dt)
        Y_local[i, -2:] = -1.0

    comm.Barrier()

    # Same rank-token write pattern as
    # databases_processing.generate_var_dt_dtb.GenerateVariable_dt: each rank
    # writes its own ITERATION_<rank> group in turn, avoiding both the
    # mpi4py gather size limit and the need for parallel HDF5.
    TAG_WRITE_TOKEN = 42
    if rank == 0:
        with h5py.File(output_path, "w") as h5f:
            grp = h5f.create_group(f"ITERATION_{rank:05d}")
            dset_X = grp.create_dataset("X", data=X_out)
            dset_Y = grp.create_dataset("Y", data=Y_local)
            dset_X.attrs["cols"] = cols
            dset_Y.attrs["cols"] = cols
    else:
        comm.recv(source=rank - 1, tag=TAG_WRITE_TOKEN)
        with h5py.File(output_path, "a") as h5f:
            grp = h5f.create_group(f"ITERATION_{rank:05d}")
            dset_X = grp.create_dataset("X", data=X_out)
            dset_Y = grp.create_dataset("Y", data=Y_local)
            dset_X.attrs["cols"] = cols
            dset_Y.attrs["cols"] = cols

    if rank < size - 1:
        comm.send(True, dest=rank + 1, tag=TAG_WRITE_TOKEN)

    comm.Barrier()

    if rank == 0:
        print(f">> Raw database written to {output_path}", flush=True)


comm = MPI.COMM_WORLD

with open(CONFIG_FILE, "r") as file:
    data_gen_parameters = yaml.safe_load(file)

build_raw_database(data_gen_parameters, comm)
