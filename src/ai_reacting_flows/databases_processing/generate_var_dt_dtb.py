"""Generate variable-∆t learning database from stochastic or flamelet HDF5 data.

This script reacts each state for multiple time steps using Cantera and writes
an augmented database including the chosen ∆t values.

The main entry point is `GenerateVariable_dt`, typically called from a driver
script with MPI (see `scripts/NH3_0D_REACTORS/generate_multi_dt.py`).
"""

import os

import numpy as np
import h5py
import pandas as pd
import cantera as ct

from mpi4py import MPI


def _validate_params(params: dict) -> None:
    """Basic validation of required parameters.

    This avoids hard-to-debug KeyError deep in the computation.
    """

    required_keys = [
        "results_folder_suffix",
        "dtb_file",
        "mech_file",
        "T_threshold",
        "time_step_type",
        "time_step",
        "time_step_range",
        "new_file_name",
    ]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Missing required parameter '{key}' in data-gen configuration")

    if params["time_step_type"] not in {"set", "random"}:
        raise ValueError("time_step_type must be either 'set' or 'random'")


def _build_dtb_folder(dtb_type: str, results_folder_suffix: str) -> str:
    """Return absolute path to the database folder for the given type."""

    if dtb_type == "stoch":
        dtb_prefix = "STOCH"
    elif dtb_type == "flamelets":
        dtb_prefix = "FLAMELETS"
    else:
        raise ValueError("dtb_type must be 'stoch' or 'flamelets'")

    run_folder = os.getcwd()
    return f"{run_folder:s}/{dtb_prefix}_DTB_" + results_folder_suffix


def GenerateVariable_dt(dtb_type: str, params: dict, comm: "MPI.Comm") -> str:
    """Generate variable-dt database.

    Parameters
    ----------
    dtb_type : {"stoch", "flamelets"}
        Type of input database.
    params : dict
        Dictionary of parameters, typically loaded from `dtb_params*.yaml`.
        Must contain at least the keys validated in `_validate_params`.
    comm : MPI.Comm
        MPI communicator used to split and gather work.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(1234)

    _validate_params(params)

    # Opening input h5 file
    stoch_results_folder = _build_dtb_folder(dtb_type, params["results_folder_suffix"])
    file_h5 = f"{stoch_results_folder:s}/{params['dtb_file'].split('.')[0]:s}.h5"

    if not os.path.isfile(file_h5):
        raise FileNotFoundError(f"Input database file not found: {file_h5}")

    if dtb_type == "stoch":
        X, col_names_X, col_names_Y = read_database_stoch(file_h5)
    elif dtb_type == "flamelets":
        X, col_names_X, col_names_Y = read_database_flmts(file_h5)

    if rank == 0:
        print(f"X columns: {col_names_X} \n")

    # Build state array with empty target column, later filled by `react_multi_dt`.
    Y = np.empty(X.shape)
    # Stack features and responses along the third axis: state[..., 0] / state[..., 1]
    state_list = np.dstack((X, Y))

    # Temperature threshold (same quantity as database processing; keep explicit).
    T_thresh = params["T_threshold"]

    # Time-step list definition.
    dt_simu = params["time_step"]
    time_step_type = params["time_step_type"]

    if time_step_type == "set":
        # Ensure we are working on a copy (and a Python list).
        dt_list = list(params["time_step_range"])
        if dt_simu not in dt_list:
            dt_list.append(dt_simu)
    else:  # "random"
        dt_range = params["time_step_range"]
        nb_dt = params["nb_dt"]

        if len(dt_range) != 2:
            raise ValueError("time_step_range must be a length-2 iterable for random sampling")

        dt_min, dt_max = float(dt_range[0]), float(dt_range[1])
        if dt_min <= 0 or dt_max <= 0:
            raise ValueError("time_step_range values must be strictly positive")

        fact_min, fact_max = 1.0, 1.0  # Extend the learned dt, kept for future flexibility.

    # Split across MPI ranks.
    state_list = np.array_split(state_list, size)
    state_list = comm.scatter(state_list, root=0)

    gas = ct.Solution(params["mech_file"])
    tot_0 = len(state_list)

    count = 0
    all_new_states = []

    # React all states.
    for state in state_list:
        count += 1
        if rank == 0 and count % 5000 == 0:
            print(f"Operation (proc 0) : {count} / {int(tot_0)}")

        if time_step_type == "random":
            # Sample dt using log-uniform distribution in [fact_min*dt_min, fact_max*dt_max].
            dt_list = np.random.uniform(
                low=np.log(fact_min * dt_min),
                high=np.log(fact_max * dt_max),
                size=(nb_dt),
            )
            dt_list = np.exp(dt_list)
            dt_list.sort()

        new_states = react_multi_dt(state, gas, T_thresh, dt_list)
        if new_states is not None:
            all_new_states.append(new_states)

    if not all_new_states:
        # Nothing above temperature threshold for this rank.
        X_new = np.empty((0,) + state_list[0].shape[:-1]) if len(state_list) > 0 else np.empty((0, 0))
        Y_new = np.empty_like(X_new)
    else:
        all_new_states = np.concatenate(all_new_states)
        X_new, Y_new = np.dsplit(all_new_states, 2)
        X_new, Y_new = np.squeeze(X_new), np.squeeze(Y_new)
        # Remove last column (dummy variable) from Y.
        Y_new = Y_new[:, :-1]

    print(f"Processor {rank} has built array all_new_states. \n")

    comm.Barrier()

    # Gather all X_new/Y_new arrays from every rank onto rank 0.
    all_X_new = comm.gather(X_new, root=0)
    all_Y_new = comm.gather(Y_new, root=0)

    if rank == 0:
        # Concatenate per-rank arrays into one global array.
        X_new_full = np.concatenate(all_X_new, axis=0) if all_X_new else np.empty((0, X_new.shape[1]))
        Y_new_full = np.concatenate(all_Y_new, axis=0) if all_Y_new else np.empty((0, Y_new.shape[1]))

        # For multi-dt stochastic all data is written by default in "ITERATION_00000" as it is
        # too complex to keep the per-iteration structure (and useless when creating processed DB).
        if dtb_type == "stoch":
            group_name = "ITERATION_00000"
        elif dtb_type == "flamelets":
            group_name = "FLAMELETS"
        else:
            raise ValueError("dtb_type must be 'stoch' or 'flamelets'")

        print("WRITING", rank)
        output_path = f"{stoch_results_folder:s}/{params['new_file_name']:s}"
        with h5py.File(output_path, "w") as f:
            grp = f.create_group(group_name)
            dset_X = grp.create_dataset("X", data=X_new_full)
            dset_Y = grp.create_dataset("Y", data=Y_new_full)
            dset_X.attrs["cols"] = np.append(col_names_X, "dt")
            dset_Y.attrs["cols"] = col_names_Y

        print(f"Processor {rank} has written output file at: {output_path}. \n")

    comm.Barrier()

    return "Done"


#TODO : mutualize these 2 functions with other scripts (e.g. data processing) by adding them to utilities

def read_database_stoch(file_h5: str):

    """Read stochastic database from HDF5 into a NumPy array.

    Returns
    -------
    X : ndarray
        Concatenated feature matrix.
    col_names_X : ndarray
        Column names for X as stored in HDF5 attributes.
    col_names_Y : ndarray
        Column names for Y (returned for consistency, although Y is not read).
    """

    h5file_r = h5py.File(file_h5, "r")

    col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"]
    col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"]

    # Loop on solutions
    list_df_X = []

    nb_solutions = len(h5file_r.keys())

    for i in range(nb_solutions):
        data_X = h5file_r.get(f"ITERATION_{i:05d}/X")[()]
        list_df_X.append(pd.DataFrame(data=data_X, columns=col_names_X))

    h5file_r.close()

    X = pd.concat(list_df_X, ignore_index=True).to_numpy().copy()

    return X, col_names_X, col_names_Y


def read_database_flmts(file_h5: str):

    """Read flamelet database from HDF5 into a NumPy array.

    Returns
    -------
    X : ndarray
        Feature matrix.
    col_names_X : ndarray
        Column names for X as stored in HDF5 attributes.
    col_names_Y : ndarray
        Column names for Y (returned for consistency, although Y is not read).
    """

    h5file_r = h5py.File(file_h5, "r")

    col_names_X = h5file_r["FLAMELETS/X"].attrs["cols"]
    col_names_Y = h5file_r["FLAMELETS/Y"].attrs["cols"]

    data_X = h5file_r.get("FLAMELETS/X")[()]
    X = pd.DataFrame(data=data_X, columns=col_names_X).to_numpy().copy()

    h5file_r.close()

    return X, col_names_X, col_names_Y


def react_multi_dt(state: np.ndarray, gas: ct.Solution, T_thresh: float, dt_list: np.ndarray):

    """React one state for multiple time steps and return computed states.

    Parameters
    ----------
    state : ndarray
        Input state array of shape (n_features, 2) where column 0 contains
        the current values and column 1 will be filled with the reacted ones.
    gas : ct.Solution
        Cantera solution used for reactor integration.
    T_thresh : float
        Temperature threshold; states with `T < T_thresh` are ignored.
    dt_list : ndarray
        Time steps for which the reactor will be advanced.

    Returns
    -------
    new_states : ndarray or None
        Stacked states (one per dt). Returns None if the initial temperature
        is below `T_thresh`.
    """

    # Unsolved pb: if Yk from NN is inputed here;
    # it may become negative and mass is lost (output from CVODE is always positive)

    time_step = np.array([[0, 0]])
    state = np.append(state, time_step, axis=0)
    new_states = np.empty_like([state])

    if state[0,0] > T_thresh:
        # Initial values are current particle state.
        T0 = state[0, 0]
        P0 = state[1, 0]
        Y0 = state[2:-3, 0]

        # Advancing to dts.
        for dt in dt_list:
            gas.TPY = T0, P0, Y0

            # Constant-pressure reactor.
            r = ct.IdealGasConstPressureReactor(gas)
            sim = ct.ReactorNet([r])

            # Advancing simulation by dt
            state[-1, 0] = dt
            sim.advance(dt)

            # Updated state.
            y = np.empty(len(Y0) + 4)  # T, p, c, HRR
            y[0] = gas.T
            y[1] = gas.P
            y[2:-2] = gas.Y
            y[-2] = -1.0  # dummy value for progvar
            y[-1] = -1.0  # dummy value for HRR

            state[:, 1] = np.append(y, [0])
            new_states = np.append(new_states, np.array([state]), axis=0)

        # First was empty and only used to enable append in loop.
        new_states = new_states[1:,]

        return new_states

    return None
