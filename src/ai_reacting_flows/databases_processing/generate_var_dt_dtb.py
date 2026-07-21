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
    # Y = np.empty(X.shape)
    # Stack features and responses along the third axis: state[..., 0] / state[..., 1]
    np.random.shuffle(X) # DAK: check why we shuffle X
    # state_list = np.dstack((X, Y))

    state_list = X.copy()

    # Temperature threshold (same quantity as database processing; keep explicit).
    # T_thresh = params["T_threshold"]

    # Time-step list definition.
    # dt_simu = params["time_step"]
    time_step_type = params["time_step_type"]

    if time_step_type == "set":
        # Ensure we are working on a copy (and a Python list).
        dt_list = list(params["time_step_range"])
        # if dt_simu not in dt_list: #Not OK if time_step is a dict (multi dt), so we let the user make sure to manually enter the dt_simu he wants
        #     dt_list.append(dt_simu)
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
    dt_array = []

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
        
        dt_array.append(dt_list)

        new_states = react_multi_dt(state, gas, dt_list)
        if new_states is not None:
            all_new_states.append(new_states)


    # all_new_states = np.concatenate(all_new_states)
    # X_new, Y_new = np.dsplit(all_new_states, 2)
    # X_new, Y_new = np.squeeze(X_new), np.squeeze(Y_new)
    # # Remove last column (dummy variable) from Y.
    # Y_new = Y_new[:, :-1]
    # #

    Y_new = np.stack(all_new_states)
    dt_array = np.stack(dt_array)


    print(f"Processor {rank} has built array all_new_states. \n")

    if dtb_type not in {"stoch", "flamelets"}:
        raise ValueError("dtb_type must be 'stoch' or 'flamelets'")
    
    output_path = f"{stoch_results_folder:s}/{params['new_file_name']:s}"

    comm.Barrier()

    # Gathering Y_new using mpi gather and then writing everything in one block is not possible because there is a limit of 2 GB for the gather
    # The solution is to write a group in h5 file per rank and write in the file rank after rank (not possible to write at the same time)
    # This is done below using send/redv command, ensuring that each rank writes when the previosu has finished. 

    token = None
    TAG_WRITE_TOKEN = 42

    if rank == 0:
        # Rank 0 creates the file fresh (truncating any old one), then writes first.
        with h5py.File(output_path, "w") as f:
            grp = f.create_group(f"ITERATION_{rank:05d}")   # ITERATION naming instead of RANK to make it readable straight away in database processing
            dset_X = grp.create_dataset("X", data=state_list)
            dset_Y = grp.create_dataset("Y", data=Y_new)
            dset_DT = grp.create_dataset("DT", data=dt_array)
            dset_X.attrs["cols"] = col_names_X
            dset_Y.attrs["cols"] = col_names_Y
        print(f"Processor {rank} has written its group to {output_path}. \n", flush=True)
    else:
        # Wait for the previous rank to signal it has finished writing.
        token = comm.recv(source=rank - 1, tag=TAG_WRITE_TOKEN)

        with h5py.File(output_path, "a") as f:
            grp = f.create_group(f"ITERATION_{rank:05d}")
            dset_X = grp.create_dataset("X", data=state_list)
            dset_Y = grp.create_dataset("Y", data=Y_new)
            dset_DT = grp.create_dataset("DT", data=dt_array)
            dset_X.attrs["cols"] = col_names_X
            dset_Y.attrs["cols"] = col_names_Y
        print(f"Processor {rank} has written its group to {output_path}. \n", flush=True)

    # Pass the token to the next rank (if any).
    if rank < size - 1:
        comm.send(True, dest=rank + 1, tag=TAG_WRITE_TOKEN)

    comm.Barrier()

    if rank == 0:
        print(f"All {size} processors have written their groups to: {output_path}\n", flush=True)

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


def react_multi_dt(state: np.ndarray, gas: ct.Solution, dt_list: np.ndarray):

    """React one state for multiple time steps and return the resulting states.

    The input state is advanced independently for each time step in
    `dt_list`, i.e. every reaction starts fresh from `state` (not
    cumulatively from the previous result).

    Parameters
    ----------
    state : ndarray
        1-D input state array of length n_features, laid out as
        [T, P, Y_1, ..., Y_nsp, progvar, HRR]. Only T, P, and Y are used
        as input (progvar and HRR are ignored on input and overwritten
        with dummy values on output).
    gas : ct.Solution
        Cantera solution used for reactor integration.
    dt_list : ndarray
        1-D array of time steps (in seconds) for which the reactor will
        be independently advanced from the initial state.

    Returns
    -------
    new_states : ndarray
        Array of shape (n_features, len(dt_list)). Column i contains the
        state reached after advancing from `state` by `dt_list[i]`:
        rows 0 and 1 are T and P, rows 2:-2 are the mass fractions Y,
        and the last two rows (progvar, HRR) are set to -1.0 dummy
        values (not computed here).
    """

    # Unsolved pb: if Yk from NN is inputed here;
    # it may become negative and mass is lost (output from CVODE is always positive)

    nb_dt = len(dt_list)

    new_states = np.empty((state.shape[0], nb_dt))

    # Initial values are current particle state.
    T0 = state[0]
    P0 = state[1]
    Y0 = state[2:-2]

    # Advancing to dts.
    for i_dt, dt in enumerate(dt_list):

        gas.TPY = T0, P0, Y0

        # Constant-pressure reactor.
        r = ct.IdealGasConstPressureReactor(gas)
        sim = ct.ReactorNet([r])

        # Advancing simulation by dt
        sim.advance(dt)

        # Updated state.
        new_states[0, i_dt] = gas.T
        new_states[1, i_dt] = gas.P
        new_states[2:-2, i_dt] = gas.Y
        new_states[-2, i_dt] = -1.0  # dummy value for progvar
        new_states[-1, i_dt] = -1.0  # dummy value for HRR

    return new_states

