"""Reading of a solution file: X, Y (and DT) of all its groups ITERATION_<i>, concatenated along the points.

Layouts of the groups:
    - stochastic reactors: one group per time step (X, Y: (n, n_cols));
    - canonical flames: one group (X, Y: (n, n_cols));
    - merged database (merge_databases.py): groups of the input databases, one after the other;
    - multi-dt database (generate_var_dt_dtb.py): one group per MPI rank (X: (n, n_cols), Y: (n, n_cols, n_dt)
      and DT: (n, n_dt) [s]; for a rollout database, Y holds the states after 1, ..., rollout_steps steps).
"""

import h5py
import numpy as np


def read_solutions(file_h5, read_Y=True):
    """Arrays of all groups of file_h5, concatenated along the points (first axis).

    Returns X, col_names_X, Y, col_names_Y, DT, with Y None if read_Y is False, and DT [s] None if the
    groups have no DT dataset. The column names are the cols attributes of the first group.
    """

    list_X = []
    list_Y = []
    list_DT = []

    with h5py.File(file_h5, "r") as h5file_r:

        col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"]
        col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"]
        has_DT = "DT" in h5file_r["ITERATION_00000"]

        nb_solutions = len(h5file_r.keys())
        for i in range(nb_solutions):

            if i % 100 == 0:
                print(f"Opening solution: {i} / {nb_solutions}")

            grp = h5file_r[f"ITERATION_{i:05d}"]
            list_X.append(grp["X"][()])
            if read_Y:
                list_Y.append(grp["Y"][()])
            if has_DT:
                list_DT.append(grp["DT"][()])

    X = np.concatenate(list_X, axis=0)
    Y = np.concatenate(list_Y, axis=0) if read_Y else None
    DT = np.concatenate(list_DT, axis=0) if has_DT else None

    return X, col_names_X, Y, col_names_Y, DT
