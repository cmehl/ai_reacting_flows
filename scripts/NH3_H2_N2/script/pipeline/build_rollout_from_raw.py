"""Rollout raw database (nb_steps chained Cantera steps) for the states pre-selected by preselect_T1D.py.

Each rank takes raw groups g = rank::size of the single-step TFM_AMR raw db, reads the selected X rows (state at t,
unchanged), advances each with ONE constant-pressure reactor to k*dt, k = 1..nb_steps (same recipe as
build_raw_database_rollout.py), and writes X, Y (= step 1) and Y_multi to a per-rank file; rank 0 then merges into
<DTB>/solutions_rollout.h5 with sequential ITERATION_* groups (raw group order), the layout LearningDatabase expects.
Sanity check: step 1 must reproduce the raw db's Y (same mechanism, same dt).
"""
import os
import gc

import h5py
import numpy as np
import cantera as ct
from mpi4py import MPI

# ---- config ----
RAW = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_TFM_AMR_ALLDATA/CFD_DTB_TFM_AMR_ALLDATA_dt5e7/solutions.h5"
SEL = "selected_T1D_mc50000.npz"
MECH = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/data/Giovanni/STEC_A_noAR.yaml"
DTB = "CFD_DTB_TFM_AMR_ROLLOUT3_dt5e7"
OUT_FILE = "solutions_rollout.h5"
DT = 5.0e-7
NB_STEPS = 3
# ----------------


def react_multi(state, gas, dt, nb_steps):
    gas.TPY = state[0], state[1], state[2:] / state[2:].sum()
    r = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([r])
    out = np.empty((nb_steps, state.shape[0]))
    for k in range(1, nb_steps + 1):
        sim.advance(k * dt)
        out[k - 1, 0] = gas.T
        out[k - 1, 1] = gas.P
        out[k - 1, 2:] = gas.Y
    return out


comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
gas = ct.Solution(MECH)
ns = gas.n_species
d = np.load(SEL)
sel, offs = d["sel"], d["offs"]
n_groups = offs.size - 1
if rank == 0:
    os.makedirs(DTB, exist_ok=True)
    print(f">> {sel.size} states in {n_groups} groups over {size} ranks", flush=True)
comm.Barrier()

rank_path = os.path.join(DTB, f".rank_{rank:05d}.h5")
max_dev = 0.0
with h5py.File(RAW, "r") as raw, h5py.File(rank_path, "w") as out:
    for g in range(rank, n_groups, size):
        lo, hi = np.searchsorted(sel, [offs[g], offs[g + 1]])
        loc = sel[lo:hi] - offs[g]
        grp_in = raw[f"ITERATION_{g:05d}"]
        cols = list(grp_in["X"].attrs["cols"])
        X = grp_in["X"][()][loc]
        Y_raw = grp_in["Y"][()][loc]
        n = loc.size
        Ym = np.empty((n, NB_STEPS, X.shape[1]))
        Ym[:, :, -2:] = -1.0
        for i in range(n):
            Ym[i, :, :2 + ns] = react_multi(X[i, :2 + ns], gas, DT, NB_STEPS)
        dev = np.max(np.abs(Ym[:, 0, :2 + ns] - Y_raw[:, :2 + ns]) / (np.abs(Y_raw[:, :2 + ns]) + 1e-20))
        max_dev = max(max_dev, float(dev))
        grp = out.create_group(f"G_{g:05d}")
        for name, arr in [("X", X), ("Y", Ym[:, 0, :]), ("Y_multi", Ym)]:
            ds = grp.create_dataset(name, data=arr)
            ds.attrs["cols"] = cols
        grp["Y_multi"].attrs["dt"] = DT
        grp["Y_multi"].attrs["steps"] = np.arange(1, NB_STEPS + 1)
        print(f"[rank {rank}] group {g}: {n} states, step-1 vs raw max rel dev {dev:.2e}", flush=True)
        del X, Y_raw, Ym
        gc.collect()

all_dev = comm.gather(max_dev, root=0)
comm.Barrier()
if rank == 0:
    print(f">> step-1 vs raw Y: max rel deviation over all ranks {max(all_dev):.3e}", flush=True)
    path = os.path.join(DTB, OUT_FILE)
    tmp = path + ".tmp"
    n_tot = 0
    with h5py.File(tmp, "w") as h_out:
        for g in range(n_groups):
            rp = os.path.join(DTB, f".rank_{g % size:05d}.h5")
            with h5py.File(rp, "r") as h_in:
                h_in.copy(f"G_{g:05d}", h_out, name=f"ITERATION_{g:05d}")
                n_tot += h_in[f"G_{g:05d}/X"].shape[0]
    for r in range(size):
        os.remove(os.path.join(DTB, f".rank_{r:05d}.h5"))
    os.replace(tmp, path)
    assert n_tot == sel.size, (n_tot, sel.size)
    print(f">> wrote {path}: {n_groups} groups, {n_tot} states", flush=True)
