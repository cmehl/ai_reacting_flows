"""A priori calibration of the hybrid ANN/CVODE threshold of P K3 tanh dt 5e-6 (Mehl & Aubagnac-Karkar, PoF 35, 067115, 2023, Fig. 11-12):
on SAGE dt 5e-6 states (whole hot domain, T > T_AI, several instants), compare the network's error proxy (max relative element drift of
the RAW output = CONVERGE's hybrid_ann_mode 1 criterion, + |sum Y - 1|) with the TRUE one-step error vs CVODE.
SLURM array: task i -> instant i // N_CHUNK, chunk i % N_CHUNK of the hot cells. Output: steps/apriori_<t_us>_<chunk>.npz"""
import glob
import os
import sys
import time

import cantera as ct
import h5py
import numpy as np
import torch

# ---- config ----
ARF = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric"
CASE = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/TEST_MODELS_32inst/CAS_TFM_AMR_SAGE_DT5E6"
TEST_FOLDER = f"{ARF}/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3/test_persp_kmeans3roll_dt5e6"     # P K3 tanh dt 5e-6 (3 clusters)
MECH_YAML = f"{ARF}/.idea3/MLP_CFD_REDUCED_A_NEW_INPUT_RESTARTS/converge_TEST_MODELS_32inst/debug_TFM_AMR_mc50000/results/mech_converge.yaml"
TIMES = [1e-3, 5e-3, 20e-3, 80e-3, 150e-3]
N_CHUNK = 40
DT = 5e-6
T_AI = 800.0
ELEMS = ["H", "O", "N"]
CHUNK_NN = 1_000_000
OUT = "steps"
# ----------------

task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
k_t, k_c = divmod(task, N_CHUNK)
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
sys.path.insert(0, f"{ARF}/src")
from ai_reacting_flows.ann_model_generation.cfd_snapshot_testing import CFDSnapshotTester  # noqa: E402

tester = CFDSnapshotTester(run_folder=TEST_FOLDER)
assert tester.output_omegas and not tester.log_excluded_species
SP = list(tester.species_names)
gas = ct.Solution(MECH_YAML)
idx = [gas.species_index(s) for s in SP]
assert len(idx) == gas.n_species, "network species must cover the whole mechanism"
Wk = gas.molecular_weights[idx]
EMAT = np.array([[gas.n_atoms(i, gas.element_index(e)) * gas.atomic_weight(gas.element_index(e)) / w for i, w in zip(idx, Wk)] for e in ELEMS])


def post_at(t):
    """SAGE post closest to t over every outputs_* segment (later segment wins on ties)."""
    segs = [f"{CASE}/outputs_original/output"] + sorted(glob.glob(f"{CASE}/outputs_restart*/output"), key=lambda d: int(d.split("restart")[1].split("/")[0]))
    best = None
    for d in segs:
        for f in glob.glob(f"{d}/post*.h5"):
            tf = float(os.path.basename(f).split("_+")[1][:-3])
            if best is None or abs(tf - t) <= abs(best[1] - t):
                best = (f, tf)
    return best


def predict(T, Y):
    out = np.empty_like(Y)
    lab = tester._assign_clusters(T, None, Y)
    for c in range(tester.nb_clusters):
        ic = np.where(lab == c)[0]
        for i0 in range(0, len(ic), CHUNK_NN):
            ii = ic[i0:i0 + CHUNK_NN]
            Xs = np.log(np.clip(Y[ii], tester.threshold, None))
            Xsc = (np.column_stack([T[ii], Xs]) - tester.Xscaler_mean[c]) / (tester.Xscaler_std[c] + 1e-7)
            with torch.no_grad():
                p = tester.models[c](torch.tensor(Xsc, dtype=torch.float64, device=tester.device)).cpu().numpy()
            out[ii] = np.exp(Xs + tester.Yscaler_mean[c] + (tester.Yscaler_std[c] + 1e-7) * p)
    return out, lab


def renorm3(Yin, Yout):
    """CONVERGE ai_renorm_yk 3: dY = W E^T (E W E^T)^-1 dZ, W = diag(Y_out), dZ = Z_in - Z_out."""
    dZ = Yin @ EMAT.T - Yout @ EMAT.T
    EW = EMAT[None, :, :] * Yout[:, None, :]                       # (n, 3, nsp) = E W
    M = np.einsum("nek,fk->nef", EW, EMAT)                         # E W E^T
    lam = np.linalg.solve(M, dZ[:, :, None])[:, :, 0]
    return Yout + np.einsum("nek,ne->nk", EW, lam)


f, tf = post_at(TIMES[k_t])
with h5py.File(f, "r") as h:
    g = h["STREAM_00/CELL_CENTER_DATA"]
    T = g["TEMPERATURE"][()].astype(np.float64)
    hot = np.where(T > T_AI)[0]
    sel = np.array_split(hot, N_CHUNK)[k_c]
    T = T[sel]
    P = g["PRESSURE"][()][sel].astype(np.float64)
    Y = np.column_stack([g[f"MASSFRAC_{s}"][()][sel] for s in SP]).astype(np.float64)
    m = (g["DENSITY"][()][sel] * g["VOLUME"][()][sel]).astype(np.float64)
print(f"task {task}: {os.path.basename(f)} t = {tf * 1e3:.3f} ms, chunk {k_c}/{N_CHUNK}: {len(sel)} of {len(hot)} hot cells", flush=True)
Y = Y / Y.sum(1, keepdims=True)

t0 = time.time()
Yraw, lab = predict(T, Y)
Zi, Zo = Y @ EMAT.T, Yraw @ EMAT.T
ee = np.max(np.abs(Zo - Zi) / Zi, axis=1)                          # CONVERGE hybrid criterion (Appendix B of the paper)
eR = np.abs(Yraw.sum(1) - 1.0)                                     # Eq. (10) of the paper
Yr3 = renorm3(Y, Yraw)
print(f"  network + renorm3 {time.time() - t0:.1f} s", flush=True)

n = len(T)
T_cv, Y_cv = np.full(n, np.nan), np.full_like(Y, np.nan)
T_r3, T_raw, t_cv = np.empty(n), np.empty(n), np.full(n, np.nan)   # t_cv: CVODE wall time per cell (cost proxy)
t0 = time.time()
for i in range(n):
    gas.TPY = T[i], P[i], Y[i]
    cp = gas.cp_mass
    hk = gas.partial_molar_enthalpies / gas.molecular_weights
    T_r3[i] = T[i] - hk @ (Yr3[i] - Y[i]) / cp                      # same enthalpy linearisation as CFDSnapshotTester
    T_raw[i] = T[i] - hk @ (Yraw[i] - Y[i]) / cp
    try:
        tc = time.perf_counter()
        rr = ct.IdealGasConstPressureReactor(gas)
        ct.ReactorNet([rr]).advance(DT)
        t_cv[i] = time.perf_counter() - tc
        T_cv[i], Y_cv[i] = gas.T, gas.Y
    except ct.CanteraError:
        pass
    if i % 50000 == 0:
        print(f"  cell {i}/{n} {time.time() - t0:.0f} s", flush=True)
print(f"  CVODE {time.time() - t0:.0f} s, failed {np.isnan(T_cv).sum()}", flush=True)
np.savez(f"{OUT}/apriori_{round(tf * 1e6):06d}_{k_c:02d}.npz", t=tf, species=np.array(SP), m=m, T0=T, Y0=Y, ee=ee, eR=eR, cluster=lab,
         T_r3=T_r3, Y_r3=Yr3, T_raw=T_raw, T_cv=T_cv, Y_cv=Y_cv, t_cv=t_cv)
print(">> done", flush=True)
