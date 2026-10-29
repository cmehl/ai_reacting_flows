"""Chemical heat release produced by CVODE vs by the network inside the P K3 dt 5e-6 CONVERGE runs (hybride 3e-5 and sans hybride), rebuilt
from the chemistry itself (MASS_SOURCE in the posts is not a pure chemical source: non-zero down to 293 K in every run, incl. SAGE).
Per post, hot cells (T > T_AI): CONVERGE choice rebuilt offline (raw network output, max relative element drift > TOL -> CVODE; TOL = inf
for the run without hybrid), one step dt: CVODE cells advanced by Cantera, network cells by the network + renorm 3; heat release
q = rho V sum_k h_f,k (Y_k(t) - Y_k(t+dt)) / dt. Cells T <= T_AI: no chemistry. Check: Q_cvode + Q_ann vs thermo.out HR_Rate.
SLURM array over (run, post): task i handles jobs[i::n]. Output: qchem/qchem_<run>_<post>.csv"""
import importlib.util
import os
import sys

import cantera as ct
import h5py
import numpy as np
import pandas as pd
import torch

# ---- config ----
ARF = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric"
C_ROOT = f"{ARF}/.idea3/MLP_CFD_REDUCED_A_NEW_INPUT_RESTARTS/converge_TEST_MODELS_32inst"
RUNS = {"PK3-H": ("PERSP_K3ROLL_DT5E6_TOL3E5", 3e-5), "PK3-noH": ("PERSP_K3ROLL_DT5E6_NOHYB", np.inf)}
T_ROOT = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/TEST_MODELS_32inst"
TEST_FOLDER = f"{ARF}/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3/test_persp_kmeans3roll_dt5e6"
MECH_YAML = f"{C_ROOT}/debug_TFM_AMR_mc50000/results/mech_converge.yaml"
T_WIN = (0.5e-3, 80e-3)
EVERY = 1e-3                  # one post per ms
DT = 5e-6
T_AI = 800.0
ELEMS = ["H", "O", "N"]
CHUNK = 1_000_000
OUT = "qchem"
# ----------------

task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
ntask = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
sys.path.insert(0, f"{ARF}/src")
from ai_reacting_flows.ann_model_generation.cfd_snapshot_testing import CFDSnapshotTester  # noqa: E402

spec = importlib.util.spec_from_file_location("cmpcfg", f"{C_ROOT}/cmp_TFM_AMR_PERSP_K3ROLL_DT5E6_TOL3E5_LONG/cmp_config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

tester = CFDSnapshotTester(run_folder=TEST_FOLDER)
SP = list(tester.species_names)
gas = ct.Solution(MECH_YAML)
idx = [gas.species_index(s) for s in SP]
assert len(idx) == gas.n_species
Wk = gas.molecular_weights[idx]
EMAT = np.array([[gas.n_atoms(i, gas.element_index(e)) * gas.atomic_weight(gas.element_index(e)) / w for i, w in zip(idx, Wk)] for e in ELEMS])
gas.TP = 298.15, ct.one_atm
HF = np.array([gas.species(s).thermo.h(298.15) for s in SP]) / Wk          # J/kg


def predict(T, Y):
    out = np.empty_like(Y)
    lab = tester._assign_clusters(T, None, Y)
    for c in range(tester.nb_clusters):
        ic = np.where(lab == c)[0]
        for i0 in range(0, len(ic), CHUNK):
            ii = ic[i0:i0 + CHUNK]
            Xs = np.log(np.clip(Y[ii], tester.threshold, None))
            Xsc = (np.column_stack([T[ii], Xs]) - tester.Xscaler_mean[c]) / (tester.Xscaler_std[c] + 1e-7)
            with torch.no_grad():
                p = tester.models[c](torch.tensor(Xsc, dtype=torch.float64, device=tester.device)).cpu().numpy()
            out[ii] = np.exp(Xs + tester.Yscaler_mean[c] + (tester.Yscaler_std[c] + 1e-7) * p)
    return out


def renorm3(Yin, Yout):
    dZ = Yin @ EMAT.T - Yout @ EMAT.T
    EW = EMAT[None, :, :] * Yout[:, None, :]
    lam = np.linalg.solve(np.einsum("nek,fk->nef", EW, EMAT), dZ[:, :, None])[:, :, 0]
    return Yout + np.einsum("nek,ne->nk", EW, lam)


jobs = []
for lab, (case, tol) in RUNS.items():
    P = cfg.posts(f"{T_ROOT}/CAS_TFM_AMR_{case}")
    ts = {k: int(k[4:]) * 1e-6 for k in P}
    for target in np.arange(T_WIN[0] + 0.5e-3, T_WIN[1] + 1e-9, EVERY):
        k = min(ts, key=lambda kk: abs(ts[kk] - target))
        if abs(ts[k] - target) < 0.3e-3:
            jobs.append((lab, case, tol, k, P[k]))
jobs = sorted(set(jobs), key=lambda j: (j[0], j[3]))
for lab, case, tol, key, f in jobs[task::ntask]:
    t = float(os.path.basename(f).split("_+")[1][:-3])
    with h5py.File(f, "r") as h:
        g = h["STREAM_00/CELL_CENTER_DATA"]
        T = g["TEMPERATURE"][()].astype(np.float64)
        hot = np.where(T > T_AI)[0]
        T = T[hot]
        P_ = g["PRESSURE"][()][hot].astype(np.float64)
        Y = np.column_stack([g[f"MASSFRAC_{s}"][()][hot] for s in SP]).astype(np.float64)
        m = (g["DENSITY"][()][hot] * g["VOLUME"][()][hot]).astype(np.float64)
    Y /= Y.sum(1, keepdims=True)
    Yraw = predict(T, Y)
    drift = np.max(np.abs(Yraw @ EMAT.T - Y @ EMAT.T) / (Y @ EMAT.T), axis=1)
    cv = drift > tol
    Y1 = renorm3(Y, Yraw)
    for i in np.where(cv)[0]:
        try:
            gas.TPY = T[i], P_[i], Y[i]
            ct.ReactorNet([ct.IdealGasConstPressureReactor(gas)]).advance(DT)
            Y1[i] = gas.Y
        except ct.CanteraError:
            pass
    q = m * ((Y - Y1) @ HF) / DT                       # W
    row = {"run": lab, "post": key, "t": t, "Q_cvode_W": q[cv].sum(), "Q_ann_W": q[~cv].sum(), "n_hot": len(T), "n_cvode": int(cv.sum())}
    pd.DataFrame([row]).to_csv(f"{OUT}/qchem_{lab}_{key}.csv", index=False)
    print(row, flush=True)
print(">> done", flush=True)
