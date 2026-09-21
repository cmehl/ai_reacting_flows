"""Full-domain ANN-vs-CVODE test of the reduced-mechanism STOCH models on the whole CFD snapshot (10.69M cells).

CVODE is computed ONCE on every cell with T > T_THRESHOLD (parallel, cached in REF_FILE), then every model of CASES is
evaluated against that same reference (CFDSnapshotTester._predict_ann, one process per model). Cells at or below
T_THRESHOLD are identity for both CVODE and the ANN (mask_cvode_below_threshold behaviour) and are left out of the
metrics; their number is reported.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np
import oyaml as yaml

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ROOT = "/work/kotlarcm/WORK/AI/clean/ai_reacting_flows"
WT = f"{ROOT}/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2"
HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT = f"{ROOT}/.idea/CFD_REDUCED_high_resolution/DATA/testing/post000065_+3.00001e+00.h5"
MECH = f"{ROOT}/.idea/NH3_H2_N2_REDUCED/MODELS/MODEL_NH3_H2_N2_REDUCED_perspecies/STEC_A_noAR.yaml"
TIME_STEP = 5.0e-6
T_THRESHOLD = 800.0
N_CVODE_WORKERS = 20
N_MODEL_WORKERS = 2
CHUNK = 500_000
REF_FILE = f"{HERE}/reference_cvode_full_domain.h5"
OUT_DIR = f"{HERE}/results"
RUN_CVODE = not os.path.exists(REF_FILE)
NCHEM = ["NH3", "NH2", "NH", "N", "NNH", "N2H2", "HNO", "NO2", "N2O", "NO"]
T_BINS = [800, 1500, 1900, 2050, 2090, 2110, 3000]

CASES = {
    "old_logk2": (f"{ROOT}/.idea/NH3_H2_N2_REDUCED", "MODEL_NH3_H2_N2_REDUCED_logk2"),
    "old_persize": (f"{ROOT}/.idea/NH3_H2_N2_REDUCED", "MODEL_NH3_H2_N2_REDUCED_persize"),
    "old_perspecies_REF": (f"{ROOT}/.idea/NH3_H2_N2_REDUCED", "MODEL_NH3_H2_N2_REDUCED_perspecies"),
    "short_rollout_0cl_thr1e14": (f"{WT}/STOCH_REDUCED_ROLLOUT", "MODEL_NH3_H2_N2_REDUCED_ROLLOUT_perspecies_2x64_noclust_thr1e14"),
    "long_single_0cl_thr1e14": (f"{WT}/STOCH_REDUCED_ROLLOUT_LONG", "MODEL_NH3_H2_N2_REDUCED_SINGLE_LONG_perspecies_2x64_noclust_thr1e14"),
    "long_single_2cl_thr1e10": (f"{WT}/STOCH_REDUCED_ROLLOUT_LONG", "MODEL_NH3_H2_N2_REDUCED_SINGLE_LONG_perspecies_k2_thr1e10"),
    "long_rollout_0cl_thr1e14": (f"{WT}/STOCH_REDUCED_ROLLOUT_LONG", "MODEL_NH3_H2_N2_REDUCED_ROLLOUT_LONG_perspecies_2x64_noclust_thr1e14"),
    # thinned-tail experiments (log threshold 1e-14 in all of them)
    "thin_E1_rollout_2cl_thr1e14": (f"{WT}/STOCH_REDUCED_ROLLOUT_LONG", "MODEL_NH3_H2_N2_REDUCED_THIN_E1_rollout_k2_thr1e14"),
    "thin_E1c_single_2cl_thr1e14": (f"{WT}/STOCH_REDUCED_ROLLOUT_LONG", "MODEL_NH3_H2_N2_REDUCED_THIN_E1c_single_k2_thr1e14"),
    "thin_E2_rollout_0cl_thr1e14": (f"{WT}/STOCH_REDUCED_ROLLOUT_LONG", "MODEL_NH3_H2_N2_REDUCED_THIN_E2_rollout_k1_thr1e14"),
}

# ---------------------------------------------------------------------------
# CVODE reference (parallel)
# ---------------------------------------------------------------------------
_gas = None


def _init_cvode_worker():
    global _gas
    import cantera as ct
    _gas = ct.Solution(MECH)


def _cvode_chunk(args):
    import cantera as ct
    T, P, Y = args
    T_new, Y_new, n_failed = np.empty(len(T)), np.empty_like(Y), 0
    for i in range(len(T)):
        try:
            _gas.TPY = T[i], P[i], Y[i]
            r = ct.IdealGasConstPressureReactor(_gas)
            ct.ReactorNet([r]).advance(TIME_STEP)
            T_new[i], Y_new[i] = _gas.T, _gas.Y
        except ct.CanteraError:
            n_failed += 1
            T_new[i], Y_new[i] = T[i], Y[i]
    return T_new, Y_new, n_failed


def build_reference():
    import cantera as ct
    species = ct.Solution(MECH).species_names
    print(f">> Reading {SNAPSHOT}", flush=True)
    with h5py.File(SNAPSHOT, "r") as f:
        c = f["STREAM_00/CELL_CENTER_DATA"]
        T_all = c["TEMPERATURE"][()].astype(np.float64)
        hot = T_all > T_THRESHOLD
        idx = np.where(hot)[0]
        P = c["PRESSURE"][()].astype(np.float64)[idx]
        Y = np.column_stack([c[f"MASSFRAC_{s}"][()].astype(np.float64)[idx] for s in species])
        cx, cy, cz = (c[k][()].astype(np.float64)[idx] for k in ("XCEN_X", "XCEN_Y", "XCEN_Z"))
    T = T_all[idx]
    Y = Y / Y.sum(axis=1, keepdims=True)
    print(f">> {T_all.size} cells, {T.size} above {T_THRESHOLD:g} K ({T_all.size - T.size} identity, left out)", flush=True)

    starts = list(range(0, T.size, 5000))
    jobs = [(T[s:s + 5000], P[s:s + 5000], Y[s:s + 5000]) for s in starts]
    T_c, Y_c, n_failed = [], [], 0
    t0 = time.time()
    with mp.get_context("fork").Pool(N_CVODE_WORKERS, initializer=_init_cvode_worker) as pool:
        for k, (Tn, Yn, nf) in enumerate(pool.imap(_cvode_chunk, jobs)):
            T_c.append(Tn); Y_c.append(Yn); n_failed += nf
            if k % 50 == 0:
                print(f"  CVODE chunk {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    T_c, Y_c = np.concatenate(T_c), np.concatenate(Y_c)
    print(f">> CVODE done in {time.time() - t0:.0f}s, {n_failed} cells fell back to identity", flush=True)

    with h5py.File(REF_FILE, "w") as h:
        h.attrs["species"] = species
        h.attrs["n_cells_total"] = T_all.size
        h.attrs["n_cvode_failed"] = n_failed
        for k, v in (("T_ini", T), ("P_ini", P), ("Y_ini", Y), ("T_cvode", T_c), ("Y_cvode", Y_c),
                     ("XCEN_X", cx), ("XCEN_Y", cy), ("XCEN_Z", cz)):
            h.create_dataset(k, data=v)


# ---------------------------------------------------------------------------
# Model evaluation (one process per model)
# ---------------------------------------------------------------------------
def _legacy_assign_clusters(self, T, P, Y):
    """Cluster assignment of databases processed BEFORE the k-means feature fix (commit d8f6a85 removed it from the tester).

    Reproduces the old 23-wide feature layout [T, P, species..., Prog_var=-1, HRR=-1, cluster=0], with the log transform
    applied to columns 1..n_species (P and all species but the last), so cells reach the cluster the model was trained on.
    """
    n = T.shape[0]
    if self.nb_clusters == 1:
        return np.zeros(n, dtype=int)
    raw = np.column_stack([T, P, Y, -np.ones(n), -np.ones(n), np.zeros(n)])
    if self.log_transform_X > 0:
        cols = list(range(1, 1 + self.n_species))
        raw[:, cols] = np.clip(raw[:, cols], self.threshold, None)
        raw[:, cols] = np.log(raw[:, cols])
    return self.kmeans.predict(self.kmeans_scaler.transform(raw))


def evaluate_model(name):
    import types
    import torch
    torch.set_num_threads(4)
    from ai_reacting_flows.ann_model_generation.cfd_snapshot_testing import CFDSnapshotTester

    run_folder, model_folder = CASES[name]
    tmp = f"{OUT_DIR}/_run_{name}"
    os.makedirs(tmp, exist_ok=True)
    for link, target in (("MODELS", f"{run_folder}/MODELS"),):
        if not os.path.lexists(f"{tmp}/{link}"):
            os.symlink(target, f"{tmp}/{link}")
    with open(f"{run_folder}/MODELS/{model_folder}/networks_params.yaml") as f:
        top = yaml.safe_load(f)["database_path"].split("/")[0]
    if not os.path.lexists(f"{tmp}/{top}"):
        os.symlink(f"{run_folder}/{top}", f"{tmp}/{top}")
    with open(f"{tmp}/test_ann_vs_cvode.yaml", "w") as f:
        yaml.safe_dump({"models_folder": model_folder, "test_data_file": SNAPSHOT, "time_step": TIME_STEP,
                        "T_threshold": T_THRESHOLD, "n_sample": 10 ** 8, "output_file": f"{tmp}/unused.h5",
                        "mask_cvode_below_threshold": True}, f)
    tester = CFDSnapshotTester(tmp)
    if name.startswith("old_"):
        tester._assign_clusters = types.MethodType(_legacy_assign_clusters, tester)

    with h5py.File(REF_FILE, "r") as h:
        species = [s for s in h.attrs["species"]]
        assert species == tester.species_names, "species order mismatch"
        T0, P0, Y0 = h["T_ini"][()], h["P_ini"][()], h["Y_ini"][()]
        Tc, Yc = h["T_cvode"][()], h["Y_cvode"][()]
    n = T0.size
    T_ann, Y_ann = np.empty(n), np.empty_like(Y0)
    t0 = time.time()
    for s in range(0, n, CHUNK):
        sl = slice(s, s + CHUNK)
        T_ann[sl], Y_ann[sl], _ = tester._predict_ann(T0[sl], P0[sl], Y0[sl])
        print(f"  [{name}] {min(s + CHUNK, n)}/{n}  {time.time() - t0:.0f}s", flush=True)

    eT = T_ann - Tc
    sse_T = float(np.sum(eT ** 2))
    res = {"model": name, "n_cells": int(n), "T_RMSE": float(np.sqrt(sse_T / n)), "T_max_abs": float(np.abs(eT).max()),
           "T_mean_err": float(eT.mean()), "sumY_min": float(Y_ann.sum(1).min()), "sumY_max": float(Y_ann.sum(1).max())}
    rm = {sp: float(np.sqrt(np.mean((Y_ann[:, j] - Yc[:, j]) ** 2))) for j, sp in enumerate(species)}
    res["species_RMSE"] = rm
    res["Nchem_RMSE_sum_1e6"] = 1e6 * sum(rm[s] for s in NCHEM)
    res["T_RMSE_by_Tini_bin"] = {}
    for lo, hi in zip(T_BINS[:-1], T_BINS[1:]):
        m = (T0 >= lo) & (T0 < hi)
        res["T_RMSE_by_Tini_bin"][f"[{lo},{hi})"] = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean(eT[m] ** 2))) if m.any() else None}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{name}.json", "w") as f:
        json.dump(res, f, indent=1)
    with h5py.File(f"{OUT_DIR}/{name}_ann.h5", "w") as h:
        h.create_dataset("T_ann", data=T_ann)
        h.create_dataset("Y_ann", data=Y_ann)
    print(f"[{name}] T_RMSE={res['T_RMSE']:.4f} K  Nchem={res['Nchem_RMSE_sum_1e6']:.1f}", flush=True)
    return res


def evaluate_model_safe(name):
    """One failing model (e.g. unloadable checkpoint) must not abort the others."""
    import traceback
    try:
        return evaluate_model(name)
    except Exception:
        print(f"[{name}] FAILED:\n{traceback.format_exc()}", flush=True)
        return None


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if RUN_CVODE:
        build_reference()
    else:
        print(f">> Reusing cached CVODE reference {REF_FILE}", flush=True)
    # A case is evaluated when it has no cached result yet AND its model has been copied locally.
    todo = [c for c, (rf, mf) in CASES.items()
            if not os.path.exists(f"{OUT_DIR}/{c}.json") and os.path.isdir(f"{rf}/MODELS/{mf}")]
    print(f">> Evaluating {len(todo)} model(s): {todo}", flush=True)
    with mp.get_context("spawn").Pool(N_MODEL_WORKERS) as pool:
        for r in pool.imap_unordered(evaluate_model_safe, todo):
            pass
    print("ALL DONE", flush=True)
