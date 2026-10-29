"""MLP 2x256 K3-ROLL dt 5e-6 sans hybride crashed at 4.195 ms (t_chem too large -> dt_recover -> Temperature extrapolation near
(-0.009, 0.007, -0.065) m). Offline: apply the MLP and the P K3 networks to the MLP state of the last posts (4.0 / 4.1 ms), find the cells
where the network makes a huge one-step jump (implied dT at constant h, P), locate them, and compare with SAGE / P K3 sans hybride at the same time."""
import glob
import os
import sys

import cantera as ct
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

# ---- config ----
ARF = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric"
T_ROOT = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/TEST_MODELS_32inst"
RUNS = {"MLP": "MLP256_K3ROLL_DT5E6_NOHYB", "PK3": "PERSP_K3ROLL_DT5E6_NOHYB", "SAGE": "SAGE_DT5E6"}
NETS = {"MLP": f"{ARF}/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3/test_mlp256_kmeans3roll_dt5e6",
        "PK3": f"{ARF}/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3/test_persp_kmeans3roll_dt5e6"}
MECH_YAML = f"{ARF}/.idea3/MLP_CFD_REDUCED_A_NEW_INPUT_RESTARTS/converge_TEST_MODELS_32inst/debug_TFM_AMR_mc50000/results/mech_converge.yaml"
POSTS = ["post000041", "post000042"]          # 4.0 / 4.1 ms (same numbering in outputs_original of every run)
WARN = np.array([[-0.00912017, 0.00637565, -0.063625], [-0.00862017, 0.00687565, -0.066375], [-0.00712017, 0.00837565, -0.066625]])
R_NEAR = 3e-3
T_AI = 800.0
N_TOP = 20000                                   # cells sent to Cantera (largest |sum h_k dY_k|)
CHUNK = 1_000_000
ELEMS = ["H", "O", "N"]
# ----------------

sys.path.insert(0, f"{ARF}/src")
from ai_reacting_flows.ann_model_generation.cfd_snapshot_testing import CFDSnapshotTester

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
testers = {k: CFDSnapshotTester(run_folder=v) for k, v in NETS.items()}
SP = list(testers["MLP"].species_names)
assert SP == list(testers["PK3"].species_names)
gas = ct.Solution(MECH_YAML)
idx = [gas.species_index(s) for s in SP]
Wk = gas.molecular_weights[idx]
EMAT = np.array([[gas.n_atoms(i, gas.element_index(e)) * gas.atomic_weight(gas.element_index(e)) / w for i, w in zip(idx, Wk)] for e in ELEMS])
gas.TP = 298.15, ct.one_atm
HF = np.array([gas.species(s).thermo.h(298.15) for s in SP]) / Wk          # J/kg


def predict(tester, T, Y):
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
    return out, lab


def load(case, post):
    f = glob.glob(f"{T_ROOT}/CAS_TFM_AMR_{case}/outputs_original/output/{post}_*.h5")[0]
    with h5py.File(f, "r") as h:
        g = h["STREAM_00/CELL_CENTER_DATA"]
        d = {"T": g["TEMPERATURE"][()].astype(np.float64), "P": g["PRESSURE"][()].astype(np.float64) if "PRESSURE" in g else None,
             "Y": np.column_stack([g[f"MASSFRAC_{s}"][()] for s in SP]).astype(np.float64),
             "xyz": np.column_stack([g[f"XCEN_{c}"][()] for c in "XYZ"]).astype(np.float64), "vol": g["VOLUME"][()].astype(np.float64)}
    d["t"] = float(os.path.basename(f).split("_+")[1][:-3])
    return d


def dT_hp(T, P, Y0, Y1):
    """T after the chemistry step at constant h and P (what CONVERGE's energy equation sees)."""
    out = np.empty(len(T))
    for i in range(len(T)):
        gas.TPY = T[i], P[i], np.pad(Y0[i], (0, 0))
        h = gas.enthalpy_mass
        y1 = np.zeros(gas.n_species); y1[idx] = Y1[i]
        try:
            gas.HPY = h, P[i], y1
            out[i] = gas.T - T[i]
        except Exception:
            out[i] = np.nan
    return out


def Yfull(Y):
    y = np.zeros((len(Y), gas.n_species)); y[:, idx] = Y
    return y


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
for post in POSTS:
    m = load(RUNS["MLP"], post)
    P = m["P"] if m["P"] is not None else np.full(len(m["T"]), ct.one_atm)
    hot = np.where(m["T"] > T_AI)[0]
    print(f"\n######## {post} t = {m['t'] * 1e3:.3f} ms, MLP state: {len(m['T'])} cells, {len(hot)} > {T_AI} K, T in [{m['T'].min():.1f}, {m['T'].max():.1f}]", flush=True)
    res = {}
    for net, tester in testers.items():
        Yo, lab = predict(tester, m["T"][hot], m["Y"][hot])
        dY = Yo - m["Y"][hot]
        q = -(dY @ HF)                                     # J/kg released in one step (approx, h_f only)
        Zi, Zo = m["Y"][hot] @ EMAT.T, Yo @ EMAT.T
        drift = np.max(np.abs(Zo - Zi) / Zi, axis=1)
        sumY = Yo.sum(1)
        top = np.argsort(-np.abs(q))[:N_TOP]
        Yn = Yo[top] / Yo[top].sum(1, keepdims=True)      # renormalised as a minimum
        gas_dT = np.full(len(hot), np.nan)
        gas_dT[top] = [np.nan] * len(top)
        yi = Yfull(m["Y"][hot][top]); yo = Yfull(Yn)
        dT = np.empty(len(top))
        for j, c in enumerate(hot[top]):
            gas.TPY = m["T"][c], P[c], yi[j]
            h = gas.enthalpy_mass
            try:
                gas.HPY = h, P[c], yo[j]; dT[j] = gas.T - m["T"][c]
            except Exception:
                dT[j] = np.nan
        gas_dT[top] = dT
        res[net] = dict(Yo=Yo, lab=lab, q=q, drift=drift, sumY=sumY, dT=gas_dT)
        ok = np.isfinite(gas_dT)
        print(f"--- net {net}: |q| p50/p99.99/max = {np.median(np.abs(q)):.3e}/{np.percentile(np.abs(q), 99.99):.3e}/{np.abs(q).max():.3e} J/kg | "
              f"element drift p99/max = {np.percentile(drift, 99):.2e}/{drift.max():.2e} | sumY-1 max = {np.abs(sumY - 1).max():.2e}")
        print(f"    implied dT (top {N_TOP} |q| cells): max = {np.nanmax(gas_dT):.1f} K, min = {np.nanmin(gas_dT):.1f} K, "
              f"n(|dT| > 50 K) = {(np.abs(gas_dT[ok]) > 50).sum()}, n(|dT|/T > 0.5) = {(np.abs(gas_dT[ok]) / m['T'][hot][ok] > 0.5).sum()}")
        w = np.argsort(-np.nan_to_num(np.abs(gas_dT), nan=-1))[:8]
        rows = []
        for j in w:
            c = hot[j]
            dist = np.min(np.linalg.norm(WARN - m["xyz"][c], axis=1))
            rows.append({"cell": c, "x": m["xyz"][c, 0], "y": m["xyz"][c, 1], "z": m["xyz"][c, 2], "dx_mm": 1e3 * m["vol"][c] ** (1 / 3),
                         "d_warn_mm": 1e3 * dist, "T": m["T"][c], "dT": gas_dT[j], "cluster": lab[j], "drift": drift[j],
                         **{f"{s}": m["Y"][c, SP.index(s)] for s in ["H2", "O2", "NH3", "H2O", "OH", "N"]},
                         **{f"{s}_out": Yo[j, SP.index(s)] for s in ["H2", "O2", "NH3", "H2O", "OH", "N"]}})
        print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.4g}"))
    # cells near the warnings
    tree = cKDTree(m["xyz"])
    near = np.unique(np.concatenate(tree.query_ball_point(WARN, R_NEAR)))
    hn = np.isin(hot, near)
    print(f"--- {len(near)} cells within {R_NEAR * 1e3:.0f} mm of the warning cells: T in [{m['T'][near].min():.1f}, {m['T'][near].max():.1f}], "
          f"{hn.sum()} > {T_AI} K, cell size {1e3 * m['vol'][near].min() ** (1 / 3):.3f}-{1e3 * m['vol'][near].max() ** (1 / 3):.3f} mm")
    for net in testers:
        r = res[net]
        if hn.any():
            print(f"    {net}: implied dT there (cells in top set): max {np.nanmax(np.r_[r['dT'][hn], -np.inf]):.1f} K, "
                  f"|q| max {np.abs(r['q'][hn]).max():.3e}, drift max {r['drift'][hn].max():.2e}, clusters {np.bincount(r['lab'][hn], minlength=3)}")
    # same place in SAGE / PK3 at the same time
    cen = WARN.mean(0)
    for lab_run in ["MLP", "PK3", "SAGE"]:
        d = m if lab_run == "MLP" else load(RUNS[lab_run], post)
        ii = cKDTree(d["xyz"]).query_ball_point(cen, R_NEAR)
        T = d["T"][ii]
        print(f"    {lab_run:4s} @ {d['t'] * 1e3:.3f} ms, sphere {R_NEAR * 1e3:.0f} mm around the warnings: {len(ii)} cells, T min/mean/max "
              f"{T.min():.1f}/{T.mean():.1f}/{T.max():.1f}, Y_O2 mean {d['Y'][ii, SP.index('O2')].mean():.4f}, Y_NH3 {d['Y'][ii, SP.index('NH3')].mean():.4f}, "
              f"Y_H2O {d['Y'][ii, SP.index('H2O')].mean():.4f}, Y_N max {d['Y'][ii, SP.index('N')].max():.3e}, Y_NO max {d['Y'][ii, SP.index('NO')].max():.3e}")
print(">> done")
