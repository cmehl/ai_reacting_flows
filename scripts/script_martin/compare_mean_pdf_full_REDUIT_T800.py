"""Full-3D-domain version of compare_mean_pdf_REDUIT_T800.py: time-averaged
domain bias per field + T-binned Y_H2 stats/PDF, SAGE vs MODEL,
CAS_LEWIS_UNITAIRE/REDUIT_T800 -- same chaos-vs-bias question, but over
*every* mesh cell (10,690,040) instead of just the X=0 slice (62,605 cells)
used by compare_mean_pdf_REDUIT_T800.py.

No slice cache exists for the full mesh (would be ~76GB for 100 frames x 19
fields x 10.69M cells), so this reads the raw post*.h5 snapshots directly,
same as animate_ann_sage_hybrid.py's build_cache(), but accumulates running
sums / 2-D histograms per frame instead of storing every frame -- O(mesh)
memory instead of O(mesh x frames). SAGE's own per-frame cell-array order is
assumed stable across frames (same assumption animate_ann_sage_hybrid.py
already makes -- it raises if the cell count changes); only MODEL's cells
are re-indexed onto SAGE's order every frame via a KDTree query (built once,
reused across frames, instead of rebuilt every frame).

Config is top-of-file UPPERCASE vars (no argparse). Needs SLURM (login node
kills long-running jobs after ~20s) -- see the .slurm wrapper next to this.
"""
import glob
import importlib.util
import os
import re
import sys
import time as _time

import numpy as np
from scipy.spatial import cKDTree

# ---- config ----
ANIMATE_SCRIPT = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/scripts/script_martin/animate_ann_sage_hybrid.py"
SAGE_DIR = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/SAGE/outputs_original/output"
MODEL_DIR = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/MODEL_perspecies_2x64_thresh1e14_noclust_T800/outputs_original/output"
MAX_FRAMES = 100
PDF_FIELDS = ["H", "H2", "H2O", "HNO", "HO2", "N", "N2", "N2H2", "N2O", "NH",
              "NH2", "NH3", "NNH", "NO", "NO2", "O", "O2", "OH"]  # all 18 species
BIAS_FIELDS = ["Temperature"] + PDF_FIELDS  # full bias table, not just a handful
T_LO, T_HI, N_T_BINS = 250.0, 2500.0, 60          # fine T grid for both the stats curve and the 2D hist
N_Y_BINS = 150                                    # per-species Y range is auto-picked from frame 0 (below)
N_T_SUPERBINS_PDF = 6                             # coarse groups of the fine T bins for the small-multiples PDF
OUT_PREFIX = "REDUIT_T800_full"
SAGE_COLOR = "#111111"    # near-black, distinct from MODEL_COLOR even where curves overlap
MODEL_COLOR = "#e0311f"   # vivid red -- high contrast vs black, still readable in both themes
# -----------------

spec = importlib.util.spec_from_file_location("animate_ann_sage_hybrid", ANIMATE_SCRIPT)
anim = importlib.util.module_from_spec(spec)
sys.modules["animate_ann_sage_hybrid"] = anim
spec.loader.exec_module(anim)

FIELDS = anim.FIELDS
labels = [anim.field_label(f) for f in FIELDS]
n_fields = len(FIELDS)
tj = labels.index("Temperature")

name_re = re.compile(r"^post\d+_(?P<time>[+-][0-9.eE+-]+)\.h5$")
sage_by_time = anim.index_by_time(SAGE_DIR, name_re)
model_by_time = anim.index_by_time(MODEL_DIR, name_re)
common = sorted(set(sage_by_time) & set(model_by_time), key=float)[:MAX_FRAMES]
n = len(common)
assert n, "no shared timestep found"
print(f"{n} shared frames, t={float(common[0]):.3e}..{float(common[-1]):.3e}s", flush=True)

T_edges = np.linspace(T_LO, T_HI, N_T_BINS + 1)
T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])

sum_cell_sage = None
sum_cell_model = None
tree_s = None
n_cells = None
t_min = np.inf
t_max = -np.inf
n_clipped_T = 0
Y_edges = {}          # per-species, picked from frame 0
hist_sage = {}
hist_model = {}
y_min = {f: np.inf for f in PDF_FIELDS}
y_max = {f: -np.inf for f in PDF_FIELDS}
n_clipped_Y = {f: 0 for f in PDF_FIELDS}

t0 = _time.time()
for i, tkey in enumerate(common):
    coords_s, data_s, t_s = anim.load(sage_by_time[tkey], FIELDS)
    coords_m, data_m, t_m = anim.load(model_by_time[tkey], FIELDS)
    assert abs(t_s - t_m) < 1e-9, (t_s, t_m)

    if tree_s is None:
        n_cells = coords_s.shape[0]
        tree_s = cKDTree(coords_s)
        sum_cell_sage = np.zeros((n_fields, n_cells), dtype=np.float64)
        sum_cell_model = np.zeros((n_fields, n_cells), dtype=np.float64)
        print(f"  full mesh: {n_cells} cells", flush=True)
        for f in PDF_FIELDS:
            key = "MASSFRAC_" + f
            v0 = np.concatenate([data_s[key], data_m[key]])
            hi = max(float(np.percentile(v0, 99.9)) * 3.0, float(v0.max()), 1e-12)
            Y_edges[f] = np.linspace(0.0, hi, N_Y_BINS + 1)
            hist_sage[f] = np.zeros((N_T_BINS, N_Y_BINS), dtype=np.float64)
            hist_model[f] = np.zeros((N_T_BINS, N_Y_BINS), dtype=np.float64)
    elif coords_s.shape[0] != n_cells:
        raise RuntimeError(f"mesh cell count changed at t={t_s:.3e}s -- adaptive mesh not supported")

    dist, idx = tree_s.query(coords_m, k=1, workers=-1)
    if dist.max() > 1e-9:
        print(f"  WARNING t={t_s:.3e}: max nearest-neighbor dist = {dist.max():.3e}", flush=True)

    T_s = data_s["TEMPERATURE"]
    T_m = np.empty(n_cells)
    T_m[idx] = data_m["TEMPERATURE"]

    for fj, f in enumerate(FIELDS):
        sum_cell_sage[fj] += data_s[f]
        vm = np.empty(n_cells)
        vm[idx] = data_m[f]
        sum_cell_model[fj] += vm

    t_min = min(t_min, T_s.min(), T_m.min())
    t_max = max(t_max, T_s.max(), T_m.max())
    n_clipped_T += int(((T_s < T_LO) | (T_s > T_HI)).sum() + ((T_m < T_LO) | (T_m > T_HI)).sum())

    for f in PDF_FIELDS:
        key = "MASSFRAC_" + f
        Y_s = data_s[key]
        Y_m = np.empty(n_cells)
        Y_m[idx] = data_m[key]
        y_min[f] = min(y_min[f], Y_s.min(), Y_m.min())
        y_max[f] = max(y_max[f], Y_s.max(), Y_m.max())
        edges_f = Y_edges[f]
        n_clipped_Y[f] += int((Y_s > edges_f[-1]).sum() + (Y_m > edges_f[-1]).sum())
        h_s, _, _ = np.histogram2d(T_s, Y_s, bins=[T_edges, edges_f])
        h_m, _, _ = np.histogram2d(T_m, Y_m, bins=[T_edges, edges_f])
        hist_sage[f] += h_s
        hist_model[f] += h_m

    if (i + 1) % 10 == 0 or i == n - 1:
        el = _time.time() - t0
        print(f"  frame {i + 1}/{n}  ({el:.0f}s, {el / (i + 1):.2f}s/frame)", flush=True)

print(f"T range seen: [{t_min:.1f}, {t_max:.1f}]K  (bins cover [{T_LO},{T_HI}])  "
      f"{n_clipped_T} points outside bin range", flush=True)
for f in PDF_FIELDS:
    print(f"Y_{f} range seen: [{y_min[f]:.4g}, {y_max[f]:.4g}]  (bins cover [0,{Y_edges[f][-1]:.4g}])  "
          f"{n_clipped_Y[f]} points outside bin range", flush=True)

mean_cell_sage = sum_cell_sage / n
mean_cell_model = sum_cell_model / n

print("\nField                domain-mean SAGE   domain-mean MODEL   bias        rel.bias   spatial std of bias", flush=True)
rows = []
for fname in BIAS_FIELDS:
    fj = labels.index(fname)
    ms, mm = mean_cell_sage[fj], mean_cell_model[fj]
    diff = mm - ms
    bias = float(diff.mean())
    bias_std = float(diff.std())
    dom_s = float(ms.mean())
    dom_m = float(mm.mean())
    rel = bias / (dom_s or 1e-30) * 100
    rows.append((fname, dom_s, dom_m, bias, rel, bias_std, float(np.abs(diff).max())))
    print(f"{fname:12s}  {dom_s:16.6g}  {dom_m:16.6g}  {bias:10.4g}  {rel:+8.4f}%  {bias_std:10.4g}", flush=True)

with open(f"bias_table_{OUT_PREFIX}.csv", "w") as fh:
    fh.write("field,mean_sage,mean_model,bias,rel_bias_pct,bias_std_spatial,max_abs_diff\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")

# ---------------------------------------------------------------------------
# Per-species Y | T stats + PDF from the accumulated 2-D histograms (no need
# to keep any raw point -- mean/percentiles are read off the (per-T-bin)
# cumulative distribution over Y).
# ---------------------------------------------------------------------------
def hist_mean_and_pct(hist_row, y_centers, y_edges):
    tot = hist_row.sum()
    if tot <= 0:
        return np.nan, np.nan, np.nan
    mean = float((hist_row * y_centers).sum() / tot)
    cdf = np.cumsum(hist_row) / tot
    p16 = float(np.interp(0.16, cdf, y_edges[1:]))
    p84 = float(np.interp(0.84, cdf, y_edges[1:]))
    return mean, p16, p84


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

group = N_T_BINS // N_T_SUPERBINS_PDF
for PDF_FIELD in PDF_FIELDS:
    edges_f = Y_edges[PDF_FIELD]
    Y_centers = 0.5 * (edges_f[:-1] + edges_f[1:])
    hs_full = hist_sage[PDF_FIELD]
    hm_full = hist_model[PDF_FIELD]

    mean_s = np.full(N_T_BINS, np.nan)
    p16_s = np.full(N_T_BINS, np.nan)
    p84_s = np.full(N_T_BINS, np.nan)
    mean_m = np.full(N_T_BINS, np.nan)
    p16_m = np.full(N_T_BINS, np.nan)
    p84_m = np.full(N_T_BINS, np.nan)
    count_s = hs_full.sum(axis=1)
    count_m = hm_full.sum(axis=1)
    for b in range(N_T_BINS):
        mean_s[b], p16_s[b], p84_s[b] = hist_mean_and_pct(hs_full[b], Y_centers, edges_f)
        mean_m[b], p16_m[b], p84_m[b] = hist_mean_and_pct(hm_full[b], Y_centers, edges_f)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(T_centers, mean_s, color=SAGE_COLOR, lw=2, label="SAGE (mean)")
    ax.fill_between(T_centers, p16_s, p84_s, color=SAGE_COLOR, alpha=0.15, label="SAGE 16-84th pct")
    ax.plot(T_centers, mean_m, color=MODEL_COLOR, lw=1.8, ls="--", label="MODEL (mean)")
    ax.fill_between(T_centers, p16_m, p84_m, color=MODEL_COLOR, alpha=0.15, label="MODEL 16-84th pct")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(f"Y_{PDF_FIELD}")
    ax.set_title(f"Y_{PDF_FIELD} conditioned on T -- FULL DOMAIN, {n} frames x {n_cells} cells "
                 f"(REDUIT_T800)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_stats = f"Y{PDF_FIELD}_vs_T_stats_{OUT_PREFIX}.png"
    fig.savefig(out_stats, dpi=140)
    plt.close(fig)

    np.savetxt(
        f"Y{PDF_FIELD}_vs_T_stats_{OUT_PREFIX}.csv",
        np.column_stack([T_centers, mean_s, p16_s, p84_s, count_s, mean_m, p16_m, p84_m, count_m]),
        header="T_center,mean_SAGE,p16_SAGE,p84_SAGE,count_SAGE,mean_MODEL,p16_MODEL,p84_MODEL,count_MODEL",
        delimiter=",", comments="",
    )

    # small-multiples PDF over N_T_SUPERBINS_PDF coarse groups of the fine T bins
    fig, axes = plt.subplots(1, N_T_SUPERBINS_PDF, figsize=(3.1 * N_T_SUPERBINS_PDF, 4.2))
    for b in range(N_T_SUPERBINS_PDF):
        lo_i, hi_i = b * group, (b + 1) * group if b < N_T_SUPERBINS_PDF - 1 else N_T_BINS
        lo_T, hi_T = T_edges[lo_i], T_edges[hi_i]
        hs = hs_full[lo_i:hi_i].sum(axis=0)
        hm = hm_full[lo_i:hi_i].sum(axis=0)
        ax = axes[b]
        ns, nm = hs.sum(), hm.sum()
        bw = edges_f[1] - edges_f[0]
        if ns > 0:
            pdf_s = hs / ns / bw
            ax.stairs(pdf_s, edges_f, fill=True, color=SAGE_COLOR, alpha=0.15)
            ax.stairs(pdf_s, edges_f, color=SAGE_COLOR, lw=2, label="SAGE")
        if nm > 0:
            pdf_m = hm / nm / bw
            ax.stairs(pdf_m, edges_f, fill=True, color=MODEL_COLOR, alpha=0.15)
            ax.stairs(pdf_m, edges_f, color=MODEL_COLOR, lw=2, ls="--", label="MODEL")
        ax.set_title(f"T in [{lo_T:.0f},{hi_T:.0f}]K\nn={int(ns)}/{int(nm)}", fontsize=9)
        ax.set_xlabel(f"Y_{PDF_FIELD}")
        if b == 0:
            ax.set_ylabel("PDF")
            ax.legend(fontsize=8)
    fig.suptitle(f"Y_{PDF_FIELD} PDF by temperature bin -- FULL DOMAIN, SAGE vs MODEL (REDUIT_T800)")
    fig.tight_layout()
    out_pdf = f"Y{PDF_FIELD}_pdf_by_Tbin_{OUT_PREFIX}.png"
    fig.savefig(out_pdf, dpi=130)
    plt.close(fig)
    print(f"{PDF_FIELD}: -> {out_stats}, {out_pdf}", flush=True)

print("done", flush=True)
