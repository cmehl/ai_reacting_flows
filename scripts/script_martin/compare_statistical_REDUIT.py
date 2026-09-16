"""Position-independent statistical comparison, SAGE vs MODEL, X=0 slice,
CAS_LEWIS_UNITAIRE/REDUIT.

compare_ann_sage_x0.py's per-cell RMSE (after KDTree-matching MODEL cells
onto SAGE's) answers "is the state at this exact physical point right", which
conflates two different failure modes: the model's *local chemical state* can
be perfectly correct while its *flame position* is shifted by even one AMR
cell -- that alone produces a large per-cell error at the shift front despite
the field being physically fine. This script never matches cells across
sides at all, so a pure position shift cannot show up as an error here:

1. **Wasserstein distance vs time**, per field -- the distributional distance
   between SAGE's and MODEL's OWN value populations on their OWN X=0 slice
   (no cross-mesh alignment). Two slices with the same set of physical states
   arranged differently in space score ~0 here even if per-cell RMSE would be
   huge.
2. **PDF overlay** at the final common timestep, per field -- the same idea,
   visual.
3. **Species-vs-Temperature manifold**: mean +/- std of each species mass
   fraction conditioned on Temperature bin, pooled over every common
   timestep and every slice cell, SAGE vs MODEL. This is the classic
   flamelet/manifold diagnostic (cf. the e_T/e_Sigma diagram already in
   compare_ann_sage_x0.py) -- it asks "does the model reproduce the same
   local composition-temperature relationship", independent of where in
   physical space that state occurs.

Config is top-of-file UPPERCASE vars (no argparse). Run with a bare
`python compare_statistical_REDUIT.py`.
"""
import glob
import os
import re

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ----
SAGE_DIR = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT/SAGE/outputs_original/output"
MODEL_DIR = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT/MODEL_perspecies_2x64_t1e-14_noclust_T600_rollout/outputs_original/output"
MODEL_LABEL = "MODEL"
OUT_DIR = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT/comparison_REDUIT_X0_slice/statistical"
SLICE_HALFWIDTH = 0.005   # slab pre-filter [m], matches the other REDUIT tools
SLICE_RES = 0.0005        # in-plane dedup bucket [m]
T_BIN_EDGES = np.linspace(280.0, 2300.0, 61)   # manifold conditioning bins
STRIDE = 1                # use every Nth common timestep (Wasserstein pass)
# -----------------

os.makedirs(OUT_DIR, exist_ok=True)

SPECIES = [
    "H", "H2", "H2O", "HNO", "HO2", "N", "N2", "N2H2", "N2O", "NH",
    "NH2", "NH3", "NNH", "NO", "NO2", "O", "O2", "OH",
]
FIELDS = ["TEMPERATURE"] + ["MASSFRAC_" + s for s in SPECIES]


def field_label(f):
    return "Temperature" if f == "TEMPERATURE" else f.replace("MASSFRAC_", "")


def load(fp, fields):
    with h5py.File(fp, "r") as f:
        g = f["STREAM_00/CELL_CENTER_DATA"]
        coords = np.stack([g["XCEN_X"][:], g["XCEN_Y"][:], g["XCEN_Z"][:]], axis=1).astype(np.float64)
        data = {k: g[k][:].astype(np.float64) for k in fields}
        t = float(f.attrs["OUTPUT_TIME"][0])
    return coords, data, t


def slice_indices(coords, slab, res):
    """One-cell-thick X=0 slice (slab pre-filter + in-plane dedup on Y,Z),
    computed independently per side -- no cross-mesh matching, by design."""
    c = coords[:, 0]
    cand = np.where(np.abs(c) < slab)[0]
    h, v = coords[cand, 1], coords[cand, 2]
    hb = np.round(h / res).astype(np.int64)
    vb = np.round(v / res).astype(np.int64)
    key = hb * 4_000_003 + vb
    order = np.argsort(np.abs(c[cand]), kind="stable")
    _, first = np.unique(key[order], return_index=True)
    return np.sort(cand[order[first]])


def index_by_time(directory):
    name_re = re.compile(r"^post\d+_(?P<time>[+-][0-9.eE+-]+)\.h5$")
    by_time = {}
    for fp in glob.glob(os.path.join(directory, "post*.h5")):
        m = name_re.match(os.path.basename(fp))
        if m:
            by_time[m.group("time")] = fp
    return by_time


sage_by_time = index_by_time(SAGE_DIR)
model_by_time = index_by_time(MODEL_DIR)
common = sorted(set(sage_by_time) & set(model_by_time), key=float)[::STRIDE]
assert common, "No common timestep between SAGE and MODEL"
print(f"{len(common)} common timesteps, t={float(common[0]):.3e}s to t={float(common[-1]):.3e}s", flush=True)

n_bins = len(T_BIN_EDGES) - 1
manifold = {
    name: {f: dict(sum=np.zeros(n_bins), sumsq=np.zeros(n_bins), count=np.zeros(n_bins))
           for f in FIELDS if f != "TEMPERATURE"}
    for name in ("SAGE", MODEL_LABEL)
}

wass_records = []
last_slice = None

for i, tkey in enumerate(common):
    coords_s, data_s, t_s = load(sage_by_time[tkey], FIELDS)
    coords_m, data_m, t_m = load(model_by_time[tkey], FIELDS)
    assert abs(t_s - t_m) < 1e-9

    sel_s = slice_indices(coords_s, SLICE_HALFWIDTH, SLICE_RES)
    sel_m = slice_indices(coords_m, SLICE_HALFWIDTH, SLICE_RES)

    vals = {"SAGE": {f: data_s[f][sel_s] for f in FIELDS},
            MODEL_LABEL: {f: data_m[f][sel_m] for f in FIELDS}}

    row = dict(timestep=i + 1, time=t_s, n_cells_sage=sel_s.size, n_cells_model=sel_m.size)
    for f in FIELDS:
        row[f"wasserstein_{field_label(f)}"] = float(
            wasserstein_distance(vals["SAGE"][f], vals[MODEL_LABEL][f])
        )
    wass_records.append(row)

    for name in ("SAGE", MODEL_LABEL):
        T = vals[name]["TEMPERATURE"]
        b = np.clip(np.digitize(T, T_BIN_EDGES) - 1, 0, n_bins - 1)
        for f in FIELDS:
            if f == "TEMPERATURE":
                continue
            y = vals[name][f]
            acc = manifold[name][f]
            acc["sum"] += np.bincount(b, weights=y, minlength=n_bins)
            acc["sumsq"] += np.bincount(b, weights=y ** 2, minlength=n_bins)
            acc["count"] += np.bincount(b, minlength=n_bins)

    if i == len(common) - 1:
        last_slice = vals

    if i % 20 == 0 or i == len(common) - 1:
        print(f"  [{i + 1}/{len(common)}] t={t_s:.3e}  "
              f"wasserstein(T)={row['wasserstein_Temperature']:.3e}", flush=True)

df_wass = pd.DataFrame(wass_records)
csv_path = os.path.join(OUT_DIR, "wasserstein_vs_time.csv")
df_wass.to_csv(csv_path, index=False)
print(f"-> {csv_path}", flush=True)

# --- Wasserstein distance vs time, one panel per field ---
n_fields = len(FIELDS)
ncols = 4
nrows = -(-n_fields // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
for j, f in enumerate(FIELDS):
    ax = axes[j // ncols][j % ncols]
    lbl = field_label(f)
    ax.plot(df_wass["time"], df_wass[f"wasserstein_{lbl}"], color="tab:red")
    ax.set_title(lbl, fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
for j in range(n_fields, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")
fig.suptitle(f"Wasserstein distance vs time, SAGE vs {MODEL_LABEL}, X=0 slice "
             "(position-independent -- no cross-mesh cell matching)")
fig.supxlabel("time [s]")
fig.tight_layout()
wass_png = os.path.join(OUT_DIR, "wasserstein_vs_time.png")
fig.savefig(wass_png, dpi=150)
plt.close(fig)
print(f"-> {wass_png}", flush=True)

# --- PDF overlay at the final common timestep ---
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
for j, f in enumerate(FIELDS):
    ax = axes[j // ncols][j % ncols]
    lbl = field_label(f)
    sv, mv = last_slice["SAGE"][f], last_slice[MODEL_LABEL][f]
    lo, hi = min(sv.min(), mv.min()), max(sv.max(), mv.max())
    if hi <= lo:
        hi = lo + 1e-30
    bins = np.linspace(lo, hi, 50)
    ax.hist(sv, bins=bins, density=True, histtype="step", color="black", lw=1.6, label="SAGE")
    ax.hist(mv, bins=bins, density=True, histtype="step", color="tab:blue", lw=1.2, label=MODEL_LABEL)
    ax.set_title(lbl, fontsize=9)
    ax.set_yscale("log")
    if j == 0:
        ax.legend(fontsize=7)
for j in range(n_fields, nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")
fig.suptitle(f"PDF overlay at t={df_wass['time'].iloc[-1]:.3e}s, SAGE vs {MODEL_LABEL}, X=0 slice")
fig.tight_layout()
pdf_png = os.path.join(OUT_DIR, "pdf_overlay_final.png")
fig.savefig(pdf_png, dpi=150)
plt.close(fig)
print(f"-> {pdf_png}", flush=True)

# --- Species-vs-Temperature manifold, pooled over all timesteps/cells ---
t_centers = 0.5 * (T_BIN_EDGES[:-1] + T_BIN_EDGES[1:])
manifold_records = []
species_fields = [f for f in FIELDS if f != "TEMPERATURE"]
nrows_m = -(-len(species_fields) // ncols)
fig, axes = plt.subplots(nrows_m, ncols, figsize=(4.2 * ncols, 3.4 * nrows_m), squeeze=False)
for j, f in enumerate(species_fields):
    lbl = field_label(f)
    ax = axes[j // ncols][j % ncols]
    means = {}
    for name, color in (("SAGE", "black"), (MODEL_LABEL, "tab:blue")):
        acc = manifold[name][f]
        cnt = acc["count"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = acc["sum"] / cnt
            std = np.sqrt(np.maximum(acc["sumsq"] / cnt - mean ** 2, 0.0))
        means[name] = mean
        valid = cnt > 0
        ax.plot(t_centers[valid], mean[valid], color=color, lw=1.6, label=name)
        ax.fill_between(t_centers[valid], (mean - std)[valid], (mean + std)[valid], color=color, alpha=0.15)
        for k in range(n_bins):
            manifold_records.append(dict(field=lbl, model=name, T_bin_center=t_centers[k],
                                          mean=mean[k], std=std[k], count=int(cnt[k])))
    valid_both = (manifold["SAGE"][f]["count"] > 0) & (manifold[MODEL_LABEL][f]["count"] > 0)
    if valid_both.any():
        err = np.sqrt(np.mean((means[MODEL_LABEL][valid_both] - means["SAGE"][valid_both]) ** 2))
        ax.set_title(f"{lbl}  (manifold RMSE={err:.2e})", fontsize=9)
    else:
        ax.set_title(lbl, fontsize=9)
    ax.set_yscale("log")
    if j == 0:
        ax.legend(fontsize=7)
for j in range(len(species_fields), nrows_m * ncols):
    axes[j // ncols][j % ncols].axis("off")
fig.suptitle(f"Species mass fraction vs Temperature (pooled over {len(common)} timesteps, "
             f"X=0 slice), SAGE vs {MODEL_LABEL} -- position-independent flamelet-manifold check")
fig.supxlabel("Temperature [K]")
fig.tight_layout()
manifold_png = os.path.join(OUT_DIR, "manifold_Y_vs_T.png")
fig.savefig(manifold_png, dpi=150)
plt.close(fig)
print(f"-> {manifold_png}", flush=True)

df_manifold = pd.DataFrame(manifold_records)
manifold_csv = os.path.join(OUT_DIR, "manifold_Y_vs_T.csv")
df_manifold.to_csv(manifold_csv, index=False)
print(f"-> {manifold_csv}", flush=True)

print("\nDone.", flush=True)
