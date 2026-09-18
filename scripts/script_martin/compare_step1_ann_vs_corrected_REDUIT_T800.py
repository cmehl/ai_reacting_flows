"""Single-timestep accuracy check, CAS_LEWIS_UNITAIRE/REDUIT_T800, "step_1"
paraview export: SAGE (reference) vs two ANN-accelerated CONVERGE runs that
differ only in the CONVERGE version used (ANN = uncorrected, ANN_corrected =
corrected) -- which of the two ANN outputs tracks SAGE more closely at this
step.

Same style as compare_snapshot_last_REDUIT_T800.py (single post*.h5 triple,
full domain, cheap enough for the login node, no SLURM), but comparing TWO
accelerated runs against SAGE instead of one, and reporting a per-field
winner instead of a stability/divergence check.

Config is top-of-file UPPERCASE vars (no argparse).
"""
import importlib.util
import sys

import numpy as np
from scipy.spatial import cKDTree

# ---- config ----
ANIMATE_SCRIPT = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/scripts/script_martin/animate_ann_sage_hybrid.py"
BASE = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/paraview/step_1"
SAGE_FILE = f"{BASE}/SAGE/post000002_+5.00000e-07.h5"
ANN_FILE = f"{BASE}/ANN/post000002_+5.00000e-07.h5"
ANN_CORRECTED_FILE = f"{BASE}/ANN_corrected/post000002_+5.00000e-07.h5"
MAP_FIELDS = ["Temperature", "H2", "OH", "N", "N2H2", "NNH", "NH", "H", "NO2"]
SLICE_AXIS = "X"
SLICE_HALFWIDTH = 0.005
SLICE_RES = 0.0005
GRID = 320
FILL_RADIUS = 0.006
CLIP_PCT = 99.5
OUT_PREFIX = "REDUIT_T800_step1"
SAGE_COLOR = "#111111"
ANN_COLOR = "#e0311f"
ANN_CORRECTED_COLOR = "#1f77e0"
# -----------------

spec = importlib.util.spec_from_file_location("animate_ann_sage_hybrid", ANIMATE_SCRIPT)
anim = importlib.util.module_from_spec(spec)
sys.modules["animate_ann_sage_hybrid"] = anim
spec.loader.exec_module(anim)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = anim.FIELDS
labels = [anim.field_label(f) for f in FIELDS]

print(f"SAGE:          {SAGE_FILE}")
print(f"ANN:           {ANN_FILE}")
print(f"ANN_corrected: {ANN_CORRECTED_FILE}")
coords_s, data_s, t_s = anim.load(SAGE_FILE, FIELDS)
coords_a, data_a, t_a = anim.load(ANN_FILE, FIELDS)
coords_c, data_c, t_c = anim.load(ANN_CORRECTED_FILE, FIELDS)
n_cells = coords_s.shape[0]
print(f"t_SAGE={t_s:.5e}s  t_ANN={t_a:.5e}s  t_ANN_corrected={t_c:.5e}s  n_cells={n_cells}")
assert abs(t_s - t_a) < 1e-9 and abs(t_s - t_c) < 1e-9, "SAGE/ANN/ANN_corrected times don't match -- check filenames"

tree = cKDTree(coords_s)
dist_a, idx_a = tree.query(coords_a, k=1, workers=-1)
dist_c, idx_c = tree.query(coords_c, k=1, workers=-1)
print(f"max nearest-neighbor dist (cell re-indexing): ANN={dist_a.max():.3e}  ANN_corrected={dist_c.max():.3e}")

# ---------------------------------------------------------------------------
# 1) Per-field error table: RMSE / relative RMSE / bias / max|diff| for both
#    ANN and ANN_corrected against SAGE, plus which one wins on this field.
# ---------------------------------------------------------------------------
rows = []
n_ann_wins = 0
n_corrected_wins = 0
print(f"\n{'field':12s}  {'mean_SAGE':>12s}  {'rmse_ANN':>12s}  {'rel%_ANN':>9s}  "
      f"{'rmse_ANNc':>12s}  {'rel%_ANNc':>9s}  {'max|d|_ANN':>11s}  {'max|d|_ANNc':>11s}  winner")
for fj, f in enumerate(FIELDS):
    name = labels[fj]
    vs = data_s[f].astype(np.float64)
    va = np.empty(n_cells)
    va[idx_a] = data_a[f]
    vc = np.empty(n_cells)
    vc[idx_c] = data_c[f]

    diff_a = va - vs
    diff_c = vc - vs
    mean_s = float(vs.mean())
    denom = abs(mean_s) if abs(mean_s) > 1e-30 else 1e-30

    rmse_a = float(np.sqrt((diff_a ** 2).mean()))
    rmse_c = float(np.sqrt((diff_c ** 2).mean()))
    rel_a = rmse_a / denom * 100
    rel_c = rmse_c / denom * 100
    bias_a = float(diff_a.mean())
    bias_c = float(diff_c.mean())
    max_a = float(np.abs(diff_a).max())
    max_c = float(np.abs(diff_c).max())

    winner = "ANN_corrected" if rmse_c < rmse_a else "ANN"
    if winner == "ANN_corrected":
        n_corrected_wins += 1
    else:
        n_ann_wins += 1

    rows.append((name, mean_s, rmse_a, rel_a, bias_a, max_a, rmse_c, rel_c, bias_c, max_c, winner))
    print(f"{name:12s}  {mean_s:12.5g}  {rmse_a:12.5g}  {rel_a:8.3f}%  "
          f"{rmse_c:12.5g}  {rel_c:8.3f}%  {max_a:11.5g}  {max_c:11.5g}  {winner}")

with open(f"error_table_{OUT_PREFIX}.csv", "w") as fh:
    fh.write("field,mean_sage,rmse_ANN,rel_rmse_ANN_pct,bias_ANN,max_abs_diff_ANN,"
              "rmse_ANN_corrected,rel_rmse_ANN_corrected_pct,bias_ANN_corrected,max_abs_diff_ANN_corrected,winner\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")

t_row = next(r for r in rows if r[0] == "Temperature")
print(f"\n=== SUMMARY (t={t_s:.4e}s, {n_cells} cells) ===")
print(f"Temperature: RMSE ANN={t_row[2]:.4g}K ({t_row[3]:.4f}%)  "
      f"ANN_corrected={t_row[6]:.4g}K ({t_row[7]:.4f}%)  -> {t_row[10]} more accurate on T")
print(f"Field-count winner tally (19 fields): ANN={n_ann_wins}  ANN_corrected={n_corrected_wins}")
overall = "ANN_corrected" if n_corrected_wins > n_ann_wins else ("ANN" if n_ann_wins > n_corrected_wins else "tie")
print(f"Overall (by field-count majority): {overall}")

# ---------------------------------------------------------------------------
# 2) Bar chart: relative RMSE (%) per field, ANN vs ANN_corrected, log scale
#    (species mass fractions span many orders of magnitude).
# ---------------------------------------------------------------------------
rel_a_all = np.array([r[3] for r in rows])
rel_c_all = np.array([r[7] for r in rows])
order = np.argsort(-np.maximum(rel_a_all, rel_c_all))
names_sorted = [rows[i][0] for i in order]
x = np.arange(len(names_sorted))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - w / 2, np.clip(rel_a_all[order], 1e-6, None), width=w, color=ANN_COLOR, label="ANN")
ax.bar(x + w / 2, np.clip(rel_c_all[order], 1e-6, None), width=w, color=ANN_CORRECTED_COLOR, label="ANN_corrected")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(names_sorted, rotation=60, ha="right")
ax.set_ylabel("relative RMSE vs SAGE [%] (log scale)")
ax.set_title(f"Step 1 (t={t_s:.3e}s) -- ANN vs ANN_corrected error vs SAGE, REDUIT_T800")
ax.legend()
ax.grid(True, axis="y", which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"rel_rmse_bars_{OUT_PREFIX}.png", dpi=140)
plt.close(fig)
print(f"\n-> rel_rmse_bars_{OUT_PREFIX}.png")

# ---------------------------------------------------------------------------
# 3) Slice maps (X=0) for the key/trace fields: SAGE / ANN / ANN_corrected
#    values (top row, shared color scale) and ANN-SAGE / ANN_corrected-SAGE
#    error (bottom row, shared symmetric color scale so the two errors are
#    visually comparable).
# ---------------------------------------------------------------------------
axcfg = anim.SLICE_AXES[SLICE_AXIS]
sel = anim.slice_indices(coords_s, axcfg, SLICE_HALFWIDTH, SLICE_RES)
h_sign, v_sign = axcfg["h"][1], axcfg["v"][1]
h_coord = h_sign * coords_s[sel, axcfg["h"][0]]
v_coord = v_sign * coords_s[sel, axcfg["v"][0]]
print(f"slice: {sel.size} cells, {axcfg['h'][2]} in [{h_coord.min():.3f},{h_coord.max():.3f}], "
      f"{axcfg['v'][2]} in [{v_coord.min():.3f},{v_coord.max():.3f}]")

binner = anim.make_binner(h_coord, v_coord, GRID, FILL_RADIUS)
to_img = binner["to_img"]
im_kw = dict(extent=binner["extent"], origin="lower", interpolation="nearest", aspect="equal")

for fname in MAP_FIELDS:
    fj = labels.index(fname)
    f = FIELDS[fj]
    sv = data_s[f][sel].astype(np.float64)
    va_full = np.empty(n_cells)
    va_full[idx_a] = data_a[f]
    av = va_full[sel]
    vc_full = np.empty(n_cells)
    vc_full[idx_c] = data_c[f]
    cv = vc_full[sel]
    diff_a = av - sv
    diff_c = cv - sv

    vmin = float(np.percentile(sv, 100 - CLIP_PCT))
    vmax = float(np.percentile(sv, CLIP_PCT))
    emax = float(np.percentile(np.abs(np.concatenate([diff_a, diff_c])), 99.0)) or 1e-30

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    val_cmap = matplotlib.colormaps["inferno"]
    err_cmap = matplotlib.colormaps["coolwarm"]

    im0 = axes[0, 0].imshow(to_img(sv), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[0, 0].set_title(f"SAGE  {fname}")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(to_img(av), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[0, 1].set_title(f"ANN  {fname}")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(to_img(cv), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[0, 2].set_title(f"ANN_corrected  {fname}")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].axis("off")

    im3 = axes[1, 1].imshow(to_img(diff_a), cmap=err_cmap, vmin=-emax, vmax=emax, **im_kw)
    axes[1, 1].set_title(f"ANN - SAGE  max|d|={np.abs(diff_a).max():.4g}")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    im4 = axes[1, 2].imshow(to_img(diff_c), cmap=err_cmap, vmin=-emax, vmax=emax, **im_kw)
    axes[1, 2].set_title(f"ANN_corrected - SAGE  max|d|={np.abs(diff_c).max():.4g}")
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.set_xlabel(axcfg["h"][2])
        ax.set_ylabel(axcfg["v"][2])
    fig.suptitle(f"{fname} at step 1 (t={t_s:.4e}s) -- REDUIT_T800: SAGE vs ANN vs ANN_corrected")
    fig.tight_layout()
    out = f"slice_{OUT_PREFIX}_{fname}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {fname}: -> {out}")

print("done")
