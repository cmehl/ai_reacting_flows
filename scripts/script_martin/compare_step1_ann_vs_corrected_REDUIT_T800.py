"""Single-timestep accuracy check, CAS_LEWIS_UNITAIRE/REDUIT_T800, "step_1"
paraview export: SAGE (reference) vs N ANN-accelerated CONVERGE runs that
differ only in the CONVERGE version used (ANN = uncorrected, ANN_corrected /
ANN_corrected_2 / ... = successive correction attempts) -- which of the ANN
outputs tracks SAGE more closely at this step.

Same style as compare_snapshot_last_REDUIT_T800.py (single post*.h5 set,
full domain, cheap enough for the login node, no SLURM), but comparing N
accelerated runs against SAGE instead of one, and reporting a per-field
winner instead of a stability/divergence check. Generalized from an
earlier 2-variant (ANN vs ANN_corrected) version to any number of variants
via the VARIANTS list below.

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
VARIANTS = [
    ("ANN", f"{BASE}/ANN/post000002_+5.00000e-07.h5", "#e0311f"),
    ("ANN_corrected", f"{BASE}/ANN_corrected/post000002_+5.00000e-07.h5", "#1f77e0"),
    ("ANN_corrected_2", f"{BASE}/ANN_corrected_2/post000002_+5.00000e-07.h5", "#2fa84a"),
]
MAP_FIELDS = ["Temperature", "H2", "OH", "N", "N2H2", "NNH", "NH", "H", "NO2"]
SLICE_AXIS = "X"
SLICE_HALFWIDTH = 0.005
SLICE_RES = 0.0005
GRID = 320
FILL_RADIUS = 0.006
CLIP_PCT = 99.5
OUT_PREFIX = "REDUIT_T800_step1"
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
n_variants = len(VARIANTS)

print(f"SAGE: {SAGE_FILE}")
coords_s, data_s, t_s = anim.load(SAGE_FILE, FIELDS)
n_cells = coords_s.shape[0]

tree = cKDTree(coords_s)
variant_data = {}  # label -> (data, idx, dist)
for label, fp, color in VARIANTS:
    print(f"{label}: {fp}")
    coords_v, data_v, t_v = anim.load(fp, FIELDS)
    assert abs(t_v - t_s) < 1e-9, f"{label} time {t_v} != SAGE time {t_s}"
    dist_v, idx_v = tree.query(coords_v, k=1, workers=-1)
    variant_data[label] = (data_v, idx_v, dist_v)
    print(f"  max nearest-neighbor dist (cell re-indexing) = {dist_v.max():.3e}")
print(f"t_SAGE={t_s:.5e}s  n_cells={n_cells}  n_variants={n_variants}")

# ---------------------------------------------------------------------------
# 1) Per-field error table: RMSE / relative RMSE / bias / max|diff| for each
#    variant against SAGE, plus which variant wins on this field.
# ---------------------------------------------------------------------------
rows = []
win_counts = {label: 0 for label, _, _ in VARIANTS}
header_line = f"{'field':12s}  {'mean_SAGE':>12s}" + "".join(
    f"  {'rmse_'+label:>14s}  {'rel%_'+label:>9s}" for label, _, _ in VARIANTS
) + "  winner"
print(f"\n{header_line}")
for fj, f in enumerate(FIELDS):
    name = labels[fj]
    vs = data_s[f].astype(np.float64)
    mean_s = float(vs.mean())
    denom = abs(mean_s) if abs(mean_s) > 1e-30 else 1e-30

    per_variant = {}
    for label, fp, color in VARIANTS:
        data_v, idx_v, _ = variant_data[label]
        vv = np.empty(n_cells)
        vv[idx_v] = data_v[f]
        diff = vv - vs
        rmse = float(np.sqrt((diff ** 2).mean()))
        rel = rmse / denom * 100
        bias = float(diff.mean())
        maxabs = float(np.abs(diff).max())
        per_variant[label] = dict(rmse=rmse, rel=rel, bias=bias, max=maxabs)

    winner = min(per_variant, key=lambda l: per_variant[l]["rmse"])
    win_counts[winner] += 1

    rows.append((name, mean_s, per_variant, winner))
    line = f"{name:12s}  {mean_s:12.5g}" + "".join(
        f"  {per_variant[label]['rmse']:14.5g}  {per_variant[label]['rel']:8.3f}%" for label, _, _ in VARIANTS
    ) + f"  {winner}"
    print(line)

with open(f"error_table_{OUT_PREFIX}.csv", "w") as fh:
    cols = ["field", "mean_sage"]
    for label, _, _ in VARIANTS:
        cols += [f"rmse_{label}", f"rel_rmse_{label}_pct", f"bias_{label}", f"max_abs_diff_{label}"]
    cols.append("winner")
    fh.write(",".join(cols) + "\n")
    for name, mean_s, per_variant, winner in rows:
        vals = [name, mean_s]
        for label, _, _ in VARIANTS:
            pv = per_variant[label]
            vals += [pv["rmse"], pv["rel"], pv["bias"], pv["max"]]
        vals.append(winner)
        fh.write(",".join(str(x) for x in vals) + "\n")

t_row = next(r for r in rows if r[0] == "Temperature")
print(f"\n=== SUMMARY (t={t_s:.4e}s, {n_cells} cells) ===")
t_line = "  ".join(f"{label}={t_row[2][label]['rmse']:.4g}K ({t_row[2][label]['rel']:.4f}%)" for label, _, _ in VARIANTS)
print(f"Temperature: RMSE {t_line}  -> {t_row[3]} most accurate on T")
print(f"Field-count winner tally ({len(FIELDS)} fields): " + "  ".join(f"{k}={v}" for k, v in win_counts.items()))
overall = max(win_counts, key=win_counts.get)
print(f"Overall (by field-count majority): {overall}")

# ---------------------------------------------------------------------------
# 2) Bar chart: relative RMSE (%) per field, one group of bars per variant,
#    log scale (species mass fractions span many orders of magnitude).
# ---------------------------------------------------------------------------
rel_by_variant = {label: np.array([r[2][label]["rel"] for r in rows]) for label, _, _ in VARIANTS}
worst = np.max(np.column_stack(list(rel_by_variant.values())), axis=1)
order = np.argsort(-worst)
names_sorted = [rows[i][0] for i in order]
x = np.arange(len(names_sorted))
w = 0.8 / n_variants

fig, ax = plt.subplots(figsize=(14, 6))
for k, (label, _, color) in enumerate(VARIANTS):
    offset = (k - (n_variants - 1) / 2) * w
    ax.bar(x + offset, np.clip(rel_by_variant[label][order], 1e-6, None), width=w, color=color, label=label)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(names_sorted, rotation=60, ha="right")
ax.set_ylabel("relative RMSE vs SAGE [%] (log scale)")
ax.set_title(f"Step 1 (t={t_s:.3e}s) -- {n_variants} ANN variants error vs SAGE, REDUIT_T800")
ax.legend()
ax.grid(True, axis="y", which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"rel_rmse_bars_{OUT_PREFIX}.png", dpi=140)
plt.close(fig)
print(f"\n-> rel_rmse_bars_{OUT_PREFIX}.png")

# ---------------------------------------------------------------------------
# 3) Slice maps (X=0) for the key/trace fields: SAGE + each variant's value
#    (top row, shared color scale) and each variant's error vs SAGE (bottom
#    row, shared symmetric color scale so all variants are visually
#    comparable).
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
ncols = 1 + n_variants

for fname in MAP_FIELDS:
    fj = labels.index(fname)
    f = FIELDS[fj]
    sv = data_s[f][sel].astype(np.float64)

    variant_vals = {}
    variant_diffs = {}
    for label, _, _ in VARIANTS:
        data_v, idx_v, _ = variant_data[label]
        vv_full = np.empty(n_cells)
        vv_full[idx_v] = data_v[f]
        vv = vv_full[sel]
        variant_vals[label] = vv
        variant_diffs[label] = vv - sv

    vmin = float(np.percentile(sv, 100 - CLIP_PCT))
    vmax = float(np.percentile(sv, CLIP_PCT))
    all_diffs = np.concatenate(list(variant_diffs.values()))
    emax = float(np.percentile(np.abs(all_diffs), 99.0)) or 1e-30

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 9))
    val_cmap = matplotlib.colormaps["inferno"]
    err_cmap = matplotlib.colormaps["coolwarm"]

    im0 = axes[0, 0].imshow(to_img(sv), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[0, 0].set_title(f"SAGE  {fname}")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    axes[1, 0].axis("off")

    for k, (label, _, _) in enumerate(VARIANTS):
        col = 1 + k
        im_v = axes[0, col].imshow(to_img(variant_vals[label]), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
        axes[0, col].set_title(f"{label}  {fname}")
        plt.colorbar(im_v, ax=axes[0, col], fraction=0.046)

        diff = variant_diffs[label]
        im_e = axes[1, col].imshow(to_img(diff), cmap=err_cmap, vmin=-emax, vmax=emax, **im_kw)
        axes[1, col].set_title(f"{label} - SAGE  max|d|={np.abs(diff).max():.4g}")
        plt.colorbar(im_e, ax=axes[1, col], fraction=0.046)

    for ax in axes.flat:
        ax.set_xlabel(axcfg["h"][2])
        ax.set_ylabel(axcfg["v"][2])
    fig.suptitle(f"{fname} at step 1 (t={t_s:.4e}s) -- REDUIT_T800: SAGE vs {n_variants} ANN variants")
    fig.tight_layout()
    out = f"slice_{OUT_PREFIX}_{fname}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {fname}: -> {out}")

print("done")
