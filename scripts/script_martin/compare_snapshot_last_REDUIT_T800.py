"""Single-timestep stability check, SAGE vs MODEL, CAS_LEWIS_UNITAIRE/REDUIT_T800,
at MODEL's own LAST available snapshot -- the ANN-accelerated CONVERGE run
(job 2820338) logged repeated "Temperature extrapolation error" warnings
right before being cancelled at t=8.985e-4s (post001798). This checks
whether the computation was still tracking SAGE at the point it stopped, or
had already diverged, by comparing the exact same timestep on both sides
(SAGE has the identical filename/time available, it kept running much
further).

Unlike compare_mean_pdf_full_REDUIT_T800.py (pools 100 frames -- chaos
cancels in a time average), a single snapshot has no time dimension to
average over: this is a direct stability/divergence check at one instant,
not a chaos-vs-bias check. Full domain (10,690,040 cells), reads exactly
one post*.h5 pair -- cheap enough to run directly on the login node, no
SLURM needed.

Config is top-of-file UPPERCASE vars (no argparse).
"""
import importlib.util
import sys

import numpy as np
from scipy.spatial import cKDTree

# ---- config ----
ANIMATE_SCRIPT = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/scripts/script_martin/animate_ann_sage_hybrid.py"
SAGE_FILE = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/SAGE/outputs_original/output/post001798_+8.98500e-04.h5"
MODEL_FILE = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/MODEL_perspecies_2x64_thresh1e14_noclust_T800/outputs_original/output/post001798_+8.98500e-04.h5"
MAP_FIELDS = ["Temperature", "H2", "OH", "N", "N2H2", "NNH", "NH", "H"]  # slice maps
PDF_FIELDS = ["H", "H2", "H2O", "HNO", "HO2", "N", "N2", "N2H2", "N2O", "NH",
              "NH2", "NH3", "NNH", "NO", "NO2", "O", "O2", "OH"]  # all 18 species
N_T_BINS_STATS = 40
N_T_BINS_PDF = 6
N_Y_BINS = 150
SLICE_AXIS = "X"
SLICE_HALFWIDTH = 0.005
SLICE_RES = 0.0005
GRID = 320
FILL_RADIUS = 0.006
CLIP_PCT = 99.5
OUT_PREFIX = "REDUIT_T800_last"
SAGE_COLOR = "#111111"
MODEL_COLOR = "#e0311f"
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

print(f"SAGE:  {SAGE_FILE}")
print(f"MODEL: {MODEL_FILE}")
coords_s, data_s, t_s = anim.load(SAGE_FILE, FIELDS)
coords_m, data_m, t_m = anim.load(MODEL_FILE, FIELDS)
n_cells = coords_s.shape[0]
print(f"t_SAGE={t_s:.5e}s  t_MODEL={t_m:.5e}s  (delta={abs(t_s - t_m):.2e}s)  n_cells={n_cells}")
assert abs(t_s - t_m) < 1e-9, "SAGE/MODEL times don't match -- check filenames"

tree = cKDTree(coords_s)
dist, idx = tree.query(coords_m, k=1, workers=-1)
print(f"max nearest-neighbor dist (cell re-indexing) = {dist.max():.3e}")

# ---------------------------------------------------------------------------
# 1) Full-domain bias table at this single instant -- domain-mean, spatial
#    std, and max|diff| (the one that matters most here: a single blown-up
#    cell reveals an incipient divergence even if the mean looks fine).
# ---------------------------------------------------------------------------
rows = []
print("\nField                domain-mean SAGE   domain-mean MODEL   bias        rel.bias   spatial std    max|diff|")
for fj, f in enumerate(FIELDS):
    name = labels[fj]
    vs = data_s[f].astype(np.float64)
    vm = np.empty(n_cells)
    vm[idx] = data_m[f]
    diff = vm - vs
    dom_s = float(vs.mean())
    dom_m = float(vm.mean())
    bias = float(diff.mean())
    bias_std = float(diff.std())
    maxabs = float(np.abs(diff).max())
    rel = bias / (dom_s or 1e-30) * 100
    rows.append((name, dom_s, dom_m, bias, rel, bias_std, maxabs))
    print(f"{name:12s}  {dom_s:16.6g}  {dom_m:16.6g}  {bias:10.4g}  {rel:+8.4f}%  {bias_std:10.4g}  {maxabs:10.4g}")

with open(f"bias_table_{OUT_PREFIX}.csv", "w") as fh:
    fh.write("field,mean_sage,mean_model,bias,rel_bias_pct,bias_std_spatial,max_abs_diff\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")

# ---------------------------------------------------------------------------
# 2) Slice (X=0) maps at this single instant for a handful of key fields --
#    shows WHERE any divergence is, unlike the table's single numbers.
# ---------------------------------------------------------------------------
axcfg = anim.SLICE_AXES[SLICE_AXIS]
sel = anim.slice_indices(coords_s, axcfg, SLICE_HALFWIDTH, SLICE_RES)
h_sign, v_sign = axcfg["h"][1], axcfg["v"][1]
h_coord = h_sign * coords_s[sel, axcfg["h"][0]]
v_coord = v_sign * coords_s[sel, axcfg["v"][0]]
print(f"\nslice: {sel.size} cells, {axcfg['h'][2]} in [{h_coord.min():.3f},{h_coord.max():.3f}], "
      f"{axcfg['v'][2]} in [{v_coord.min():.3f},{v_coord.max():.3f}]")

binner = anim.make_binner(h_coord, v_coord, GRID, FILL_RADIUS)
to_img = binner["to_img"]
im_kw = dict(extent=binner["extent"], origin="lower", interpolation="nearest", aspect="equal")

for fname in MAP_FIELDS:
    fj = labels.index(fname)
    f = FIELDS[fj]
    sv = data_s[f][sel].astype(np.float64)
    vm_full = np.empty(n_cells)
    vm_full[idx] = data_m[f]
    mv = vm_full[sel]
    diff = mv - sv

    vmin = float(np.percentile(sv, 100 - CLIP_PCT))
    vmax = float(np.percentile(sv, CLIP_PCT))
    emax = float(np.percentile(np.abs(diff), 99.0)) or 1e-30

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    val_cmap = matplotlib.colormaps["inferno"]
    err_cmap = matplotlib.colormaps["coolwarm"]

    im0 = axes[0].imshow(to_img(sv), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[0].set_title(f"SAGE  {fname}  (t={t_s:.3e}s)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(to_img(mv), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[1].set_title(f"MODEL  {fname}  (last frame before cancel)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(to_img(diff), cmap=err_cmap, vmin=-emax, vmax=emax, **im_kw)
    axes[2].set_title(f"MODEL - SAGE  max|diff|={np.abs(diff).max():.4g}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("-Z [m]")
        ax.set_ylabel("Y [m]")
    fig.suptitle(f"{fname} at MODEL's last snapshot before cancel (t={t_s:.4e}s) -- REDUIT_T800")
    fig.tight_layout()
    out = f"snapshot_{OUT_PREFIX}_{fname}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {fname}: -> {out}")

# ---------------------------------------------------------------------------
# 3) Per-species Y vs T at this single instant, full domain (10.69M points,
#    plenty of statistics from one frame alone) -- a state-space stability
#    check: has MODEL drifted into a different (Y|T) relationship than SAGE?
# ---------------------------------------------------------------------------
tj = labels.index("Temperature")
T_s = data_s["TEMPERATURE"].astype(np.float64)
T_m = np.empty(n_cells)
T_m[idx] = data_m["TEMPERATURE"]
t_lo, t_hi = min(T_s.min(), T_m.min()), max(T_s.max(), T_m.max())
edges = np.linspace(t_lo, t_hi, N_T_BINS_STATS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
pdf_edges = np.linspace(t_lo, t_hi, N_T_BINS_PDF + 1)
print(f"\nT range at this snapshot: SAGE [{T_s.min():.1f},{T_s.max():.1f}]K  "
      f"MODEL [{T_m.min():.1f},{T_m.max():.1f}]K")


def binned_stats(T, Y, edges):
    idxb = np.digitize(T, edges) - 1
    n = len(edges) - 1
    mean = np.full(n, np.nan)
    p16 = np.full(n, np.nan)
    p84 = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int64)
    for b in range(n):
        m = idxb == b
        count[b] = m.sum()
        if count[b] > 0:
            vals = Y[m]
            mean[b] = vals.mean()
            p16[b] = np.percentile(vals, 16)
            p84[b] = np.percentile(vals, 84)
    return mean, p16, p84, count


for PDF_FIELD in PDF_FIELDS:
    pj = labels.index(PDF_FIELD)
    f = FIELDS[pj]
    Y_s = data_s[f].astype(np.float64)
    Y_m = np.empty(n_cells)
    Y_m[idx] = data_m[f]

    mean_s, p16_s, p84_s, cnt_s = binned_stats(T_s, Y_s, edges)
    mean_m, p16_m, p84_m, cnt_m = binned_stats(T_m, Y_m, edges)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(centers, mean_s, color=SAGE_COLOR, lw=2, label="SAGE (mean)")
    ax.fill_between(centers, p16_s, p84_s, color=SAGE_COLOR, alpha=0.15, label="SAGE 16-84th pct")
    ax.plot(centers, mean_m, color=MODEL_COLOR, lw=1.8, ls="--", label="MODEL (mean)")
    ax.fill_between(centers, p16_m, p84_m, color=MODEL_COLOR, alpha=0.15, label="MODEL 16-84th pct")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(f"Y_{PDF_FIELD}")
    ax.set_title(f"Y_{PDF_FIELD} vs T -- SINGLE SNAPSHOT t={t_s:.4e}s, {n_cells} cells (REDUIT_T800)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_stats = f"Y{PDF_FIELD}_vs_T_{OUT_PREFIX}.png"
    fig.savefig(out_stats, dpi=140)
    plt.close(fig)

    np.savetxt(
        f"Y{PDF_FIELD}_vs_T_{OUT_PREFIX}.csv",
        np.column_stack([centers, mean_s, p16_s, p84_s, cnt_s, mean_m, p16_m, p84_m, cnt_m]),
        header="T_center,mean_SAGE,p16_SAGE,p84_SAGE,count_SAGE,mean_MODEL,p16_MODEL,p84_MODEL,count_MODEL",
        delimiter=",", comments="",
    )

    fig, axes = plt.subplots(1, N_T_BINS_PDF, figsize=(3.1 * N_T_BINS_PDF, 4.2))
    for b in range(N_T_BINS_PDF):
        lo, hi = pdf_edges[b], pdf_edges[b + 1]
        ax = axes[b]
        ms = (T_s >= lo) & (T_s < hi)
        mm = (T_m >= lo) & (T_m < hi)
        ys, ym = Y_s[ms], Y_m[mm]
        if ys.size > 10 and ym.size > 10:
            lo_y = min(ys.min(), ym.min())
            hi_y = max(ys.max(), ym.max())
            bins = np.linspace(lo_y, hi_y, 40) if hi_y > lo_y else 40
            hs, hedges = np.histogram(ys, bins=bins, density=True)
            hm, _ = np.histogram(ym, bins=hedges, density=True)
            ax.stairs(hs, hedges, fill=True, color=SAGE_COLOR, alpha=0.15)
            ax.stairs(hs, hedges, color=SAGE_COLOR, lw=2, label="SAGE")
            ax.stairs(hm, hedges, fill=True, color=MODEL_COLOR, alpha=0.15)
            ax.stairs(hm, hedges, color=MODEL_COLOR, lw=2, ls="--", label="MODEL")
        ax.set_title(f"T in [{lo:.0f},{hi:.0f}]K\nn={ms.sum()}/{mm.sum()}", fontsize=9)
        ax.set_xlabel(f"Y_{PDF_FIELD}")
        if b == 0:
            ax.set_ylabel("PDF")
            ax.legend(fontsize=8)
    fig.suptitle(f"Y_{PDF_FIELD} PDF by T bin -- SINGLE SNAPSHOT t={t_s:.4e}s, SAGE vs MODEL (REDUIT_T800)")
    fig.tight_layout()
    out_pdf = f"Y{PDF_FIELD}_pdf_by_Tbin_{OUT_PREFIX}.png"
    fig.savefig(out_pdf, dpi=130)
    plt.close(fig)
    print(f"{PDF_FIELD}: -> {out_stats}, {out_pdf}")

print("done")
