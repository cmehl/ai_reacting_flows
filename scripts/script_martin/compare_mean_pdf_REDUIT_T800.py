"""Time-averaged fields + temperature-binned Y_H2 statistics/PDF, SAGE vs MODEL,
CAS_LEWIS_UNITAIRE/REDUIT_T800 -- distinguishes frame-to-frame chaos (averages
out) from a systematic bias (survives the average / shows up as a shifted
mean-Y_H2(T) curve or PDF).

Reads the slice cache already built by animate_ann_sage_hybrid.py
(--cache-only run, job 2820374) -- no HDF5 reload needed. Cache arrays are
[n_fields, n_frames, n_cells] float32, one cell value per slice cell (X=0,
100 shared timesteps, t=0->4.95e-5s).

Config is top-of-file UPPERCASE vars (no argparse). Run with a bare
`python compare_mean_pdf_REDUIT_T800.py` (needs ~2GB RAM, no SLURM needed).
"""
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- config ----
CACHE = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/slice_cache_X0_100steps.npz"
ANIMATE_SCRIPT = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/scripts/script_martin/animate_ann_sage_hybrid.py"
MEAN_FIELDS = ["Temperature", "H2", "OH", "N", "N2H2"]  # time-averaged spatial maps
PDF_FIELDS = ["H", "H2", "H2O", "HNO", "HO2", "N", "N2", "N2H2", "N2O", "NH",
              "NH2", "NH3", "NNH", "NO", "NO2", "O", "O2", "OH"]  # all 18 species
N_T_BINS_STATS = 40       # bins for the mean-Y(T) curve
N_T_BINS_PDF = 6          # coarser bins for the small-multiples PDF-by-bin figure
GRID = 320
FILL_RADIUS = 0.006
CLIP_PCT = 99.5
OUT_PREFIX = "REDUIT_T800"
SAGE_COLOR = "#111111"    # near-black, distinct from MODEL_COLOR even where curves overlap
MODEL_COLOR = "#e0311f"   # vivid red -- high contrast vs black, still readable in both themes
# -----------------

spec = importlib.util.spec_from_file_location("animate_ann_sage_hybrid", ANIMATE_SCRIPT)
anim = importlib.util.module_from_spec(spec)
sys.modules["animate_ann_sage_hybrid"] = anim
spec.loader.exec_module(anim)

print(f"Loading cache {CACHE} ...", flush=True)
cache = np.load(CACHE)
labels = [str(l) for l in cache["labels"]]
sage = cache["sage"]          # [n_fields, n_frames, n_cells]
model = cache["model_0"]
h_coord = cache["h_coord"]
v_coord = cache["v_coord"]
times = cache["times"]
model_label = str(cache["model_labels"][0])
n_fields, n_frames, n_cells = sage.shape
print(f"  {n_fields} fields, {n_frames} frames (t={times[0]:.3e}..{times[-1]:.3e}s), "
      f"{n_cells} cells, model label = {model_label!r}", flush=True)

binner = anim.make_binner(h_coord, v_coord, GRID, FILL_RADIUS)
to_img = binner["to_img"]
im_kw = dict(extent=binner["extent"], origin="lower", interpolation="nearest", aspect="equal")

# ---------------------------------------------------------------------------
# 1) Time-averaged spatial fields: chaos cancels in the mean, a systematic
#    bias (MODEL-SAGE) does not.
# ---------------------------------------------------------------------------
for fname in MEAN_FIELDS:
    fj = labels.index(fname)
    mean_sage = sage[fj].mean(axis=0)
    mean_model = model[fj].mean(axis=0)
    diff = mean_model - mean_sage

    vmin = float(np.percentile(mean_sage, 100 - CLIP_PCT))
    vmax = float(np.percentile(mean_sage, CLIP_PCT))
    emax = float(np.percentile(np.abs(diff), 99.0)) or 1e-30

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    val_cmap = matplotlib.colormaps["inferno"]
    err_cmap = matplotlib.colormaps["coolwarm"]

    im0 = axes[0].imshow(to_img(mean_sage), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[0].set_title(f"SAGE  <{fname}>_t")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(to_img(mean_model), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    axes[1].set_title(f"{model_label}  <{fname}>_t")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(to_img(diff), cmap=err_cmap, vmin=-emax, vmax=emax, **im_kw)
    axes[2].set_title(f"{model_label} - SAGE  (bias)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("-Z [m]")
        ax.set_ylabel("Y [m]")
    fig.suptitle(f"Time-averaged {fname} over {n_frames} frames "
                 f"(t={times[0]:.2e}-{times[-1]:.2e}s) -- REDUIT_T800")
    fig.tight_layout()
    out = f"mean_field_{OUT_PREFIX}_{fname}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    bias = float(diff.mean())
    bias_std = float(diff.std())
    rel = bias / (float(mean_sage.mean()) or 1e-30) * 100
    print(f"{fname}: domain-mean SAGE={mean_sage.mean():.4g}  {model_label}={mean_model.mean():.4g}  "
          f"bias(mean diff)={bias:.4g} ({rel:+.3f}%)  bias std over space={bias_std:.4g}  "
          f"max|diff|={np.abs(diff).max():.4g}  -> {out}", flush=True)

# ---------------------------------------------------------------------------
# 2) & 3) Per-species Y statistics conditioned on Temperature, pooling all
#    cells x all frames -- removes space/time structure entirely, isolates
#    the state-space (Y | T) relationship itself -- plus the full PDF within
#    a handful of coarser T-bins for each species.
# ---------------------------------------------------------------------------
tj = labels.index("Temperature")
T_sage = sage[tj].ravel().astype(np.float64)
T_model = model[tj].ravel().astype(np.float64)
t_lo = min(T_sage.min(), T_model.min())
t_hi = max(T_sage.max(), T_model.max())
edges = np.linspace(t_lo, t_hi, N_T_BINS_STATS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
pdf_edges = np.linspace(t_lo, t_hi, N_T_BINS_PDF + 1)


def binned_stats(T, Y, edges):
    idx = np.digitize(T, edges) - 1
    n = len(edges) - 1
    mean = np.full(n, np.nan)
    p16 = np.full(n, np.nan)
    p84 = np.full(n, np.nan)
    count = np.zeros(n, dtype=np.int64)
    for b in range(n):
        m = idx == b
        count[b] = m.sum()
        if count[b] > 0:
            vals = Y[m]
            mean[b] = vals.mean()
            p16[b] = np.percentile(vals, 16)
            p84[b] = np.percentile(vals, 84)
    return mean, p16, p84, count


for PDF_FIELD in PDF_FIELDS:
    pj = labels.index(PDF_FIELD)
    Y_sage = sage[pj].ravel().astype(np.float64)
    Y_model = model[pj].ravel().astype(np.float64)

    mean_s, p16_s, p84_s, cnt_s = binned_stats(T_sage, Y_sage, edges)
    mean_m, p16_m, p84_m, cnt_m = binned_stats(T_model, Y_model, edges)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(centers, mean_s, color=SAGE_COLOR, lw=2, label="SAGE (mean)")
    ax.fill_between(centers, p16_s, p84_s, color=SAGE_COLOR, alpha=0.15, label="SAGE 16-84th pct")
    ax.plot(centers, mean_m, color=MODEL_COLOR, lw=1.8, ls="--", label=f"{model_label} (mean)")
    ax.fill_between(centers, p16_m, p84_m, color=MODEL_COLOR, alpha=0.15, label=f"{model_label} 16-84th pct")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(f"Y_{PDF_FIELD}")
    ax.set_title(f"Y_{PDF_FIELD} conditioned on T -- pooled over {n_frames} frames x {n_cells} cells "
                 f"(REDUIT_T800)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_stats = f"Y{PDF_FIELD}_vs_T_stats_{OUT_PREFIX}.png"
    fig.savefig(out_stats, dpi=140)
    plt.close(fig)

    np.savetxt(
        f"Y{PDF_FIELD}_vs_T_stats_{OUT_PREFIX}.csv",
        np.column_stack([centers, mean_s, p16_s, p84_s, cnt_s, mean_m, p16_m, p84_m, cnt_m]),
        header="T_center,mean_SAGE,p16_SAGE,p84_SAGE,count_SAGE,mean_MODEL,p16_MODEL,p84_MODEL,count_MODEL",
        delimiter=",", comments="",
    )

    fig, axes = plt.subplots(1, N_T_BINS_PDF, figsize=(3.1 * N_T_BINS_PDF, 4.2), sharey=False)
    for b in range(N_T_BINS_PDF):
        lo, hi = pdf_edges[b], pdf_edges[b + 1]
        ax = axes[b]
        ms = (T_sage >= lo) & (T_sage < hi)
        mm = (T_model >= lo) & (T_model < hi)
        ys, ym = Y_sage[ms], Y_model[mm]
        if ys.size > 10 and ym.size > 10:
            lo_y = min(ys.min(), ym.min())
            hi_y = max(ys.max(), ym.max())
            bins = np.linspace(lo_y, hi_y, 40) if hi_y > lo_y else 40
            hs, hedges = np.histogram(ys, bins=bins, density=True)
            hm, _ = np.histogram(ym, bins=hedges, density=True)
            ax.stairs(hs, hedges, fill=True, color=SAGE_COLOR, alpha=0.15)
            ax.stairs(hs, hedges, color=SAGE_COLOR, lw=2, label="SAGE")
            ax.stairs(hm, hedges, fill=True, color=MODEL_COLOR, alpha=0.15)
            ax.stairs(hm, hedges, color=MODEL_COLOR, lw=2, ls="--", label=model_label)
        ax.set_title(f"T in [{lo:.0f},{hi:.0f}]K\nn={ms.sum()}/{mm.sum()}", fontsize=9)
        ax.set_xlabel(f"Y_{PDF_FIELD}")
        if b == 0:
            ax.set_ylabel("PDF")
            ax.legend(fontsize=8)
    fig.suptitle(f"Y_{PDF_FIELD} PDF by temperature bin -- SAGE vs {model_label} (REDUIT_T800)")
    fig.tight_layout()
    out_pdf = f"Y{PDF_FIELD}_pdf_by_Tbin_{OUT_PREFIX}.png"
    fig.savefig(out_pdf, dpi=130)
    plt.close(fig)
    print(f"{PDF_FIELD}: -> {out_stats}, {out_pdf}", flush=True)

print("done", flush=True)
