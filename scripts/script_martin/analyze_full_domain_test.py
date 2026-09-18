"""Analysis of a full-domain ANN vs CVODE run (cfd_full_domain_testing.py).

Reads every ``<stem>_part*.h5`` written by FullDomainTester and produces, for
each rollout step k:

- per-field metrics (Temperature + species) over the cells that were actually
  reacted (T > T_threshold): RMSE of (ANN - CVODE), RMSE relative to the
  domain-mean CVODE value (same convention as the earlier step_1 tables),
  and a **skill ratio** = RMSE(ANN-CVODE) / RMSE(CVODE - initial state), i.e.
  the fraction of the reaction increment the network misses (identity
  baseline = 1, perfect = 0). Raw Y errors alone are misleading here because
  a 5e-7 s step barely changes most species.
- fraction of cells with per-cell relative error > 1 %;
- error-vs-initial-temperature heatmap of the skill ratio;
- X=0 slice maps of the Temperature error and of the worst-species error;
- the worst cells.

Config is top-of-file UPPERCASE vars (no argparse).
"""
import glob
import importlib.util
import os
import sys

import h5py
import numpy as np

# ---- config ----
RESULTS_GLOB = "results/full_domain_rollout3_part*.h5"
OUT_PREFIX = "full_domain_rollout3"
ANIMATE_SCRIPT = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/scripts/script_martin/animate_ann_sage_hybrid.py"
N_T_BINS = 25
TOP_K_WORST = 30
SLICE_AXIS = "X"
SLICE_HALFWIDTH = 0.005
SLICE_RES = 0.0005
GRID = 320
FILL_RADIUS = 0.006
# -----------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parts = sorted(glob.glob(RESULTS_GLOB))
assert parts, f"no part files match {RESULTS_GLOB}"
print(f"{len(parts)} part file(s)")

with h5py.File(parts[0], "r") as f:
    species = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["species"]]
    nb_steps = int(f.attrs["nb_steps"])
    T_threshold = float(f.attrs["T_threshold"])
    dt = float(f.attrs["time_step"])
ns = len(species)
fields = ["Temperature"] + species


def cat(name, axis=0):
    out = []
    for p in parts:
        with h5py.File(p, "r") as f:
            out.append(f[name][()])
    return np.concatenate(out, axis=axis)


above = cat("above")
coords_all = cat("coords")
T0_all = cat("T_ini")
n_all = above.size
n = int(above.sum())
print(f"{n_all} cells, {n} reacted (T > {T_threshold:g} K), nb_steps={nb_steps}, dt={dt:g}s")

X0 = np.column_stack([T0_all[above], cat("Y_ini")[above]])              # [n, 1+ns]
Tcv = cat("T_cvode", axis=1)[:, above]                                  # [nb, n]
Tan = cat("T_ann", axis=1)[:, above]
Ycv = cat("Y_cvode", axis=1)[:, above, :]                               # [nb, n, ns]
Yan = cat("Y_ann", axis=1)[:, above, :]
failed = cat("cvode_failed")[above]
print(f"CVODE failures (identity fallback): {int(failed.sum())}")
ok = ~failed

coords = coords_all[above]

rows = []
err_norm_worst = []     # per step: [n] max over fields of |err| / mean_cv
worst_field = []
T_ini = X0[:, 0]
edges = np.linspace(T_ini.min(), T_ini.max(), N_T_BINS + 1)
bin_id = np.clip(np.digitize(T_ini, edges) - 1, 0, N_T_BINS - 1)
skill_T_heat = np.full((nb_steps, len(fields), N_T_BINS), np.nan)

print(f"\n{'step':>4s} {'field':12s} {'rmse_err':>12s} {'rel%':>10s} {'rmse_incr':>12s} {'skill':>8s} {'frac>1%':>9s}")
for k in range(nb_steps):
    ref = np.column_stack([Tcv[k], Ycv[k]])
    prd = np.column_stack([Tan[k], Yan[k]])
    err = prd - ref
    incr = ref - X0
    mean_ref = ref[ok].mean(axis=0)
    denom = np.where(np.abs(mean_ref) > 1e-30, np.abs(mean_ref), 1e-30)
    rel_cell = np.abs(err) / denom                                      # [n, 1+ns], fraction

    worst = rel_cell.max(axis=1)
    err_norm_worst.append(worst)
    worst_field.append(rel_cell.argmax(axis=1))

    for j, name in enumerate(fields):
        rmse_e = float(np.sqrt((err[ok, j] ** 2).mean()))
        rmse_i = float(np.sqrt((incr[ok, j] ** 2).mean()))
        rel = rmse_e / abs(denom[j]) * 100
        skill = rmse_e / rmse_i if rmse_i > 0 else np.nan
        frac = float((rel_cell[ok, j] > 0.01).mean())
        rows.append((k + 1, name, rmse_e, rel, rmse_i, skill, frac, float(np.abs(err[ok, j]).max())))
        print(f"{k+1:4d} {name:12s} {rmse_e:12.4e} {rel:9.4f}% {rmse_i:12.4e} {skill:8.3f} {100*frac:8.3f}%")
        for b in range(N_T_BINS):
            m = ok & (bin_id == b)
            if m.sum() > 50:
                ri = np.sqrt((incr[m, j] ** 2).mean())
                skill_T_heat[k, j, b] = np.sqrt((err[m, j] ** 2).mean()) / ri if ri > 0 else np.nan

    Ysum = Yan[k].sum(axis=1)
    print(f"  step {k+1}: sum(Y_ann)-1  mean={np.mean(Ysum-1):+.3e}  max|.|={np.abs(Ysum-1).max():.3e}")

with open(f"metrics_{OUT_PREFIX}.csv", "w") as fh:
    fh.write("step,field,rmse_err,rel_rmse_pct,rmse_increment,skill_ratio,frac_cells_gt_1pct,max_abs_err\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")

# ---- Figure 1: skill ratio and rel% per field per step ----
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(fields))
w = 0.8 / nb_steps
for k in range(nb_steps):
    sk = [r[5] for r in rows if r[0] == k + 1]
    rl = [r[3] for r in rows if r[0] == k + 1]
    axes[0].bar(x + (k - (nb_steps - 1) / 2) * w, sk, width=w, label=f"step {k+1}")
    axes[1].bar(x + (k - (nb_steps - 1) / 2) * w, np.clip(rl, 1e-6, None), width=w, label=f"step {k+1}")
axes[0].axhline(1.0, color="k", ls="--", lw=1)
axes[0].set_title("skill ratio = RMSE(ANN-CVODE)/RMSE(CVODE-initial)   (1 = identity, 0 = perfect)")
axes[0].set_yscale("log")
axes[1].set_title("RMSE(ANN-CVODE) / mean(CVODE)  [%]")
axes[1].set_yscale("log")
for a in axes:
    a.set_xticks(x); a.set_xticklabels(fields, rotation=60, ha="right"); a.legend(); a.grid(True, axis="y", alpha=0.3)
fig.suptitle(f"Full domain, {n} cells, rollout of {nb_steps} x {dt:g}s")
fig.tight_layout(); fig.savefig(f"summary_{OUT_PREFIX}.png", dpi=140); plt.close(fig)

# ---- Figure 2: skill ratio vs initial temperature (step 1) ----
fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(np.log10(np.clip(skill_T_heat[0], 1e-4, 1e3)), aspect="auto", origin="lower",
               extent=[edges[0], edges[-1], -0.5, len(fields) - 0.5], cmap="RdYlGn_r", vmin=-2, vmax=1)
ax.set_yticks(range(len(fields))); ax.set_yticklabels(fields)
ax.set_xlabel("initial Temperature [K]")
plt.colorbar(im, label="log10(skill ratio), step 1  (0 = identity)")
ax.set_title("Where the network misses the reaction increment (step 1)")
fig.tight_layout(); fig.savefig(f"skill_vs_T_{OUT_PREFIX}.png", dpi=140); plt.close(fig)

# ---- Figure 3: X=0 slice maps ----
spec = importlib.util.spec_from_file_location("animate_ann_sage_hybrid", ANIMATE_SCRIPT)
anim = importlib.util.module_from_spec(spec)
sys.modules["animate_ann_sage_hybrid"] = anim
spec.loader.exec_module(anim)

axcfg = anim.SLICE_AXES[SLICE_AXIS]
sel = anim.slice_indices(coords_all, axcfg, SLICE_HALFWIDTH, SLICE_RES)
h = axcfg["h"][1] * coords_all[sel, axcfg["h"][0]]
v = axcfg["v"][1] * coords_all[sel, axcfg["v"][0]]
binner = anim.make_binner(h, v, GRID, FILL_RADIUS)
to_img = binner["to_img"]
im_kw = dict(extent=binner["extent"], origin="lower", interpolation="nearest", aspect="equal")


def full_from_above(vals):
    out = np.zeros(n_all)
    out[above] = vals
    return out


k_last = nb_steps - 1
panels = [
    (f"Temperature error, step 1 [K]", full_from_above(Tan[0] - Tcv[0]), "coolwarm", True),
    (f"Temperature error, step {nb_steps} [K]", full_from_above(Tan[k_last] - Tcv[k_last]), "coolwarm", True),
    ("worst species error / mean, step 1", full_from_above(np.log10(np.clip(err_norm_worst[0], 1e-6, None))), "inferno", False),
]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (title, arr, cmap, sym) in zip(axes, panels):
    a = arr[sel]
    if sym:
        lim = float(np.percentile(np.abs(a), 99.5)) or 1e-30
        im = ax.imshow(to_img(a), cmap=cmap, vmin=-lim, vmax=lim, **im_kw)
    else:
        im = ax.imshow(to_img(a), cmap=cmap, vmin=-4, vmax=1, **im_kw)
    ax.set_title(title + ("  (log10)" if not sym else ""))
    ax.set_xlabel(axcfg["h"][2]); ax.set_ylabel(axcfg["v"][2])
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(f"maps_{OUT_PREFIX}.png", dpi=130); plt.close(fig)

# ---- Worst cells (step 1, max normalized error over fields) ----
order = np.argsort(-err_norm_worst[0])[:TOP_K_WORST]
with open(f"worst_cells_{OUT_PREFIX}.csv", "w") as fh:
    fh.write("rank,X,Y,Z,T_ini,worst_field,err_over_mean\n")
    for r, i in enumerate(order, start=1):
        fh.write(f"{r},{coords[i,0]},{coords[i,1]},{coords[i,2]},{T_ini[i]},{fields[worst_field[0][i]]},{err_norm_worst[0][i]}\n")
print(f"\n-> metrics_{OUT_PREFIX}.csv, summary_/skill_vs_T_/maps_{OUT_PREFIX}.png, worst_cells_{OUT_PREFIX}.csv")
