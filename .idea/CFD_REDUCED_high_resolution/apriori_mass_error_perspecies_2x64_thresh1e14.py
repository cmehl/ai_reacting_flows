"""A priori error analysis for MODEL_REDUCED_HR_dt5e7_1D_perspecies_2x64_thresh1e14.

Reads the a priori (CFD-snapshot) test results already produced by
test_ann_vs_cvode.py / CFDSnapshotTester on the 9 CSV test slices and computes:

  * E_sigma  = |sum_k Y_k^ANN - 1|          (mass-conservation error of the ANN state)
  * T error  = 100 * |T_ANN - T_CVODE| / T_CVODE   (in %)

then produces the requested scatter plots:

  (1) E_sigma vs T, coloured by T                         [ the plot you asked for ]
  (2) E_sigma vs T error (%), coloured by T               [ mass err vs T err ]

The ANN is only queried for cells with T_ini >= T_threshold (800 K); below that
CFDSnapshotTester returns the state unchanged (identity), so those cells carry
E_sigma == input mass-closure (~1e-16) and are excluded from the "ANN active"
statistics / plots. Set PLOT_ALL_CELLS = True to keep them.
"""

import os

import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
DATA_DIR = "/work/kotlarcm/WORK/AI/clean/ai_reacting_flows/.idea/CFD_REDUCED_high_resolution"
RESULTS_H5 = os.path.join(
    DATA_DIR, "testing_results_REDUCED_HR_dt5e7_1D_perspecies_2x64_thresh1e14_csv.h5"
)
MODEL_TAG = "perspecies_2x64_thresh1e14"
T_THRESHOLD = 800.0            # cells below this are ANN-identity (not predicted)
T_AXIS = "T_ini"              # x-axis / colour temperature: T_ini | T_ann | T_cvode
PLOT_ALL_CELLS = False         # True -> also show the below-threshold identity cells
OUT_PREFIX = os.path.join(DATA_DIR, f"apriori_mass_error_{MODEL_TAG}")
# ----------------------------------------------------------------------

with h5py.File(RESULTS_H5, "r") as f:
    Y_ann = f["Y_ann"][:]
    Y_cvode = f["Y_cvode"][:]
    T_ann = f["T_ann"][:]
    T_cvode = f["T_cvode"][:]
    T_ini = f["T_ini"][:]

# --- error metrics -----------------------------------------------------
E_sigma = np.abs(Y_ann.sum(axis=1) - 1.0)          # ANN mass-conservation error
E_sigma_cvode = np.abs(Y_cvode.sum(axis=1) - 1.0)  # reference (numerical floor)
T_err_pct = 100.0 * np.abs(T_ann - T_cvode) / T_cvode
T_err_K = np.abs(T_ann - T_cvode)

T_axis = {"T_ini": T_ini, "T_ann": T_ann, "T_cvode": T_cvode}[T_AXIS]

ann_active = T_ini >= T_THRESHOLD
mask = np.ones_like(ann_active) if PLOT_ALL_CELLS else ann_active


def _stats(name, a):
    print(f"  {name:24s}  mean={a.mean():.4e}  median={np.median(a):.4e}  "
          f"p99={np.percentile(a, 99):.4e}  max={a.max():.4e}")


print(f"\n=== A priori errors -- MODEL {MODEL_TAG} ===")
print(f"cells total = {len(E_sigma)}   ANN-active (T_ini>=800K) = {ann_active.sum()}   "
      f"identity = {(~ann_active).sum()}")
print("\n-- ANN-active cells --")
_stats("E_sigma = |sum Yk - 1|", E_sigma[ann_active])
_stats("E_sigma (CVODE ref)", E_sigma_cvode[ann_active])
_stats("T error [%]", T_err_pct[ann_active])
_stats("T error [K]", T_err_K[ann_active])
print(f"\ncorr(E_sigma, T error %) = "
      f"{np.corrcoef(E_sigma[ann_active], T_err_pct[ann_active])[0, 1]:.3f}")

with open(f"{OUT_PREFIX}_summary.txt", "w") as fo:
    fo.write(f"A priori errors -- MODEL {MODEL_TAG}\n")
    fo.write(f"results file: {os.path.basename(RESULTS_H5)}\n")
    fo.write(f"cells: total={len(E_sigma)} ann_active={ann_active.sum()} "
             f"identity={(~ann_active).sum()}\n\n")
    for name, a in [("E_sigma_ANN", E_sigma[ann_active]),
                    ("E_sigma_CVODE_ref", E_sigma_cvode[ann_active]),
                    ("T_error_pct", T_err_pct[ann_active]),
                    ("T_error_K", T_err_K[ann_active])]:
        fo.write(f"{name:20s} mean={a.mean():.6e} median={np.median(a):.6e} "
                 f"p99={np.percentile(a, 99):.6e} max={a.max():.6e}\n")
    fo.write(f"\ncorr(E_sigma, T_error_pct) = "
             f"{np.corrcoef(E_sigma[ann_active], T_err_pct[ann_active])[0, 1]:.4f}\n")

# --- Figure 1 : E_sigma vs T, coloured by T --------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(T_axis[mask], E_sigma[mask], c=T_axis[mask], s=4, cmap="inferno",
                alpha=0.6, linewidths=0)
ax.set_yscale("log")
ax.set_xlabel(f"{T_AXIS} [K]")
ax.set_ylabel(r"$E_\sigma = \left|\sum_k Y_k^{\mathrm{ANN}} - 1\right|$")
ax.set_title(f"A priori mass-conservation error -- {MODEL_TAG}")
cb = fig.colorbar(sc, ax=ax)
cb.set_label(f"{T_AXIS} [K]")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT_PREFIX}_Esigma_vs_T.png", dpi=150)
print(f"\nwrote {OUT_PREFIX}_Esigma_vs_T.png")

# --- Figure 2 : E_sigma vs T error (%), coloured by T ----------------
fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(T_err_pct[mask], E_sigma[mask], c=T_axis[mask], s=4, cmap="inferno",
                alpha=0.6, linewidths=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"T error [%] $= 100\,|T_{\mathrm{ANN}} - T_{\mathrm{CVODE}}| / T_{\mathrm{CVODE}}$")
ax.set_ylabel(r"$E_\sigma = \left|\sum_k Y_k^{\mathrm{ANN}} - 1\right|$")
ax.set_title(f"Mass error vs temperature error -- {MODEL_TAG}")
cb = fig.colorbar(sc, ax=ax)
cb.set_label(f"{T_AXIS} [K]")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT_PREFIX}_Esigma_vs_Terr.png", dpi=150)
print(f"wrote {OUT_PREFIX}_Esigma_vs_Terr.png")
