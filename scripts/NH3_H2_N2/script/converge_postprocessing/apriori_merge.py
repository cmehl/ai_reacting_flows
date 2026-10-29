"""Merge of apriori_step.py: a priori choice of the hybrid ANN/CVODE threshold of P K3 tanh dt 5e-6 (method of Mehl & Aubagnac-Karkar,
PoF 2023, Sec. VI B / Fig. 11-12 / Eq. 15), on the WHOLE hot domain (T > 800 K) of SAGE dt 5e-6 at several instants.
For each threshold eps on the CONVERGE criterion e_e (max relative element drift of the raw network output): cells with e_e > eps -> CVODE
(exact), others -> network + renorm 3. Writes:
  apriori_thresholds.csv            per instant and all instants: cost (cells, mass, CVODE time share), T error, species per-step mass bias
  apriori_scatter.png               e_e vs true one-step error (T, O, NO, N), log-log 2D histograms, threshold lines (paper Fig. 11-12)
  apriori_vs_threshold.png          cost / remaining error / captured error / species bias vs threshold"""
import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---- config ----
EPS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 2e-4, np.inf]          # inf = no hybrid
EPS_NOW = 3e-5                                                # current CONVERGE hybrid_ann_tol
SPEC = ["O", "OH", "H", "NO", "N", "N2H2", "H2O", "NH3", "HO2"]
SCAT = [("T", "e_T [K]"), ("O", "|err Y_O| / Y_O"), ("NO", "|err Y_NO| / Y_NO"), ("N", "|err Y_N| / Y_N")]
N_SCAT = 400_000                                              # points per instant in the scatter (plot only; metrics use every cell)
ET_BIG = 0.1                                                  # [K] "large" one-step T error for the recall
COL_EPS = {1e-6: "#08306b", 3e-6: "#2171b5", 1e-5: "#41b6c4", 3e-5: "#2b9348", 1e-4: "#7b4fa0", 2e-4: "#6c757d"}
plt.rcParams.update({"font.size": 11})
# ----------------

rng = np.random.default_rng(0)
files = sorted(glob.glob("steps/apriori_*.npz"))
instants = sorted({f.split("_")[-2] for f in files})
rows, scat = [], {}
acc = {e: {} for e in EPS}                                    # sums over all instants


def add(d, k, v):
    d[k] = d.get(k, 0.0) + v


for tu in instants:
    zs = [np.load(f) for f in files if f.split("_")[-2] == tu]
    SP = list(zs[0]["species"])
    cat = lambda k: np.concatenate([z[k] for z in zs])
    m, T0, Y0, ee = cat("m"), cat("T0"), cat("Y0"), cat("ee")
    T_r3, Y_r3, T_cv, Y_cv, t_cv = cat("T_r3"), cat("Y_r3"), cat("T_cv"), cat("Y_cv"), cat("t_cv")
    ok = np.isfinite(T_cv)
    m, T0, Y0, ee, T_r3, Y_r3, T_cv, Y_cv, t_cv = (a[ok] for a in (m, T0, Y0, ee, T_r3, Y_r3, T_cv, Y_cv, t_cv))
    t = float(zs[0]["t"])
    eT = T_r3 - T_cv
    dTcv = T_cv - T0
    js = {s: SP.index(s) for s in SPEC}
    print(f"t = {t * 1e3:.1f} ms: {len(T0)} hot cells ({(~ok).sum()} CVODE failures dropped), corr(log e_e, log |e_T|) = "
          f"{np.corrcoef(np.log10(ee + 1e-12), np.log10(np.abs(eT) + 1e-9))[0, 1]:.3f}", flush=True)
    # scatter subsample
    ii = rng.choice(len(T0), min(N_SCAT, len(T0)), replace=False)
    scat[t] = {"ee": ee[ii], "T": np.abs(eT[ii])}
    for s in ["O", "NO", "N"]:
        j = SP.index(s)
        scat[t][s] = np.abs(Y_r3[ii, j] - Y_cv[ii, j]) / np.maximum(Y_cv[ii, j], 1e-14)
    for e in EPS:
        cv = ee > e
        a = acc[e]
        r = {"t_ms": t * 1e3, "eps": e, "alpha_cells_%": 100 * cv.mean(), "alpha_mass_%": 100 * m[cv].sum() / m.sum(),
             "cvode_time_share_%": 100 * t_cv[cv].sum() / t_cv.sum()}
        eTh = np.where(cv, 0.0, eT)
        r["T_rmse_K"] = np.sqrt(np.mean(eTh ** 2))
        r["T_rmse_%incr"] = 100 * r["T_rmse_K"] / np.sqrt(np.mean(dTcv ** 2))
        r["T_bias_mK"] = 1e3 * np.sum(m * eTh) / m.sum()
        r["T_max_K"] = np.abs(eTh).max()
        r["T_sqerr_removed_%"] = 100 * (1 - np.sum(eTh ** 2) / np.sum(eT ** 2))
        big = np.abs(eT) > ET_BIG
        r[f"recall_eT>{ET_BIG}K_%"] = 100 * cv[big].mean() if big.any() else np.nan
        for k, v in (("n", len(T0)), ("ncv", cv.sum()), ("m", m.sum()), ("mcv", m[cv].sum()), ("tcv", t_cv[cv].sum()), ("tall", t_cv.sum()),
                     ("se", np.sum(eTh ** 2)), ("se0", np.sum(eT ** 2)), ("sd", np.sum(dTcv ** 2)), ("mbT", np.sum(m * eTh))):
            add(a, k, v)
        for s, j in js.items():
            err = np.where(cv, 0.0, Y_r3[:, j] - Y_cv[:, j])
            b = np.sum(m * err)
            r[f"{s}_bias_%mass/step"] = 100 * b / np.sum(m * Y_cv[:, j])
            r[f"{s}_bias_%incr"] = 100 * b / np.sum(m * np.abs(Y_cv[:, j] - Y0[:, j]))
            r[f"{s}_abserr_removed_%"] = 100 * (1 - np.sum(m * np.abs(err)) / np.sum(m * np.abs(Y_r3[:, j] - Y_cv[:, j])))
            add(a, f"b_{s}", b); add(a, f"M_{s}", np.sum(m * Y_cv[:, j])); add(a, f"I_{s}", np.sum(m * np.abs(Y_cv[:, j] - Y0[:, j])))
            add(a, f"ae_{s}", np.sum(m * np.abs(err))); add(a, f"ae0_{s}", np.sum(m * np.abs(Y_r3[:, j] - Y_cv[:, j])))
        rows.append(r)
    del zs

for e in EPS:
    a = acc[e]
    r = {"t_ms": "all", "eps": e, "alpha_cells_%": 100 * a["ncv"] / a["n"], "alpha_mass_%": 100 * a["mcv"] / a["m"],
         "cvode_time_share_%": 100 * a["tcv"] / a["tall"], "T_rmse_K": np.sqrt(a["se"] / a["n"]), "T_rmse_%incr": 100 * np.sqrt(a["se"] / a["sd"]),
         "T_bias_mK": 1e3 * a["mbT"] / a["m"], "T_sqerr_removed_%": 100 * (1 - a["se"] / a["se0"])}
    for s in SPEC:
        r[f"{s}_bias_%mass/step"] = 100 * a[f"b_{s}"] / a[f"M_{s}"]
        r[f"{s}_bias_%incr"] = 100 * a[f"b_{s}"] / a[f"I_{s}"]
        r[f"{s}_abserr_removed_%"] = 100 * (1 - a[f"ae_{s}"] / a[f"ae0_{s}"])
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv("apriori_thresholds.csv", index=False)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)
show = ["t_ms", "eps", "alpha_cells_%", "alpha_mass_%", "cvode_time_share_%", "T_rmse_%incr", "T_bias_mK", "T_sqerr_removed_%"] + \
       [f"{s}_bias_%incr" for s in ["O", "OH", "NO", "N", "N2H2", "H2O"]] + [f"{s}_abserr_removed_%" for s in ["O", "NO", "N"]]
print(tab[[c for c in show if c in tab]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# ---- scatter (paper Fig. 11-12: proxy vs true error, threshold lines) ----
ts = sorted(scat)
fig, axs = plt.subplots(len(SCAT), len(ts), figsize=(4.3 * len(ts), 3.6 * len(SCAT)), constrained_layout=True, squeeze=False)
for i, (k, lab) in enumerate(SCAT):
    for j, t in enumerate(ts):
        ax = axs[i, j]
        x, y = np.maximum(scat[t][k], 1e-10), np.maximum(scat[t]["ee"], 1e-10)
        h = ax.hist2d(np.log10(x), np.log10(y), bins=120, range=[[-8 if k == "T" else -9, 1 if k == "T" else 1], [-9, -1]],
                      norm=LogNorm(), cmap="viridis")
        for e, c in COL_EPS.items():
            ax.axhline(np.log10(e), color=c, ls="--", lw=1.2, label=f"ε = {e:.0e}" + (" (actuel)" if e == EPS_NOW else ""))
        ax.set_xlabel(f"log10 {lab}")
        ax.set_ylabel("log10 e_e (dérive élémentaire max)")
        if i == 0:
            ax.set_title(f"t = {t * 1e3:.0f} ms")
axs[0, -1].legend(fontsize=8, loc="lower right")
fig.suptitle("P K3 tanh dt 5e-6, a priori sur SAGE dt 5e-6 (toutes les cellules T > 800 K) : critère hybride e_e vs vraie erreur à 1 pas (CVODE)",
             fontsize=13)
fig.savefig("apriori_scatter.png", dpi=110, facecolor="white")
plt.close(fig)

# ---- metrics vs threshold (all instants + per instant) ----
d = tab[tab.eps < np.inf]
fin = tab[(tab.eps == np.inf)]
fig, axs = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
tcols = ["#08306b", "#2171b5", "#41b6c4", "#2b9348", "#7b4fa0"]
panels = [("cvode_time_share_%", "part du temps CVODE envoyée en CVODE [%]", True), ("alpha_cells_%", "α : cellules en CVODE [%]", True),
          ("T_rmse_%incr", "erreur RMS T restante [% de l'incrément CVODE]", False), ("T_sqerr_removed_%", "erreur² T supprimée [%]", False),
          ("O_bias_%incr", "biais O restant [% de l'incrément]", False), ("NO_bias_%incr", "biais NO restant [% de l'incrément]", False)]
for ax, (col, lab, logy) in zip(axs.flat, panels):
    for c, tm in zip(tcols, [x for x in d.t_ms.unique() if x != "all"]):
        dd = d[d.t_ms == tm]
        ax.plot(dd.eps, dd[col], "o-", color=c, lw=1.4, ms=4, label=f"{float(tm):.0f} ms")
        if col not in ("T_sqerr_removed_%",) and not col.startswith("alpha") and not col.startswith("cvode"):
            ax.axhline(fin[fin.t_ms == tm][col].iloc[0], color=c, lw=0.8, ls=":")
    dd = d[d.t_ms == "all"]
    ax.plot(dd.eps, dd[col], "s-", color="k", lw=2.2, ms=5, label="tous instants")
    ax.axvline(EPS_NOW, color="#6c757d", ls="--", lw=1)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("seuil ε (hybrid_ann_tol)")
    ax.set_title(lab, fontsize=11)
    ax.grid(alpha=0.3, which="both")
axs[0, 0].legend(fontsize=9)
fig.suptitle("Choix a priori du seuil hybride, P K3 tanh dt 5e-6 (pointillés horizontaux : sans hybride ; tiret vertical : ε actuel 3e-5)", fontsize=13)
fig.savefig("apriori_vs_threshold.png", dpi=110, facecolor="white")
plt.close(fig)
print(">> done")
