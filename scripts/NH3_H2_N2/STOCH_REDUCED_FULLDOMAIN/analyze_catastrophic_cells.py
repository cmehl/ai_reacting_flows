"""Which CFD states are badly predicted, and are they far from the stochastic-reactor database?

Adapted from the multi-step CFD version (scripts/script_martin/analyze_catastrophic_cells.py) to the single-step
full-domain test of the STOCH reduced models (full_domain_test.py): reference CVODE in REF_FILE, ANN predictions in
results/<model>_ann.h5, and the model's own reactor database (its training_data.h5) as the "database".

Catastrophic cell of a species = |ANN - CVODE| > CATA_FRAC x domain-mean CVODE value of that species (same definition as
the original). For Temperature: |dT| > T_CATA_K. For every entry of SPECIES_LIST it reports and plots:
- probability of being catastrophic versus initial temperature, initial species value and distance to the database;
- over- versus under-prediction, and whether the cell sits at the log floor;
- nearest-neighbour distance (standardised [T, log Y] input space) of catastrophic cells versus random cells and versus
  held-out database states (X_val: the "inside the database" reference), and how much of the squared error lies in cells
  beyond the database p99;
- which input variables differ most from the nearest database state;
- where the cells are in the domain.
It also compares the marginal distributions database vs CFD (all cells and catastrophic cells) and the fraction of CFD
cells outside the database per initial-temperature bin.

Config is top-of-file UPPERCASE vars (no argparse).
"""
import json
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

# ---- config ----
ROOT = "/work/kotlarcm/WORK/AI/clean/ai_reacting_flows"
WT = f"{ROOT}/.claude/worktrees/cfd-rollout-dtb/scripts/NH3_H2_N2"
HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = f"{HERE}/reference_cvode_full_domain.h5"
RESULTS_DIR = f"{HERE}/results"
OUT_DIR = f"{HERE}/catastrophic"

# model name (as in full_domain_test.py) -> training set of the model + its log threshold
MODELS = {
    "old_perspecies_REF": dict(
        training_h5=f"{ROOT}/.idea/NH3_H2_N2_REDUCED/STOCH_DTB_NH3_H2_N2/dtb_log_k2/training_data.h5",
        threshold=1e-10),
    "long_rollout_0cl_thr1e14": dict(
        training_h5=f"{WT}/STOCH_REDUCED_ROLLOUT_LONG/STOCH_DTB_NH3_H2_N2_ROLLOUT_LONG/dtb_rollout_noclust_thr1e14/training_data.h5",
        threshold=1e-14),
}
MODELS_TO_ANALYZE = ["old_perspecies_REF"]

SPECIES_LIST = ["N", "NH", "NNH", "NH2", "N2H2", "HNO", "H", "OH", "Temperature"]
CATA_FRAC = 1.0
T_CATA_K = 1.0
FLOOR_Y = 1e-12
TREE_MAX = 1_000_000      # database states in the kd-tree
N_SAMPLE = 2_000_000      # random CFD cells used for distance statistics
N_CATA_MAX = 30_000       # catastrophic cells (per species) queried
N_GAP = 20_000            # cells used for the per-variable gap to the nearest database state
N_VAL = 20_000
T_EDGES = [800, 1500, 1900, 2050, 2090, 2110, 3000]
N_T_BINS = 30
N_Y_BINS = 40
QUERY_CHUNK = 500_000
SEED = 0
# -----------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def features(T, Y, thr):
    """Model input space: [T, log(clip(Y, thr))] (same transform as the training data)."""
    return np.column_stack([T, np.log(np.clip(Y, thr, None))])


def load_database(path, rng):
    """Unscaled training features (both clusters together) + held-out validation features."""
    Xs, Xv = [], []
    with h5py.File(path, "r") as f:
        clusters = sorted(k for k in f.keys() if k.startswith("CLUSTER_"))
        cols = [str(c) for c in f[clusters[0]]["X_train"].attrs["cols"]]
        for c in clusters:
            g = f[c]
            sc = g["Xscaler"][:]
            mean, std = sc[:, 0], np.sqrt(sc[:, 1])
            Xs.append(g["X_train"][:] * std + mean)
            v = g["X_val"]
            take = np.sort(rng.choice(v.shape[0], min(N_VAL, v.shape[0]), replace=False))
            Xv.append(v[take] * std + mean)
    return np.vstack(Xs), np.vstack(Xv), cols


def nn_query(tree, Xq, mu, sd, want_idx=False):
    d = np.empty(Xq.shape[0])
    idx = np.empty(Xq.shape[0], dtype=np.int64) if want_idx else None
    for s in range(0, Xq.shape[0], QUERY_CHUNK):
        dd, ii = tree.query((Xq[s:s + QUERY_CHUNK] - mu) / sd, k=1, workers=-1)
        d[s:s + QUERY_CHUNK] = dd
        if want_idx:
            idx[s:s + QUERY_CHUNK] = ii
    return (d, idx) if want_idx else d


def analyze(model):
    cfg = MODELS[model]
    thr = cfg["threshold"]
    prefix = f"{OUT_DIR}/cata_{model}"
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {"model": model, "log_threshold": thr}

    # ---- CFD cells (T > 800 K, the ones the model is used on) ----
    with h5py.File(REF_FILE, "r") as f:
        species = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["species"]]
        T0, Y0 = f["T_ini"][()], f["Y_ini"][()]
        Tc, Yc = f["T_cvode"][()], f["Y_cvode"][()]
        coords = np.column_stack([f["XCEN_X"][()], f["XCEN_Y"][()], f["XCEN_Z"][()]])
    with h5py.File(f"{RESULTS_DIR}/{model}_ann.h5", "r") as f:
        Ta, Ya = f["T_ann"][()], f["Y_ann"][()]
    n = T0.size
    print(f"\n##### {model}: {n} reacted CFD cells, log threshold {thr:g}", flush=True)

    # ---- database ----
    Xtr, Xval, cols = load_database(cfg["training_h5"], rng)
    expected = ["Temperature_X"] + [f"{s}_X" for s in species]
    assert cols == expected, "training columns do not match the reference species order"
    mu, sd = Xtr.mean(0), Xtr.std(0)
    tree_X = Xtr[rng.choice(Xtr.shape[0], min(TREE_MAX, Xtr.shape[0]), replace=False)]
    tree = cKDTree((tree_X - mu) / sd)
    d_val = nn_query(tree, Xval, mu, sd)
    p99 = float(np.percentile(d_val, 99))
    print(f"database: {Xtr.shape[0]} training states ({tree_X.shape[0]} in the tree); held-out val->train NN distance "
          f"p50 {np.median(d_val):.3f} p90 {np.percentile(d_val, 90):.3f} p99 {p99:.3f}", flush=True)
    summary["db_rows"] = int(Xtr.shape[0])
    summary["db_val_nn_dist"] = {"p50": float(np.median(d_val)), "p90": float(np.percentile(d_val, 90)), "p99": p99}

    # ---- catastrophic masks ----
    entries = {}
    for name in SPECIES_LIST:
        if name == "Temperature":
            y0, cv, an = T0, Tc, Ta
            cata = np.abs(an - cv) > T_CATA_K
            scale = T_CATA_K
        else:
            j = species.index(name)
            y0, cv, an = Y0[:, j], Yc[:, j], Ya[:, j]
            scale = float(np.abs(cv).mean())
            cata = np.abs(an - cv) > CATA_FRAC * scale
        entries[name] = dict(y0=y0, cv=cv, an=an, cata=cata, scale=scale)
        print(f"{name}: catastrophic threshold {CATA_FRAC * scale if name != 'Temperature' else T_CATA_K:.3e}, "
              f"{int(cata.sum())} cells ({100 * cata.mean():.3f}% of reacted)", flush=True)

    # ---- distances: random sample + catastrophic cells ----
    rand_idx = rng.choice(n, min(N_SAMPLE, n), replace=False)
    cata_used = {nm: (rng.choice(np.where(e["cata"])[0], min(N_CATA_MAX, int(e["cata"].sum())), replace=False)
                      if e["cata"].any() else np.array([], dtype=int)) for nm, e in entries.items()}
    query_idx = np.unique(np.concatenate([rand_idx] + list(cata_used.values())))
    d_full = np.full(n, np.nan)
    Xq = features(T0[query_idx], Y0[query_idx], thr)
    d_full[query_idx] = nn_query(tree, Xq, mu, sd)
    d_rand = d_full[rand_idx]
    print(f"CFD random cells -> database NN distance: p50 {np.median(d_rand):.3f} p90 {np.percentile(d_rand, 90):.3f} "
          f"p99 {np.percentile(d_rand, 99):.3f}; beyond the database p99: {100 * (d_rand > p99).mean():.1f}% "
          f"(held-out database states: 1.0% by construction)", flush=True)
    summary["cfd_random_nn_dist"] = {"p50": float(np.median(d_rand)), "p90": float(np.percentile(d_rand, 90)),
                                     "p99": float(np.percentile(d_rand, 99)), "frac_beyond_db_p99": float((d_rand > p99).mean())}

    # ---- overview: marginals database vs CFD ----
    Xcfd_rand = features(T0[rand_idx], Y0[rand_idx], thr)
    names = ["T"] + species
    fig, axes = plt.subplots(4, 5, figsize=(22, 15))
    marg = {}
    for i, ax in enumerate(axes.ravel()):
        if i >= len(names):
            ax.axis("off")
            continue
        lo = min(np.percentile(Xtr[:, i], 0.2), np.percentile(Xcfd_rand[:, i], 0.2))
        hi = max(np.percentile(Xtr[:, i], 99.8), np.percentile(Xcfd_rand[:, i], 99.8))
        bins = np.linspace(lo, hi, 70)
        ax.hist(Xtr[:, i], bins=bins, density=True, alpha=0.5, color="tab:blue", label="reactor database")
        ax.hist(Xcfd_rand[:, i], bins=bins, density=True, alpha=0.5, color="gray", label="CFD (all reacted cells)")
        ax.set_title(names[i]); ax.set_yticks([])
        out = float(((Xcfd_rand[:, i] < np.percentile(Xtr[:, i], 0.5)) | (Xcfd_rand[:, i] > np.percentile(Xtr[:, i], 99.5))).mean())
        marg[names[i]] = {"db_median": float(np.median(Xtr[:, i])), "cfd_median": float(np.median(Xcfd_rand[:, i])),
                          "cfd_frac_outside_db_0.5_99.5pct": out}
        ax.set_xlabel(f"{'T [K]' if i == 0 else 'ln Y'}   outside DB [0.5,99.5]%: {100 * out:.1f}%", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{model}: reactor database vs CFD, model input variables (ln Y clipped at {thr:g})")
    fig.tight_layout(); fig.savefig(f"{prefix}_marginals.png", dpi=110); plt.close(fig)
    summary["marginals"] = marg
    print("variables where >5% of CFD cells fall outside the database [0.5,99.5]% range: " +
          ", ".join(f"{k} {100 * v['cfd_frac_outside_db_0.5_99.5pct']:.0f}%" for k, v in marg.items()
                    if v["cfd_frac_outside_db_0.5_99.5pct"] > 0.05), flush=True)

    # ---- OOD versus initial temperature ----
    T_r = T0[rand_idx]
    tb = [(a, b) for a, b in zip(T_EDGES[:-1], T_EDGES[1:])]
    rows = []
    err2_T = (Ta - Tc) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(tb))
    frac_out, med_d = [], []
    for a, b in tb:
        m = (T_r >= a) & (T_r < b)
        frac_out.append(float((d_rand[m] > p99).mean()) if m.any() else np.nan)
        med_d.append(float(np.median(d_rand[m])) if m.any() else np.nan)
        rows.append({"bin": f"[{a},{b})", "cells_pct": float(100 * ((T0 >= a) & (T0 < b)).mean()),
                     "frac_beyond_db_p99": frac_out[-1], "median_nn_dist": med_d[-1]})
    ax[0].bar(xs, 100 * np.array(frac_out), color="tab:red")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f"{a}-{b}" for a, b in tb])
    ax[0].set_ylabel("% of CFD cells beyond database p99"); ax[0].set_xlabel("initial T [K]")
    ax[1].bar(xs, med_d, color="tab:blue"); ax[1].axhline(p99, color="k", ls="--", label="database val p99")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{a}-{b}" for a, b in tb]); ax[1].legend()
    ax[1].set_ylabel("median NN distance to database"); ax[1].set_xlabel("initial T [K]")
    fig.suptitle(f"{model}: distance of the CFD cells to the reactor database, per initial temperature")
    fig.tight_layout(); fig.savefig(f"{prefix}_ood_vs_T.png", dpi=120); plt.close(fig)
    summary["ood_by_T_bin"] = rows
    print("OOD by initial-T bin (fraction of CFD cells beyond the database p99 | median NN dist | % of cells):")
    for r in rows:
        print(f"  {r['bin']:12s} {100 * r['frac_beyond_db_p99']:6.1f}% | {r['median_nn_dist']:.3f} | {r['cells_pct']:.1f}%", flush=True)

    # ---- per species ----
    z_edges = np.linspace(coords[:, 2].min(), coords[:, 2].max(), 80)
    y_edges = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 60)
    H_all, _, _ = np.histogram2d(coords[:, 2], coords[:, 1], bins=[z_edges, y_edges])
    t_edges = np.linspace(T0.min(), T0.max(), N_T_BINS + 1)
    t_id = np.clip(np.digitize(T0, t_edges) - 1, 0, N_T_BINS - 1)
    dq_edges = np.quantile(d_rand, np.linspace(0, 1, 11))
    gaps, maps = {}, {}
    sp_summary = {}
    for nm in SPECIES_LIST:
        e = entries[nm]
        y0, cv, an, cata = e["y0"], e["cv"], e["an"], e["cata"]
        nc = int(cata.sum())
        s = {"n_catastrophic": nc, "pct_of_reacted": float(100 * cata.mean())}
        sp_summary[nm] = s
        if nc == 0:
            continue
        err = an - cv
        s["over_predicted_pct"] = float(100 * (err[cata] > 0).mean())
        s["median_T0_cata"] = float(np.median(T0[cata]))
        if nm != "Temperature":
            ratio = np.clip(an[cata], thr, None) / np.clip(cv[cata], thr, None)
            s["median_ratio_ann_over_cvode"] = float(np.median(ratio))
            s["cata_initial_at_floor_pct"] = float(100 * (y0[cata] < FLOOR_Y).mean())
            s["cata_cvode_at_floor_pct"] = float(100 * (cv[cata] < FLOOR_Y).mean())
        d_c = d_full[cata_used[nm]]
        s["nn_dist_cata_median"] = float(np.median(d_c))
        s["nn_dist_random_median"] = float(np.median(d_rand))
        s["cata_beyond_db_p99_pct"] = float(100 * (d_c > p99).mean())
        s["random_beyond_db_p99_pct"] = float(100 * (d_rand > p99).mean())
        # error explained by distance (random sample, unbiased)
        e2 = err[rand_idx] ** 2
        far = d_rand > p99
        s["sse_share_in_cells_beyond_db_p99_pct"] = float(100 * e2[far].sum() / e2.sum())
        s["cells_beyond_db_p99_pct"] = float(100 * far.mean())
        sub = rng.choice(rand_idx.size, min(200_000, rand_idx.size), replace=False)
        s["spearman_nn_dist_vs_abs_err"] = float(spearmanr(d_rand[sub], np.abs(err[rand_idx][sub]))[0])
        print(f"\n=== {nm}: {nc} catastrophic cells ({s['pct_of_reacted']:.3f}% of reacted)", flush=True)
        print(f"  over-predicted in {s['over_predicted_pct']:.1f}%; median initial T {s['median_T0_cata']:.0f} K (all reacted {np.median(T0):.0f} K)")
        if nm != "Temperature":
            print(f"  median ANN/CVODE ratio {s['median_ratio_ann_over_cvode']:.3g}; at the log floor (<{FLOOR_Y:g}): initial {s['cata_initial_at_floor_pct']:.1f}%, CVODE {s['cata_cvode_at_floor_pct']:.1f}%")
        print(f"  distance to database: catastrophic median {s['nn_dist_cata_median']:.3f} vs random {s['nn_dist_random_median']:.3f}; "
              f"beyond database p99: {s['cata_beyond_db_p99_pct']:.1f}% of catastrophic vs {s['random_beyond_db_p99_pct']:.1f}% of random cells")
        print(f"  cells beyond the database p99 hold {s['sse_share_in_cells_beyond_db_p99_pct']:.1f}% of the squared error "
              f"(they are {s['cells_beyond_db_p99_pct']:.1f}% of cells); Spearman(dist, |err|) = {s['spearman_nn_dist_vs_abs_err']:.3f}", flush=True)

        # gap to nearest database state, per input variable
        ci = rng.choice(cata_used[nm], min(N_GAP, cata_used[nm].size), replace=False)
        ri = rng.choice(rand_idx, N_GAP, replace=False)
        g = {}
        for label, ids in (("catastrophic", ci), ("random", ri)):
            Xg = features(T0[ids], Y0[ids], thr)
            _, nn = nn_query(tree, Xg, mu, sd, want_idx=True)
            g[label] = (np.abs(Xg - tree_X[nn]) / sd).mean(0)
        gaps[nm] = g
        maps[nm] = np.histogram2d(coords[cata, 2], coords[cata, 1], bins=[z_edges, y_edges])[0]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        frac_T = np.array([cata[t_id == b].mean() if (t_id == b).any() else np.nan for b in range(N_T_BINS)])
        axes[0, 0].plot(0.5 * (t_edges[1:] + t_edges[:-1]), 100 * frac_T, marker="o")
        axes[0, 0].set_xlabel("initial T [K]"); axes[0, 0].set_ylabel("% of cells catastrophic")
        axes[0, 0].set_title(f"{nm}: catastrophic probability vs T"); axes[0, 0].grid(alpha=0.3)

        if nm != "Temperature":
            ly = np.log10(np.clip(y0, thr, None))
            y_e = np.linspace(ly.min(), ly.max(), N_Y_BINS + 1)
            y_id = np.clip(np.digitize(ly, y_e) - 1, 0, N_Y_BINS - 1)
            frac_y = np.array([cata[y_id == b].mean() if (y_id == b).sum() > 100 else np.nan for b in range(N_Y_BINS)])
            axes[0, 1].plot(0.5 * (y_e[1:] + y_e[:-1]), 100 * frac_y, marker="o")
            axes[0, 1].set_xlabel(f"log10 initial Y_{nm}")
            axes[0, 2].hist(np.log10(ratio), bins=60, color="#e0311f"); axes[0, 2].axvline(0, color="k", lw=1)
            axes[0, 2].set_xlabel("log10(ANN / CVODE) of catastrophic cells")
        else:
            axes[0, 1].hist(err[cata], bins=80, color="#e0311f"); axes[0, 1].set_xlabel("dT of catastrophic cells [K]")
            axes[0, 2].axis("off")
        axes[0, 1].set_ylabel("% catastrophic" if nm != "Temperature" else "cells"); axes[0, 1].grid(alpha=0.3)
        axes[0, 2].set_title(f"over-predicted in {s['over_predicted_pct']:.0f}%")

        axes[1, 0].hist(d_val, bins=60, density=True, alpha=0.5, color="tab:green", label="held-out database states")
        axes[1, 0].hist(d_rand, bins=60, density=True, alpha=0.5, color="gray", label="CFD random cells")
        axes[1, 0].hist(d_c, bins=60, density=True, alpha=0.6, color="#e0311f", label="CFD catastrophic cells")
        axes[1, 0].axvline(p99, color="k", ls="--", label="database val p99")
        axes[1, 0].set_xlabel("nearest-neighbour distance to the reactor database"); axes[1, 0].legend(fontsize=8)

        pc = []
        for a, b in zip(dq_edges[:-1], dq_edges[1:]):
            m = (d_rand >= a) & (d_rand <= b)
            pc.append(100 * cata[rand_idx][m].mean())
        axes[1, 1].bar(np.arange(10), pc, color="tab:red")
        axes[1, 1].set_xticks(np.arange(10)); axes[1, 1].set_xticklabels([f"{0.5 * (a + b):.2f}" for a, b in zip(dq_edges[:-1], dq_edges[1:])], rotation=45)
        axes[1, 1].set_xlabel("NN distance to database (decile centre)"); axes[1, 1].set_ylabel("% catastrophic")
        axes[1, 1].set_title("is the error larger when the state is far from the database?")

        dbs = rng.choice(Xtr.shape[0], 100_000, replace=False)
        cfs = rng.choice(n, 100_000, replace=False)
        Ydb = np.exp(Xtr[dbs, 1 + species.index(nm)]) if nm != "Temperature" else Xtr[dbs, 0]
        Tdb = Xtr[dbs, 0]
        axes[1, 2].scatter(Tdb, Ydb, s=1, alpha=0.15, color="tab:blue", label="reactor database")
        axes[1, 2].scatter(T0[cfs], np.clip(y0[cfs], thr, None) if nm != "Temperature" else y0[cfs], s=1, alpha=0.15, color="gray", label="CFD")
        cs = rng.choice(np.where(cata)[0], min(5000, nc), replace=False)
        axes[1, 2].scatter(T0[cs], np.clip(y0[cs], thr, None) if nm != "Temperature" else y0[cs], s=5, color="#e0311f", label="CFD catastrophic")
        if nm != "Temperature":
            axes[1, 2].set_yscale("log")
        axes[1, 2].set_xlabel("initial T [K]"); axes[1, 2].set_ylabel(f"initial {nm}" if nm == "Temperature" else f"initial Y_{nm}")
        axes[1, 2].legend(markerscale=6, fontsize=8)
        fig.suptitle(f"{model} - {nm}: catastrophic cells ({nc}, {s['pct_of_reacted']:.3f}% of reacted) vs the reactor database")
        fig.tight_layout(); fig.savefig(f"{prefix}_{nm}.png", dpi=120); plt.close(fig)

    # ---- which variables are far from the database ----
    if gaps:
        k = len(gaps)
        fig, axes = plt.subplots((k + 1) // 2, 2, figsize=(18, 4 * ((k + 1) // 2)), squeeze=False)
        for ax, (nm, g) in zip(axes.ravel(), gaps.items()):
            x = np.arange(len(names))
            ax.bar(x - 0.2, g["catastrophic"], 0.4, color="#e0311f", label="catastrophic cells")
            ax.bar(x + 0.2, g["random"], 0.4, color="gray", label="random cells")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, fontsize=8)
            ax.set_ylabel("mean |gap| / std"); ax.set_title(f"{nm}: gap to the nearest database state, per input variable")
        axes[0, 0].legend()
        fig.tight_layout(); fig.savefig(f"{prefix}_variable_gap.png", dpi=110); plt.close(fig)
        summary["variable_gap_top3"] = {nm: [names[i] for i in np.argsort(-(g["catastrophic"] - g["random"]))[:3]] for nm, g in gaps.items()}
        print("\nvariables that differ most from the nearest database state for catastrophic vs random cells (top 3):")
        for nm, top in summary["variable_gap_top3"].items():
            print(f"  {nm}: {', '.join(top)}", flush=True)

    # ---- spatial maps ----
    if maps:
        k = len(maps)
        fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(5 * ((k + 1) // 2), 9), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (nm, Hc) in zip(axes.ravel(), maps.items()):
            ax.axis("on")
            with np.errstate(invalid="ignore", divide="ignore"):
                fm = np.where(H_all > 20, 100 * Hc / H_all, np.nan)
            im = ax.imshow(fm.T, origin="lower", aspect="auto", cmap="inferno",
                           extent=[z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]])
            plt.colorbar(im, ax=ax, label="% catastrophic per bin"); ax.set_title(nm)
            ax.set_xlabel("Z [m]"); ax.set_ylabel("Y [m]")
        fig.tight_layout(); fig.savefig(f"{prefix}_maps.png", dpi=110); plt.close(fig)

    summary["species"] = sp_summary
    with open(f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n-> figures and {prefix}_summary.json", flush=True)


for m in MODELS_TO_ANALYZE:
    analyze(m)
