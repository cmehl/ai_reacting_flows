"""For the 8 trace species flagged as >1% relative RMSE in the step_1
ANN-variants-vs-SAGE comparison (N, NNH, NH, N2H2, HO2, NH2, HNO, NO2):
for EACH ANN variant (ANN, ANN_corrected, ANN_corrected_2, ...), extract the
individual cells whose per-cell relative error exceeds 1%, and check whether
the underlying thermochemical state at those cells falls inside the model's
own training database -- i.e. is each variant inaccurate on these points
because they're an extrapolation the network never saw, or is it inaccurate
on in-distribution states too? Also compares the three variants against each
other on this metric.

Training database: the T800 perspecies-2x64 models deployed as
MODEL_perspecies_2x64_thresh1e14_noclust_T800(_correction_NNICE[_2]) in this
CONVERGE case were trained from
.idea/CFD_REDUCED_MECH_A/CFD_DTB_REDUCED_MECH_A_dt5e7_rollout/dtb_rollout_thresh1e14_noclust_T800/training_data.h5
(dtb_processing.yaml: log_transform_X=1, threshold=1e-14, T_threshold=800,
nb_clusters=1). CLUSTER_0/X_train stores the STANDARDIZED network input
(species: clip to 1e-14, natural log, then StandardScaler; Temperature: raw,
also standardized) -- CLUSTER_0/Xscaler holds the (mean, var) used, so the
same transform is applied here to the CONVERGE cells before a nearest-
neighbor distance check against X_train (cKDTree, Euclidean, standardized
19-D space: Temperature + 18 species in the ARF/Cantera mechanism order).

Uses the true t=0 input state (post000001_+0.00000e+00.h5, exported
alongside post000002 for SAGE and every variant) as X -- this is the actual
state CONVERGE fed the network for this step, not a same-timestep proxy.
Confirmed identical across all sides (common CFD initial condition before
any reaction) and identical cell ordering against post000002 (same run,
verified by direct coordinate comparison), so no KDTree re-indexing is
needed for it -- only SAGE's post000001 is read.

Config is top-of-file UPPERCASE vars (no argparse).
"""
import importlib.util
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

# ---- config ----
ANIMATE_SCRIPT = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/scripts/script_martin/animate_ann_sage_hybrid.py"
BASE = "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/REDUIT_T800/paraview/step_1"
SAGE_FILE = f"{BASE}/SAGE/post000002_+5.00000e-07.h5"
INPUT_FILE = f"{BASE}/SAGE/post000001_+0.00000e+00.h5"  # t=0, the actual state the network reacted FROM
VARIANTS = [
    ("ANN", f"{BASE}/ANN/post000002_+5.00000e-07.h5", "#e0311f"),
    ("ANN_corrected", f"{BASE}/ANN_corrected/post000002_+5.00000e-07.h5", "#1f77e0"),
    ("ANN_corrected_2", f"{BASE}/ANN_corrected_2/post000002_+5.00000e-07.h5", "#2fa84a"),
]
TRAINING_DB = ("/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea/"
               "CFD_REDUCED_MECH_A/CFD_DTB_REDUCED_MECH_A_dt5e7_rollout/"
               "dtb_rollout_thresh1e14_noclust_T800/training_data.h5")
FLAGGED_SPECIES = ["N", "NNH", "NH", "N2H2", "HO2", "NH2", "HNO", "NO2"]  # >1% rel. RMSE at step 1
REL_ERR_THRESHOLD_PCT = 1.0
LOG_THRESHOLD = 1e-14
# ARF/Cantera mechanism species order (matches X_train's column order and
# state_T800_compare_arf_nnice.txt)
ARF_SPECIES_ORDER = ["N2", "H2", "H", "O2", "O", "H2O", "OH", "HO2", "NO", "NH3",
                      "NH2", "NH", "N", "NNH", "N2H2", "HNO", "NO2", "N2O"]
OUT_PREFIX = "REDUIT_T800_step1_vs_trainingDB"
MAX_ROWS_CSV = 200000  # safety cap, per variant
# -----------------

spec = importlib.util.spec_from_file_location("animate_ann_sage_hybrid", ANIMATE_SCRIPT)
anim = importlib.util.module_from_spec(spec)
sys.modules["animate_ann_sage_hybrid"] = anim
spec.loader.exec_module(anim)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = anim.FIELDS  # ["TEMPERATURE", "MASSFRAC_<name>", ...]
labels = [anim.field_label(f) for f in FIELDS]
name_to_field = dict(zip(labels, FIELDS))
n_variants = len(VARIANTS)

print(f"SAGE:  {SAGE_FILE}\nINPUT: {INPUT_FILE}")
coords_s, data_s, t_s = anim.load(SAGE_FILE, FIELDS)
coords_in, data_in, t_in = anim.load(INPUT_FILE, FIELDS)
n_cells = coords_s.shape[0]
assert t_in == 0.0, f"expected the t=0 input snapshot, got t={t_in}"
assert np.array_equal(coords_in, coords_s), "post000001/post000002 cell order differs -- re-check indexing"

# t=0 input state (Temperature + 18 species, ARF order) -- common to every variant.
T_field = name_to_field["Temperature"]
T_in = data_in[T_field].astype(np.float64)
species_arrays_in = {sp: data_in[name_to_field[sp]].astype(np.float64) for sp in ARF_SPECIES_ORDER}
dim_names = ["Temperature"] + ARF_SPECIES_ORDER

# Training database: load once, shared by every variant.
with h5py.File(TRAINING_DB, "r") as f:
    X_train = f["CLUSTER_0/X_train"][:]        # (n_train, 19), already standardized
    Xscaler = f["CLUSTER_0/Xscaler"][:]         # (19, 2): mean, var (of the LOG-transformed, pre-standardization X)
print(f"Training database: X_train {X_train.shape} (CLUSTER_0, {TRAINING_DB.split('/')[-2]})")
means = Xscaler[:, 0]
stds = np.sqrt(Xscaler[:, 1])
tree_train = cKDTree(X_train)

rng = np.random.default_rng(0)
sample_idx = rng.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
self_dist, _ = tree_train.query(X_train[sample_idx], k=2, workers=-1)
self_dist = self_dist[:, 1]
p50_ref, p95_ref, p99_ref = np.percentile(self_dist, [50, 95, 99])
OOD_THRESHOLD = p99_ref
print(f"Training-set self nearest-neighbor distance (reference, n={sample_idx.size}): "
      f"p50={p50_ref:.3f}  p95={p95_ref:.3f}  p99={p99_ref:.3f}\n")

tree_mesh = cKDTree(coords_s)

results = {}  # label -> dict of everything needed downstream
for label, fp, color in VARIANTS:
    print(f"=== {label} ({fp}) ===")
    coords_v, data_v, t_v = anim.load(fp, FIELDS)
    assert abs(t_v - t_s) < 1e-9
    dist_mesh, idx_v = tree_mesh.query(coords_v, k=1, workers=-1)
    print(f"n_cells={n_cells}  max mesh re-index dist={dist_mesh.max():.3e}")

    # 1) Per-cell relative error (%) for the 8 flagged species; combined mask.
    err_pct = {}
    mask_any = np.zeros(n_cells, dtype=bool)
    mask_by_species = {}
    for name in FLAGGED_SPECIES:
        f = name_to_field[name]
        vs = data_s[f].astype(np.float64)
        va = np.empty(n_cells)
        va[idx_v] = data_v[f]
        m = float(vs.mean())
        e = np.abs(va - vs) / (abs(m) if abs(m) > 1e-30 else 1e-30) * 100.0
        err_pct[name] = e
        mflag = e > REL_ERR_THRESHOLD_PCT
        mask_by_species[name] = mflag
        mask_any |= mflag
        print(f"  {name:6s}: {mflag.sum():>9d} / {n_cells} cells > {REL_ERR_THRESHOLD_PCT}% "
              f"({100*mflag.sum()/n_cells:.3f}% of domain)")

    n_flagged = int(mask_any.sum())
    flagged_idx = np.where(mask_any)[0]
    print(f"  Union: {n_flagged} cells ({100*n_flagged/n_cells:.3f}% of domain)")

    # 2) t=0 input state for the flagged cells (shared array, common to all variants).
    X_phys = np.column_stack(
        [T_in[flagged_idx]] + [species_arrays_in[sp][flagged_idx] for sp in ARF_SPECIES_ORDER]
    )

    # 3) Transform + nearest-neighbor distance to the training set.
    X_transformed = X_phys.copy()
    X_transformed[:, 1:] = np.clip(X_transformed[:, 1:], LOG_THRESHOLD, None)
    X_transformed[:, 1:] = np.log(X_transformed[:, 1:])
    X_standardized = (X_transformed - means) / stds
    nn_dist, _ = tree_train.query(X_standardized, k=1, workers=-1)
    is_ood = nn_dist > OOD_THRESHOLD
    pct_close = 100 * (~is_ood).sum() / n_flagged if n_flagged else float("nan")
    pct_far = 100 * is_ood.sum() / n_flagged if n_flagged else float("nan")
    print(f"  Close to training data: {(~is_ood).sum()}/{n_flagged} ({pct_close:.2f}%)  "
          f"Far (plausibly OOD): {is_ood.sum()}/{n_flagged} ({pct_far:.2f}%)")

    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    out_of_range = (X_standardized < train_min) | (X_standardized > train_max)

    results[label] = dict(
        color=color, n_flagged=n_flagged, flagged_idx=flagged_idx, err_pct=err_pct,
        mask_by_species=mask_by_species, nn_dist=nn_dist, is_ood=is_ood,
        pct_close=pct_close, pct_far=pct_far, X_phys=X_phys, out_of_range=out_of_range,
    )

    order = np.argsort(-nn_dist)
    if flagged_idx.size > MAX_ROWS_CSV:
        order = order[:MAX_ROWS_CSV]
        print(f"  capping CSV to the {MAX_ROWS_CSV} worst cells (by training distance)")
    with open(f"flagged_cells_{OUT_PREFIX}_{label}.csv", "w") as fh:
        header = (["cell_idx", "X", "Y", "Z", "nn_dist_to_training", "is_ood"]
                   + [f"err_pct_{sp}" for sp in FLAGGED_SPECIES]
                   + [f"state_{d}" for d in dim_names]
                   + [f"oor_{d}" for d in dim_names])
        fh.write(",".join(header) + "\n")
        for k in order:
            ci = flagged_idx[k]
            row = ([ci, coords_s[ci, 0], coords_s[ci, 1], coords_s[ci, 2], nn_dist[k], int(is_ood[k])]
                   + [err_pct[sp][ci] for sp in FLAGGED_SPECIES]
                   + list(X_phys[k])
                   + [int(x) for x in out_of_range[k]])
            fh.write(",".join(str(x) for x in row) + "\n")
    print(f"  -> flagged_cells_{OUT_PREFIX}_{label}.csv ({order.size} rows, worst nn_dist first)\n")

# ---------------------------------------------------------------------------
# Cross-variant summary.
# ---------------------------------------------------------------------------
print("=== CROSS-VARIANT SUMMARY ===")
print(f"{'variant':18s}  {'flagged':>10s}  {'% domain':>9s}  {'% close':>8s}  {'% far (OOD)':>11s}")
with open(f"summary_by_variant_{OUT_PREFIX}.csv", "w") as fh:
    fh.write("variant,n_flagged,pct_domain,pct_close,pct_far\n")
    for label, _, _ in VARIANTS:
        r = results[label]
        pct_domain = 100 * r["n_flagged"] / n_cells
        print(f"{label:18s}  {r['n_flagged']:10d}  {pct_domain:8.3f}%  {r['pct_close']:7.2f}%  {r['pct_far']:10.2f}%")
        fh.write(f"{label},{r['n_flagged']},{pct_domain},{r['pct_close']},{r['pct_far']}\n")
print(f"-> summary_by_variant_{OUT_PREFIX}.csv")

for label, _, _ in VARIANTS:
    r = results[label]
    n_flagged = r["n_flagged"]
    is_ood = r["is_ood"]
    X_phys = r["X_phys"]
    T_col = X_phys[:, 0]
    cold = int(((T_col < 800) & is_ood).sum())
    hot = int(is_ood.sum() - cold)
    print(f"{label}: of the {is_ood.sum()} far/OOD cells, {cold} ({100*cold/max(is_ood.sum(),1):.1f}%) are "
          f"cold (<800K, below the training mask), {hot} ({100*hot/max(is_ood.sum(),1):.1f}%) are >=800K "
          f"(nominally in-range but in an under-sampled state combination)")

# ---------------------------------------------------------------------------
# Plot: NN-distance histograms (one per variant + training reference) and
# per-species flagged-cell counts grouped by variant.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

axes[0].hist(self_dist, bins=60, density=True, alpha=0.5, label="training set (self NN dist.)", color="#555555")
for label, _, color in VARIANTS:
    axes[0].hist(results[label]["nn_dist"], bins=60, density=True, alpha=0.45,
                 label=f"{label} flagged cells", color=color)
axes[0].axvline(OOD_THRESHOLD, color="black", ls="--", lw=1, label=f"p99 training self-dist. ({OOD_THRESHOLD:.2f})")
axes[0].set_xlabel("nearest-neighbor distance, standardized 19-D state space")
axes[0].set_ylabel("density")
axes[0].set_title("Are the >1%-error cells close to the training database?")
axes[0].legend(fontsize=8)

x = np.arange(len(FLAGGED_SPECIES))
w = 0.8 / n_variants
for k, (label, _, color) in enumerate(VARIANTS):
    counts = [int(results[label]["mask_by_species"][sp].sum()) for sp in FLAGGED_SPECIES]
    offset = (k - (n_variants - 1) / 2) * w
    axes[1].bar(x + offset, counts, width=w, color=color, label=label)
axes[1].set_xticks(x)
axes[1].set_xticklabels(FLAGGED_SPECIES, rotation=45, ha="right")
axes[1].set_ylabel("cell count > 1% rel. error")
axes[1].set_yscale("log")
axes[1].set_title("Flagged cells per species, per variant")
axes[1].legend(fontsize=8)

fig.suptitle(f"REDUIT_T800 step 1 (t={t_s:.3e}s): >1% error cells vs the training database, {n_variants} ANN variants")
fig.tight_layout()
fig.savefig(f"ood_check_{OUT_PREFIX}.png", dpi=140)
plt.close(fig)
print(f"\n-> ood_check_{OUT_PREFIX}.png")
print("done")
