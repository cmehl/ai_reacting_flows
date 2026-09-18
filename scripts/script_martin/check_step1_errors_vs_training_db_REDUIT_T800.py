"""For the 8 trace species flagged as >1% relative RMSE in the step_1
ANN-vs-ANN_corrected comparison (N, NNH, NH, N2H2, HO2, NH2, HNO, NO2):
extract the individual cells whose per-cell relative error (ANN vs SAGE)
exceeds 1%, and check whether the underlying thermochemical state at those
cells falls inside the model's own training database -- i.e. is the ANN
inaccurate on these points because they're an extrapolation the network
never saw, or is it inaccurate on in-distribution states too?

Training database: the T800 perspecies-2x64 model deployed as
MODEL_perspecies_2x64_thresh1e14_noclust_T800(_correction_NNICE) in this
CONVERGE case was trained from
.idea/CFD_REDUCED_MECH_A/CFD_DTB_REDUCED_MECH_A_dt5e7_rollout/dtb_rollout_thresh1e14_noclust_T800/training_data.h5
(dtb_processing.yaml: log_transform_X=1, threshold=1e-14, T_threshold=800,
nb_clusters=1). CLUSTER_0/X_train stores the STANDARDIZED network input
(species: clip to 1e-14, natural log, then StandardScaler; Temperature: raw,
also standardized) -- CLUSTER_0/Xscaler holds the (mean, var) used, so the
same transform is applied here to the CONVERGE cells before a nearest-
neighbor distance check against X_train (cKDTree, Euclidean, standardized
19-D space: Temperature + 18 species in the ARF/Cantera mechanism order).

Uses the true t=0 input state (post000001_+0.00000e+00.h5, now exported
alongside post000002 for all three of SAGE/ANN/ANN_corrected) as X -- this
is the actual state CONVERGE fed the network for this step, not a same-
timestep proxy. Confirmed identical across all three sides (common CFD
initial condition before any reaction) and identical cell ordering against
post000002 (same run, verified by direct coordinate comparison), so no
KDTree re-indexing is needed for it.

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
ANN_FILE = f"{BASE}/ANN/post000002_+5.00000e-07.h5"
INPUT_FILE = f"{BASE}/SAGE/post000001_+0.00000e+00.h5"  # t=0, the actual state the network reacted FROM
                                                          # (identical across SAGE/ANN/ANN_corrected -- common
                                                          # CFD initial condition, confirmed by direct comparison)
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
MAX_ROWS_CSV = 200000  # safety cap
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

print(f"SAGE:  {SAGE_FILE}\nANN:   {ANN_FILE}\nINPUT: {INPUT_FILE}")
coords_s, data_s, t_s = anim.load(SAGE_FILE, FIELDS)
coords_a, data_a, t_a = anim.load(ANN_FILE, FIELDS)
coords_in, data_in, t_in = anim.load(INPUT_FILE, FIELDS)
n_cells = coords_s.shape[0]
assert abs(t_s - t_a) < 1e-9
assert t_in == 0.0, f"expected the t=0 input snapshot, got t={t_in}"
assert np.array_equal(coords_in, coords_s), "post000001/post000002 cell order differs -- re-check indexing"

tree_mesh = cKDTree(coords_s)
dist, idx = tree_mesh.query(coords_a, k=1, workers=-1)
print(f"n_cells={n_cells}  max mesh re-index dist={dist.max():.3e}")

# ---------------------------------------------------------------------------
# 1) Per-cell relative error (%) for the 8 flagged species; combined mask.
# ---------------------------------------------------------------------------
mean_sage = {}
err_pct = {}
mask_any = np.zeros(n_cells, dtype=bool)
mask_by_species = {}
for name in FLAGGED_SPECIES:
    f = name_to_field[name]
    vs = data_s[f].astype(np.float64)
    va = np.empty(n_cells)
    va[idx] = data_a[f]
    m = float(vs.mean())
    mean_sage[name] = m
    e = np.abs(va - vs) / (abs(m) if abs(m) > 1e-30 else 1e-30) * 100.0
    err_pct[name] = e
    mflag = e > REL_ERR_THRESHOLD_PCT
    mask_by_species[name] = mflag
    mask_any |= mflag
    print(f"{name:6s}: {mflag.sum():>9d} / {n_cells} cells > {REL_ERR_THRESHOLD_PCT}% "
          f"({100*mflag.sum()/n_cells:.3f}% of domain)")

n_flagged = int(mask_any.sum())
print(f"\nUnion (any of the 8 species > {REL_ERR_THRESHOLD_PCT}%): {n_flagged} cells "
      f"({100*n_flagged/n_cells:.3f}% of domain)")
flagged_idx = np.where(mask_any)[0]
if flagged_idx.size > MAX_ROWS_CSV:
    print(f"capping CSV output to the {MAX_ROWS_CSV} worst cells (by max flagged-species error)")

# ---------------------------------------------------------------------------
# 2) Full physical state (Temperature + 18 species, ARF order) for the
#    flagged cells, from the true t=0 INPUT snapshot -- the actual state the
#    network reacted from, not a same-timestep proxy (post000001 is now
#    available, and identical across SAGE/ANN/ANN_corrected, confirmed above).
# ---------------------------------------------------------------------------
T_field = name_to_field["Temperature"]
T_in = data_in[T_field].astype(np.float64)
species_arrays = {sp: data_in[name_to_field[sp]].astype(np.float64) for sp in ARF_SPECIES_ORDER}

X_phys = np.column_stack([T_in[flagged_idx]] + [species_arrays[sp][flagged_idx] for sp in ARF_SPECIES_ORDER])
print(f"Extracted t=0 input state for {X_phys.shape[0]} flagged cells, {X_phys.shape[1]} dims "
      f"(Temperature + {len(ARF_SPECIES_ORDER)} species)")

# ---------------------------------------------------------------------------
# 3) Load training database (CLUSTER_0/X_train, standardized) + Xscaler,
#    apply the identical transform to the flagged states, and do a
#    nearest-neighbor distance check in the standardized 19-D space.
# ---------------------------------------------------------------------------
with h5py.File(TRAINING_DB, "r") as f:
    X_train = f["CLUSTER_0/X_train"][:]        # (n_train, 19), already standardized
    Xscaler = f["CLUSTER_0/Xscaler"][:]         # (19, 2): mean, var -- of the LOG-transformed (pre-standardization) X
print(f"Training database: X_train {X_train.shape} (CLUSTER_0, {TRAINING_DB.split('/')[-2]})")

means = Xscaler[:, 0]
stds = np.sqrt(Xscaler[:, 1])

# Apply the same transform as database_processing.py: species clipped to
# LOG_THRESHOLD then natural log; Temperature left raw; then standardize
# with the training set's own (mean, std).
X_transformed = X_phys.copy()
X_transformed[:, 1:] = np.clip(X_transformed[:, 1:], LOG_THRESHOLD, None)
X_transformed[:, 1:] = np.log(X_transformed[:, 1:])
X_standardized = (X_transformed - means) / stds

tree_train = cKDTree(X_train)
nn_dist, nn_idx = tree_train.query(X_standardized, k=1, workers=-1)

# Reference: typical training-set self nearest-neighbor distance (how far
# apart training samples normally sit from each other), to calibrate what
# "close to the training database" means for this specific model/DB.
rng = np.random.default_rng(0)
sample_idx = rng.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
self_dist, _ = tree_train.query(X_train[sample_idx], k=2, workers=-1)
self_dist = self_dist[:, 1]  # skip distance-to-self (0)
p50_ref, p95_ref, p99_ref = np.percentile(self_dist, [50, 95, 99])
print(f"\nTraining-set self nearest-neighbor distance (reference, n={sample_idx.size}): "
      f"p50={p50_ref:.3f}  p95={p95_ref:.3f}  p99={p99_ref:.3f}")

OOD_THRESHOLD = p99_ref  # a flagged cell farther than this from its nearest training sample
                          # sits outside where training samples themselves typically cluster
is_ood = nn_dist > OOD_THRESHOLD
print(f"Flagged cells farther than the training set's own p99 self-distance "
      f"(nn_dist > {OOD_THRESHOLD:.3f}): {is_ood.sum()} / {n_flagged} "
      f"({100*is_ood.sum()/n_flagged:.2f}%) -> plausibly out-of-distribution")
print(f"Flagged cells within that range: {(~is_ood).sum()} / {n_flagged} "
      f"({100*(~is_ood).sum()/n_flagged:.2f}%) -> the ANN is inaccurate here even though "
      f"the state looks like ones it was trained on")

# Per-dimension out-of-training-range flags (does the flagged cell exceed
# the training set's own min/max on this standardized coordinate?).
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)
dim_names = ["Temperature"] + ARF_SPECIES_ORDER
out_of_range = (X_standardized < train_min) | (X_standardized > train_max)
print("\nPer-dimension: fraction of flagged cells exceeding the training set's own range:")
for j, name in enumerate(dim_names):
    frac = out_of_range[:, j].mean() * 100
    if frac > 0:
        print(f"  {name:12s}: {frac:6.2f}%  (train range z=[{train_min[j]:.2f},{train_max[j]:.2f}])")

# ---------------------------------------------------------------------------
# 4) Save the full flagged-cell list to CSV.
# ---------------------------------------------------------------------------
order = np.argsort(-nn_dist)  # worst (farthest from training data) first
if flagged_idx.size > MAX_ROWS_CSV:
    order = order[:MAX_ROWS_CSV]

with open(f"flagged_cells_{OUT_PREFIX}.csv", "w") as fh:
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
print(f"\n-> flagged_cells_{OUT_PREFIX}.csv ({order.size} rows, worst nn_dist first)")

# ---------------------------------------------------------------------------
# 5) Plot: NN-distance histogram (flagged cells vs training-set reference)
#    + per-species flagged-cell counts.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(self_dist, bins=60, density=True, alpha=0.55, label="training set (self NN dist.)", color="#555555")
axes[0].hist(nn_dist, bins=60, density=True, alpha=0.55, label="flagged cells (NN dist. to training set)", color="#e0311f")
axes[0].axvline(OOD_THRESHOLD, color="black", ls="--", lw=1, label=f"p99 training self-dist. ({OOD_THRESHOLD:.2f})")
axes[0].set_xlabel("nearest-neighbor distance, standardized 19-D state space")
axes[0].set_ylabel("density")
axes[0].set_title("Are the >1%-error cells close to the training database?")
axes[0].legend(fontsize=8)

sp_counts = [mask_by_species[sp].sum() for sp in FLAGGED_SPECIES]
sp_ood_counts = [int(is_ood[np.isin(flagged_idx, np.where(mask_by_species[sp])[0])].sum()) for sp in FLAGGED_SPECIES]
x = np.arange(len(FLAGGED_SPECIES))
axes[1].bar(x, sp_counts, color="#1f77e0", label="cells > 1% rel. error")
axes[1].bar(x, sp_ood_counts, color="#e0311f", label="...of which out-of-distribution")
axes[1].set_xticks(x)
axes[1].set_xticklabels(FLAGGED_SPECIES, rotation=45, ha="right")
axes[1].set_ylabel("cell count")
axes[1].set_yscale("log")
axes[1].set_title("Flagged cells per species")
axes[1].legend(fontsize=8)

fig.suptitle(f"REDUIT_T800 step 1 (t={t_s:.3e}s): >1% error cells vs the model's training database")
fig.tight_layout()
fig.savefig(f"ood_check_{OUT_PREFIX}.png", dpi=140)
plt.close(fig)
print(f"-> ood_check_{OUT_PREFIX}.png")
print("done")
