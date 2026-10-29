"""Streaming replay of LearningDatabase.undersample_1D("Temperature", min_count=50000, seed=1991) on the TFM_AMR raw db.

LearningDatabase loads ITERATION_00000..N in order, apply_temperature_threshold is a no-op (the raw build already kept
T > 800 K only), so replaying the exact same algorithm on the concatenated Temperature column reproduces the exact set of
states kept by the mc50000 processing of MLP_CFD_TFM_AMR_ALLDATA (81,222,200 rows) without loading the 271 GB db.
Writes selected_T1D_mc50000.npz: sorted global row indices + per-group offsets.
"""
import time

import h5py
import numpy as np

import ai_reacting_flows.tools.utilities as utils

# ---- config ----
RAW = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_TFM_AMR_ALLDATA/CFD_DTB_TFM_AMR_ALLDATA_dt5e7/solutions.h5"
T_THRESHOLD = 800.0
MIN_COUNT = 50000
SEED = 1991
EXPECTED = 81_222_200  # rows kept by the mc50000 processing (its log)
OUT = "selected_T1D_mc50000.npz"
# ----------------

t0 = time.time()
with h5py.File(RAW, "r") as h:
    n_groups = len(h.keys())
    T_list = [h[f"ITERATION_{i:05d}/X"][:, 0] for i in range(n_groups)]
sizes = np.array([t.size for t in T_list])
offs = np.concatenate([[0], np.cumsum(sizes)])
v = np.concatenate(T_list)
del T_list
assert (v > T_THRESHOLD).all(), "raw db has states <= T_threshold: threshold would not be a no-op"
print(f">> {n_groups} groups, {v.size} states read in {time.time() - t0:.0f}s", flush=True)

# ---- exact replay of undersample_1D ----
rng = np.random.default_rng(SEED)
binwidth = utils.smart_binwidth(v)
print("Computed binwidth is", binwidth, flush=True)
edges = np.arange(v.min(), v.max() + binwidth, binwidth)
counts, _ = np.histogram(v, bins=edges)
bin_id = np.digitize(v, edges[:-1], right=False)
occupied = counts > 0
eligible_mask = occupied & (counts >= MIN_COUNT)
n_keep = counts[eligible_mask].min()
print(f"Subsampling to {n_keep} samples per eligible bin; sparse bins kept whole: {(occupied & ~eligible_mask).sum()}",
      flush=True)
order = np.argsort(bin_id, kind="stable")  # indices grouped by bin, ascending inside each bin (= np.where order)
bounds = np.searchsorted(bin_id[order], np.arange(1, len(edges) + 1))
selected = []
for b in range(1, len(edges)):
    idx = order[bounds[b - 1]:bounds[b]]
    if len(idx) == 0:
        continue
    if eligible_mask[b - 1]:
        selected.append(rng.choice(idx, size=n_keep, replace=False))
    else:
        selected.append(idx)
sel = np.sort(np.concatenate(selected))
print(f">> selected {sel.size} rows ({100 * sel.size / v.size:.4f}%)  {time.time() - t0:.0f}s", flush=True)
assert sel.size == EXPECTED, f"{sel.size} != {EXPECTED}: not the same selection as the mc50000 processing"
assert np.unique(sel).size == sel.size
np.savez(OUT, sel=sel, offs=offs)
print("wrote", OUT)
