"""Thin the converged tail of a stochastic-reactor raw database.

Once the particle ensemble is homogeneous (temperature standard deviation below T_STD_CONVERGED), every particle carries
(almost) the same state, so keeping all of them only duplicates rows. Iterations before that point are copied untouched;
from that point on only 1 particle out of KEEP_EVERY is kept (a different subset each iteration). Groups are renumbered
contiguously, as LearningDatabase expects.
"""
import os
import shutil

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = f"{HERE}/STOCH_DTB_NH3_H2_N2_ROLLOUT_LONG"
DST_FOLDER = f"{HERE}/STOCH_DTB_NH3_H2_N2_ROLLOUT_THIN"
T_STD_CONVERGED = 1.0     # K
N_CONSECUTIVE = 20        # iterations the std must stay below the threshold
KEEP_EVERY = 100
DATASETS = ("X", "Y", "Y_multi")

os.makedirs(DST_FOLDER, exist_ok=True)
for f in ("dtb_params.yaml", "STEC_A_noAR.yaml"):
    shutil.copy(f"{SRC_FOLDER}/{f}", DST_FOLDER)

with h5py.File(f"{SRC_FOLDER}/solutions.h5", "r") as src:
    n_it = len([k for k in src.keys() if k.startswith("ITERATION_")])
    t_std = np.array([src[f"ITERATION_{i:05d}/X"][:, 0].std() for i in range(n_it)])
    below = t_std < T_STD_CONVERGED
    conv = next(i for i in range(n_it - N_CONSECUTIVE) if below[i:i + N_CONSECUTIVE].all())
    print(f">> {n_it} iterations; T std < {T_STD_CONVERGED} K from iteration {conv} on "
          f"(std there {t_std[conv]:.3f} K, last {t_std[-1]:.3f} K)")

    rows_in = rows_out = 0
    with h5py.File(f"{DST_FOLDER}/solutions.h5", "w") as dst:
        for i in range(n_it):
            g = src[f"ITERATION_{i:05d}"]
            n = g["X"].shape[0]
            if i < conv:
                idx = slice(None)
                n_kept = n
            else:
                idx = np.arange((i - conv) % KEEP_EVERY, n, KEEP_EVERY)
                n_kept = len(idx)
            og = dst.create_group(f"ITERATION_{i:05d}")
            for name in DATASETS:
                data = g[name][()][idx]
                ds = og.create_dataset(name, data=data)
                for k, v in g[name].attrs.items():
                    ds.attrs[k] = v
            rows_in += n
            rows_out += n_kept
            if i % 200 == 0:
                print(f"  iteration {i}/{n_it}", flush=True)
print(f">> rows {rows_in} -> {rows_out} ({100 * rows_out / rows_in:.1f} %), converged from iteration {conv}")
