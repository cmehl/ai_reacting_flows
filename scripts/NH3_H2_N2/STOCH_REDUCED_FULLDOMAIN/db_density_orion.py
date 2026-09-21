"""Is the flame-zone region of the catastrophic cells thinly covered by the reactor database in the (T, ln Y_sp) plane?

Orion version (run with sbatch): inputs are copied next to this script under DATA_DIR. For each species it counts, for
5000 catastrophic flame cells (1500 K < T0 < 2050 K, |ANN-CVODE| > domain-mean CVODE) and 5000 non-catastrophic flame
cells, how many training states of the model's reactor database lie in the same (T, ln Y) window
(+-DT_WINDOW K, +-DL_WINDOW in ln Y, i.e. a factor 2 for 0.7). Config is top-of-file UPPERCASE vars.
"""
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

# ---- config ----
DATA_DIR = "/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea/NH3_H2_N2_FULLDOMAIN_ANALYSIS"
REF_FILE = f"{DATA_DIR}/reference_cvode_full_domain.h5"
ANN_FILE = f"{DATA_DIR}/old_perspecies_REF_ann.h5"
TRAINING_H5 = f"{DATA_DIR}/training_data.h5"
LOG_THRESHOLD = 1e-10
SPECIES_LIST = ["N", "NH", "NNH", "NH2", "N2H2", "HNO"]
DT_WINDOW = 50.0
DL_WINDOW = 0.7
N_CELLS = 5000
T_FLAME = (1500.0, 2050.0)
SEED = 0
# -----------------

rng = np.random.default_rng(SEED)
Xs = []
with h5py.File(TRAINING_H5) as f:
    for c in sorted(k for k in f.keys() if k.startswith("CLUSTER_")):
        sc = f[c]["Xscaler"][:]
        Xs.append(f[c]["X_train"][:] * np.sqrt(sc[:, 1]) + sc[:, 0])
X = np.vstack(Xs)
print(f"database: {X.shape[0]} training states", flush=True)
with h5py.File(REF_FILE) as f:
    sp = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["species"]]
    T0, Y0, Yc = f["T_ini"][()], f["Y_ini"][()], f["Y_cvode"][()]
with h5py.File(ANN_FILE) as f:
    Ya = f["Y_ann"][()]

flame = (T0 > T_FLAME[0]) & (T0 < T_FLAME[1])
print(f"{flame.sum()} flame cells ({100 * flame.mean():.2f}% of reacted)", flush=True)
print(f"DB rows in the same (T, ln Y) window (+-{DT_WINDOW:g} K, +-{DL_WINDOW:g} in ln Y) for catastrophic vs "
      "non-catastrophic flame cells:", flush=True)
for name in SPECIES_LIST:
    j = sp.index(name)
    cv, an = Yc[:, j], Ya[:, j]
    cata = np.abs(an - cv) > np.abs(cv).mean()
    tree = cKDTree(np.column_stack([X[:, 0] / DT_WINDOW, X[:, 1 + j] / DL_WINDOW]))
    out = {}
    for label, m in (("cata", cata & flame), ("other", (~cata) & flame)):
        idx = rng.choice(np.where(m)[0], min(N_CELLS, int(m.sum())), replace=False)
        q = np.column_stack([T0[idx] / DT_WINDOW, np.log(np.clip(Y0[idx, j], LOG_THRESHOLD, None)) / DL_WINDOW])
        # return_length: only the neighbour COUNT per cell (the neighbour lists themselves ran the first job out of memory)
        out[label] = np.asarray(tree.query_ball_point(q, r=1.0, p=np.inf, workers=-1, return_length=True))
    a, b = out["cata"], out["other"]
    print(f"  {name:5s} DB neighbours: catastrophic median {np.median(a):7.0f} (p10 {np.percentile(a, 10):6.0f}, "
          f"empty {100 * (a == 0).mean():4.1f}%)  |  non-catastrophic flame median {np.median(b):7.0f} "
          f"(p10 {np.percentile(b, 10):6.0f}, empty {100 * (b == 0).mean():4.1f}%)", flush=True)
print("DENSITY DONE", flush=True)
