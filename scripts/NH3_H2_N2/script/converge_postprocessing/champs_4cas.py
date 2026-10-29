"""Champs (T, 2 espèces majoritaires N2/H2O, pire espèce N2H2) pour les 4 cas simulés/testés à dt 5e-6 :
PK3-H, PK3-noH, STOCH5 rollout (coupes X=0 réelles des runs CONVERGE, dernier pas dispo ~5 ms, slices/*.npz de cmp_amr_step_stats.py)
et STOCH6 (jamais tourné dans CONVERGE : reconstruit depuis le test a priori post107, coords + T/Y_cvode/_ann de full_domain_*.h5,
coupe |x|<0.5mm rasterisée en grille 2D pour un rendu comparable). Réf. SAGE commune affichée une fois par ligne.
Écrit champs_4cas.png + champs_4cas_err.csv (RMSE/biais par champ et par cas, sur la coupe affichée)."""
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

# ---- config ----
C = '/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_REDUCED_A_NEW_INPUT_RESTARTS/converge_TEST_MODELS_32inst'
R = '/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_TFM_AMR_ROLLOUT3'
SLICE_CASES = {'PK3-H': (f'{C}/cmp_TFM_AMR_PERSP_K3ROLL_DT5E6_TOL3E5/slices/slice_000050.npz', '#1f5fa8'),
               'PK3-noH': (f'{C}/cmp_TFM_AMR_PERSP_K3ROLL_DT5E6_NOHYB/slices/slice_000050.npz', '#1f5fa8'),
               'STOCH5': (f'{C}/cmp_TFM_AMR_STOCH5ROLL_DT5E6_TOL3E4/slices/slice_000050.npz', '#2b9348')}
STOCH6_PARTS = sorted(glob.glob(f'{R}/results/parts_stoch6_dt5e6_pk3roll/*.h5'))
FIELDS = [('Temperature', 'T [K]', (800, 2150), False), ('N2', 'Y N2', (0.1, 0.78), False),
          ('H2O', 'Y H2O', (1e-3, 0.23), False), ('N2H2', 'Y N2H2 (pire espèce)', (1e-12, 3e-6), True)]
EXTENT = (-0.3775, 0.0, -0.077, 0.077)   # (z_min, z_max, y_min, y_max), slice at X=0
GRID = (300, 90)                          # STOCH6 rasterization (z, y)
OUT = 'champs_4cas.png'
# ----------------
plt.rcParams.update({'font.size': 13})


def load_slice_case(path):
    z = np.load(path)
    return {f: (z[f'ref_{f}'], z[f'ann_{f}']) for f, *_ in FIELDS}


def load_stoch6():
    xs, ys, zs, T_c, T_a, Y_c, Y_a = [], [], [], [], [], [], []
    for f in STOCH6_PARTS:
        with h5py.File(f, 'r') as h:
            c = h['coords'][:]
            m = np.abs(c[:, 0]) < 5e-4
            if not m.any():
                continue
            xs.append(c[m, 1]); ys.append(c[m, 2])
            T_c.append(h['T_cvode'][0][m]); T_a.append(h['T_ann'][0][m])
            Y_c.append(h['Y_cvode'][0][m]); Y_a.append(h['Y_ann'][0][m])
            if 'species' not in globals():
                globals()['species'] = list(h.attrs['species'])
    y, z = np.concatenate(xs), np.concatenate(ys)
    Tc, Ta = np.concatenate(T_c), np.concatenate(T_a)
    Yc, Ya = np.concatenate(Y_c, axis=0), np.concatenate(Y_a, axis=0)
    zi = np.linspace(EXTENT[0], EXTENT[1], GRID[0]); yi = np.linspace(EXTENT[2], EXTENT[3], GRID[1])
    iz = np.clip(np.searchsorted(zi, z) - 1, 0, GRID[0] - 2)
    iy = np.clip(np.searchsorted(yi, y) - 1, 0, GRID[1] - 2)
    def grid(v):
        g = np.full(GRID[::-1], np.nan); cnt = np.zeros(GRID[::-1])
        np.add.at(g, (iy, iz), np.nan_to_num(v)); np.add.at(cnt, (iy, iz), 1)
        with np.errstate(invalid='ignore'):
            return np.where(cnt > 0, g / np.maximum(cnt, 1), np.nan)
    out = {'Temperature': (grid(Tc), grid(Ta))}
    for f, *_ in FIELDS[1:]:
        i = species.index(f)
        out[f] = (grid(Yc[:, i]), grid(Ya[:, i]))
    return out


DATA = {k: load_slice_case(p) for k, (p, _) in SLICE_CASES.items()}
DATA['STOCH6'] = load_stoch6()
CASES = ['PK3-H', 'PK3-noH', 'STOCH5', 'STOCH6']

fig, ax = plt.subplots(len(FIELDS), 1 + len(CASES), figsize=(23, 13.5))
rows = []
for r, (fname, label, (vmin, vmax), log) in enumerate(FIELDS):
    norm = LogNorm(vmin, vmax) if log else Normalize(vmin, vmax)
    cmap = 'viridis'
    refimg = DATA['PK3-H'][fname][0]
    im = ax[r, 0].imshow(np.clip(refimg, vmin, vmax), extent=EXTENT, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax[r, 0].set_title('SAGE (réf.)' if r == 0 else '', fontsize=14)
    ax[r, 0].set_ylabel(label, fontsize=13)
    for c, case in enumerate(CASES):
        ref, ann = DATA[case][fname]
        a = ax[r, c + 1]
        a.imshow(np.clip(ann, vmin, vmax), extent=EXTENT, origin='lower', aspect='auto', cmap=cmap, norm=norm)
        if r == 0:
            a.set_title(case, fontsize=15, fontweight='bold')
        d = ann - ref if not log else np.log10(np.clip(ann, 1e-14, None)) - np.log10(np.clip(ref, 1e-14, None))
        rmse = float(np.sqrt(np.nanmean(d[np.isfinite(d)] ** 2))) if np.isfinite(d).any() else np.nan
        bias = float(np.nanmean(d[np.isfinite(d)])) if np.isfinite(d).any() else np.nan
        rows.append({'champ': fname, 'cas': case, 'rmse': rmse, 'biais': bias, 'unite': ('K' if fname == 'Temperature' else 'dex' if log else 'Y')})
    fig.colorbar(im, ax=ax[r, :], fraction=0.012, pad=0.006)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Coupe X = 0, 4 cas dt 5e-6 : température, espèces majoritaires (N2, H2O) et pire espèce (N2H2)\n"
             "PK3-H / PK3-noH / STOCH5 : run CONVERGE réel (~5 ms)  —  STOCH6 : jamais tourné dans CONVERGE, reconstruit depuis le test a priori post107", fontsize=15)
fig.tight_layout(rect=(0, 0, 0.99, 0.93))
fig.savefig(OUT, dpi=140)
pd.DataFrame(rows).to_csv('champs_4cas_err.csv', index=False)
print(pd.DataFrame(rows).pivot(index='champ', columns='cas', values='rmse').to_string())
