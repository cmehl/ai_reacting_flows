"""Species mass fraction profiles over time: PK3-H / PK3-noH / STOCH5 rollout vs SAGE dt 5e-6, same raw-file loading as recap_3cas.py
(thermo.out/species_mass.out, no heavy postpro). One panel per species (18), domain-mean mass fraction vs t, log scale (spans decades
for trace species), SAGE in black, PK3-H/PK3-noH in blue (solid/dotted), STOCH5 in green, capped at 100 ms (STOCH5: until its current
progress). Writes species_profiles_3cas.png + species_profiles_3cas_table.csv (final value + mean relative error per species/model)."""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- config ----
T_ROOT = '/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/TEST_MODELS_32inst'
REF = 'SAGE_DT5E6'
RUNS = {'PK3-H': ('PERSP_K3ROLL_DT5E6_TOL3E5', '#1f5fa8', '-'),
        'PK3-noH': ('PERSP_K3ROLL_DT5E6_NOHYB', '#1f5fa8', ':'),
        'STOCH5': ('STOCH5ROLL_DT5E6_TOL3E4', '#2b9348', '-')}
SPECIES = ['N2', 'H2', 'H', 'O2', 'O', 'H2O', 'OH', 'HO2', 'NO', 'NH3', 'NH2', 'NH', 'N', 'NNH', 'N2H2', 'HNO', 'NO2', 'N2O']
T_MAX_MS = 100.0
# ----------------


def segments(case, sub):
    base = f'{T_ROOT}/CAS_TFM_AMR_{case}'
    segs = [f'{base}/outputs_original/{sub}']
    n = 1
    while os.path.exists(f'{base}/outputs_restart{n}/{sub}'):
        segs.append(f'{base}/outputs_restart{n}/{sub}')
        n += 1
    return segs


def read_out(case, name):
    segs = segments(case, f'stream0/{name}')
    cols = [l for l in open(segs[0]) if l.startswith('#')][2].lstrip('#').split()
    parts = [pd.DataFrame(np.loadtxt(f, comments='#', ndmin=2)[:, :len(cols)], columns=cols) for f in segs]
    parts = [p[p.Time < q.Time.min()] for p, q in zip(parts[:-1], parts[1:])] + parts[-1:]
    d = pd.concat(parts).drop_duplicates('Time').set_index('Time')
    return d[d.index <= T_MAX_MS * 1e-3]


def load_species(case):
    return read_out(case, 'species_mass.out')[SPECIES]


ref = load_species(REF)
data = {lab: load_species(case) for lab, (case, c, ls) in RUNS.items()}

plt.rcParams.update({'font.size': 11})
fig, axs = plt.subplots(3, 6, figsize=(26, 12), constrained_layout=True)
rows = []
for sp, ax in zip(SPECIES, axs.flat):
    tr = 1e3 * ref.index.values
    ax.plot(tr, ref[sp].values, color='k', lw=1.3, label='SAGE (réf.)')
    for lab, (case, c, ls) in RUNS.items():
        a = data[lab][sp].reindex(ref.index, method='nearest', tolerance=2.5e-6).dropna()
        t = 1e3 * a.index.values
        ax.plot(t, a.values, color=c, ls=ls, lw=1.1, alpha=0.9, label=lab)
        r = ref[sp].reindex(a.index)
        rel = 100 * np.abs(a.values / np.maximum(r.values, 1e-30) - 1)
        rows.append({'espèce': sp, 'modèle': lab, 't_fin [ms]': t.max() if len(t) else np.nan,
                      'Y_fin': a.values[-1] if len(a) else np.nan, 'erreur rel. moyenne [%]': np.nanmean(rel)})
    ax.set_title(sp, fontsize=13)
    ax.set_yscale('log')
    ax.set_xlim(0, T_MAX_MS)
    ax.grid(alpha=0.3, which='both')
for ax in axs[-1]:
    ax.set_xlabel('t [ms]')
axs.flat[0].legend(fontsize=9, loc='best')
fig.suptitle('Fraction massique moyenne du domaine par espèce — PK3-H / PK3-noH / STOCH5 rollout vs SAGE, 0–100 ms (STOCH5 : jusqu’à son avancement actuel)', fontsize=15)
fig.savefig('species_profiles_3cas.png', dpi=130, facecolor='white')

tab = pd.DataFrame(rows)
tab.to_csv('species_profiles_3cas_table.csv', index=False)
pd.set_option('display.width', 220)
print(tab.pivot(index='espèce', columns='modèle', values='erreur rel. moyenne [%]').round(3).to_string())
