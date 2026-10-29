"""Tableau + courbes récap PK3-H / PK3-noH / STOCH5 (rollout) vs SAGE dt 5e-6, capé à 100 ms (même si STOCH5 n'est pas terminé) :
erreur masse moyenne (18 esp.), delta T moyenne, erreur HRR, part d'appels CVODE, Q produit par CVODE, gain de vitesse (sagetime),
écarts éléments H/N/O (fraction massique), erreur <Y|T> (lue séparément depuis les _LONG stat_all_steps.csv, cf. recap_ythr.py).
Tout lu depuis les fichiers CONVERGE bruts (thermo.out, species_mass.out, converge.log) : aucun post-traitement lourd requis ici.
Écrit recap_pk3_table.csv (hors <Y|T>) + recap_pk3_curves.png."""
import os
import re

import cantera as ct
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- config ----
T_ROOT = '/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/TEST_MODELS_32inst'
MECH = '/ifpengpfs/scratch/ifpen/kotlarcm/AI/ai_reacting_flows-master_cedric/.idea3/MLP_CFD_REDUCED_A_NEW_INPUT_RESTARTS/converge_TEST_MODELS_32inst/debug_TFM_AMR_mc50000/results/mech_converge.yaml'
REF = 'SAGE_DT5E6'
RUNS = {'PK3-H': ('PERSP_K3ROLL_DT5E6_TOL3E5', '#1f5fa8', '-'),
        'PK3-noH': ('PERSP_K3ROLL_DT5E6_NOHYB', '#2b9348', ':')}
SPECIES = ['N2', 'H2', 'H', 'O2', 'O', 'H2O', 'OH', 'HO2', 'NO', 'NH3', 'NH2', 'NH', 'N', 'NNH', 'N2H2', 'HNO', 'NO2', 'N2O']
ELEM_COL = {'O': '#1f5fa8', 'H': '#2b9348', 'N': '#7b4fa0'}
T_MAX_MS = 200.0          # ne jamais dépasser, même si une base dure plus longtemps
SMOOTH_MS = 1.0
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


def load_globals(case):
    return pd.concat([read_out(case, 'species_mass.out')[SPECIES], read_out(case, 'thermo.out')[['Mean_Temp', 'HR_Rate', 'Mass']]], axis=1).dropna()


gas = ct.Solution(MECH)
E = {e: np.array([gas.n_atoms(s, e) * gas.atomic_weight(gas.element_index(e)) / gas.molecular_weights[gas.species_index(s)] for s in SPECIES]) for e in 'OHN'}


def elem_frac_err(a, r):
    out = {}
    for e in 'OHN':
        ea = a[SPECIES].values @ E[e]
        er = r[SPECIES].values @ E[e]
        out[e] = 100 * ((ea / a.Mass.values) / (er / r.Mass.values) - 1)
    return out


def cvode_call_frac(case):
    """Part des appels chimie envoyés à CVODE par pas, depuis converge.log (compteurs ANN/CVODE), capé à T_MAX_MS."""
    out_t, out_f = [], []
    t0 = 0.0
    for seg in segments(case, 'converge.log'):
        txt = open(seg, errors='ignore').read()
        na = np.array([int(x) for x in re.findall(r'Successful calls to ANN = (-?\d+)', txt)], float)
        nv = np.array([int(x) for x in re.findall(r'Alternative calls to CVODE = (-?\d+)', txt)], float)
        tt = np.array([float(x) for x in re.findall(r'ncyc\s+\d+;\s+Time\s+([0-9.eE+-]+)', txt)])
        n = min(len(na), len(nv), len(tt) - 1 if len(tt) else 0)
        if n < 2:
            continue
        da, dv = np.diff(na[:n + 1]), np.diff(nv[:n + 1])
        ok = (da > 0) & (dv >= 0)
        f = np.where(ok, dv / np.where(ok, da + dv, 1) * 100, np.nan)
        out_t.append(tt[1:n + 1]); out_f.append(f)
    if not out_t:
        return pd.Series(dtype=float)
    t = np.concatenate(out_t); f = np.concatenate(out_f)
    s = pd.Series(f, index=t).sort_index()
    s = s[~s.index.duplicated(keep='last')]
    return s[s.index <= T_MAX_MS * 1e-3]


def speed_sagetime(case):
    """sagetime/pas vs t depuis converge.log, capé à T_MAX_MS."""
    out_t, out_s = [], []
    for i, seg in enumerate(segments(case, 'converge.log')):
        txt = open(seg, errors='ignore').read()
        rows = []
        for blk in re.split(r'(?=^Total Time Step Run Time:)', txt, flags=re.M)[1:]:
            mtime = re.search(r'ncyc\s+\d+;\s+Time\s+([0-9.eE+-]+)', blk)
            ms = re.search(r'sagetime:\s+([0-9.]+)\s+[0-9.]+ seconds', blk)
            if mtime and ms:
                rows.append((float(mtime.group(1)), float(ms.group(1))))
        if not rows:
            continue
        d = pd.DataFrame(rows, columns=['t', 'sage']).iloc[10 if i == 0 else 0:]
        out_t.append(d.t.values); out_s.append(d.sage.values)
    t = np.concatenate(out_t); s = np.concatenate(out_s)
    ser = pd.Series(s, index=t).sort_index()
    ser = ser[~ser.index.duplicated(keep='last')]
    return ser[ser.index <= T_MAX_MS * 1e-3]


ref = load_globals(REF)
sage_sage = speed_sagetime(REF)

rows = {}
plt.rcParams.update({'font.size': 13})
fig, axs = plt.subplots(2, 4, figsize=(23, 9), constrained_layout=True)
panels = ['err', 'dT', 'HRR', 'cvode', 'O', 'H', 'N', 'speed']
titles = {'err': 'Erreur de masse moyenne, 18 esp. [%]', 'dT': 'T moyenne − SAGE [K]',
          'HRR': f'Dégagement de chaleur / SAGE − 1 [%] (moy. {SMOOTH_MS:g} ms)', 'cvode': 'Appels chimie en CVODE [%]',
          'O': 'O : fraction massique / SAGE − 1 [%]', 'H': 'H : fraction massique / SAGE − 1 [%]',
          'N': 'N : fraction massique / SAGE − 1 [%]', 'speed': 'Vitesse chimie (sagetime), cumulée depuis t=0'}
ax_of = dict(zip(panels, axs.flat))

for lab, (case, c, ls) in RUNS.items():
    a = load_globals(case).reindex(ref.index, method='nearest', tolerance=2.5e-6).dropna()
    r = ref.loc[a.index]
    t = 1e3 * a.index.values
    rel = 100 * (a[SPECIES] / r[SPECIES] - 1)
    err = rel.abs().mean(axis=1)
    dT = a.Mean_Temp - r.Mean_Temp
    hrr = (100 * (a.HR_Rate / r.HR_Rate - 1)).rolling(int(SMOOTH_MS * 1e-3 / 5e-6), center=True, min_periods=1).mean()
    ef = elem_frac_err(a, r)
    cv = cvode_call_frac(case)
    sg = speed_sagetime(case)
    su = sage_sage.reindex(sg.index, method='nearest', tolerance=5e-6).dropna()
    sg2 = sg.reindex(su.index)
    speed_cum = (su.expanding().mean() / sg2.expanding().mean())

    for k, y, tt in [('err', err.values, t), ('dT', dT.values, t), ('HRR', hrr.values, t),
                      ('O', ef['O'], t), ('H', ef['H'], t), ('N', ef['N'], t)]:
        ax_of[k].plot(tt, y, color=c, ls=ls, lw=1.8, label=f'{lab}, jusqu’à {tt.max():.0f} ms')
    ax_of['cvode'].plot(1e3 * cv.index, cv.values, color=c, ls=ls, lw=1.2, alpha=0.85, label=lab)
    ax_of['speed'].plot(1e3 * speed_cum.index, speed_cum.values, color=c, ls=ls, lw=1.8, label=lab)

    rows[lab] = {
        'Erreur masse moyenne [%]': err.mean(), 'Delta T moyenne [K]': dT.mean(), 'Erreur HRR [%]': (100 * (a.HR_Rate / r.HR_Rate - 1)).abs().mean(),
        'Appels CVODE [%]': cv.mean() if len(cv) else 0.0,
        'Gain de vitesse (sagetime)': speed_cum.iloc[-1] if len(speed_cum) else np.nan,
        'Ecart O [%]': np.mean(np.abs(ef['O'])), 'Ecart H [%]': np.mean(np.abs(ef['H'])), 'Ecart N [%]': np.mean(np.abs(ef['N'])),
        't_fin [ms]': t.max(), 'n_pas_vitesse': len(sg2)}
    print(lab, 't_fin', t.max(), 'ms | err', round(err.mean(), 3), '| dT', round(dT.mean(), 3), '| CVODE%', round(cv.mean() if len(cv) else 0, 2),
          '| speed x', round(speed_cum.iloc[-1] if len(speed_cum) else np.nan, 3), flush=True)

for k in panels:
    ax = ax_of[k]
    ax.set_title(titles[k], fontsize=13); ax.set_xlabel('t [ms]'); ax.grid(alpha=0.3); ax.set_xlim(0, T_MAX_MS)
    if k not in ('cvode', 'speed'):
        ax.axhline(0, color='#14213d', lw=0.7)
ax_of['err'].legend(fontsize=11, loc='upper left')
fig.suptitle('PK3-H / PK3-noH / STOCH5 rollout vs SAGE dt 5e-6 — statistiques globales, toute la durée disponible', fontsize=15)
fig.savefig('recap_pk3_curves.png', dpi=130, facecolor='white')

tab = pd.DataFrame(rows).T
tab.to_csv('recap_pk3_table.csv')
pd.set_option('display.width', 220)
print(tab.round(4).to_string())
