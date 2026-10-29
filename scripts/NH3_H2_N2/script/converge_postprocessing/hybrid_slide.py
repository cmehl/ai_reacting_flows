"""Slide figure: a priori hybrid threshold sweep of PK3 dt 5e-6 (renorm 3, SAGE states 1/5/20/80/150 ms, all hot cells), all instants:
(left) cost = share of hot cells / of CVODE time sent to CVODE, (right) what remains of each error relative to no hybrid (T RMS error,
|per-step mass bias| of O, OH, NO). Writes hybrid_slide.png."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- config ----
CSV = 'apriori_thresholds.csv'
EPS_NOW = 3e-5
# ----------------

d = pd.read_csv(CSV)
d = d[d.t_ms.astype(str) == 'all'].copy()
d['eps'] = d.eps.astype(float)
ref = d[np.isinf(d.eps)].iloc[0]
d = d[np.isfinite(d.eps)].sort_values('eps')
plt.rcParams.update({'font.size': 15})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.4), constrained_layout=True)
a1.plot(d.eps, d['alpha_cells_%'], 'o-', color='#1f5fa8', lw=2.4, ms=7, label='cellules chaudes envoyées en CVODE')
a1.plot(d.eps, d['cvode_time_share_%'], 's--', color='#2b9348', lw=2.4, ms=7, label='part du temps CVODE')
a1.set_xscale('log'); a1.set_yscale('log'); a1.set_ylabel('[%]'); a1.set_title('Coût de l’hybride')
for lab, col, c, m in (('erreur RMS sur T', 'T_rmse_%incr', '#1f5fa8', 'o-'), ('biais O', 'O_bias_%incr', '#2b9348', 's-'),
                       ('biais OH', 'OH_bias_%incr', '#7b4fa0', '^-'), ('biais NO', 'NO_bias_%incr', '#6c757d', 'd-')):
    a2.plot(d.eps, 100 * d[col].abs() / abs(ref[col]), m, color=c, lw=2.4, ms=7, label=lab)
a2.axhline(100, color='#14213d', lw=0.8, ls=':')
a2.set_xscale('log'); a2.set_ylim(0, 125); a2.set_ylabel('% de la valeur sans hybride'); a2.set_title('Ce qui reste de l’erreur')
for ax in (a1, a2):
    ax.axvline(EPS_NOW, color='#14213d', lw=1.2, ls='--'); ax.text(EPS_NOW * 1.08, (30 if ax is a1 else 116), 'seuil retenu 3e-5', fontsize=13)
    ax.set_xlabel('seuil ε (hybrid_ann_tol)'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=13, loc='lower left' if ax is a1 else 'lower right')
fig.savefig('hybrid_slide.png', dpi=130, facecolor='white')
print(d[['eps', 'alpha_cells_%', 'cvode_time_share_%', 'T_rmse_%incr', 'O_bias_%incr', 'OH_bias_%incr', 'NO_bias_%incr']].round(3).to_string(index=False))
