#!/usr/bin/env python3
"""Compare deux fichiers CONVERGE ``species_mass_frac.out`` (ex. ANN vs SAGE).

Tout se configure dans le bloc CONFIG ci-dessous, puis :  python compare_species_mass_frac.py

Genere dans OUTDIR :
  - compare_species_grid.png    : une sous-figure par espece, les 2 courbes
  - compare_species_rmse.png    : RMSE par espece (barres, echelle log)
  - compare_species_metrics.csv : RMSE, RMSE_rel, MAE, erreur max, biais par espece

Le fichier de reference des metriques est choisi par REF ('a' ou 'b', defaut 'b',
typiquement SAGE). L'autre fichier est interpole sur la grille de temps de la
reference, sur la plage de temps commune ; biais = (autre - ref).
"""

import os

import numpy as np
import matplotlib

# ============================ CONFIG ============================
# Fichier A (ex. ANN) et fichier B (ex. SAGE)
FILE_A = (
    "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/"
    "REDUCED_MECH_A_AI_MULTIRESEAU/outputs_ann_log1e14_renorm/stream0/"
    "species_mass_frac.out"
)
FILE_B = (
    "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/"
    "REDUCED_MECH_A_AI_MULTIRESEAU/outputs_original_log1e14_renorm/stream0/"
    "species_mass_frac.out"
)

LABEL_A = "ANN"
LABEL_B = "SAGE"

# Dossier de sortie des figures et du CSV
OUTDIR = "./compare_species_plots"

# Reference pour les metriques : 'a' ou 'b' (l'autre est interpole sur sa grille)
REF = "b"

# Especes a tracer : None (ou []) = toutes les especes communes
SPECIES = None

# Axe y : True = lineaire, False = log
LINEAR = False

# Afficher les figures a l'ecran en plus de les sauver
SHOW = False
# ===============================================================

if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_converge_out(path):
    """Parse un fichier de sortie CONVERGE -> (names, units, data ndarray)."""
    header_lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header_lines.append(s.lstrip("#").strip())
            else:
                break

    names, units = [], []
    for i, h in enumerate(header_lines):
        tok = h.split()
        if tok and tok[0].lower() == "column":
            if i + 1 < len(header_lines):
                names = header_lines[i + 1].split()
            if i + 2 < len(header_lines):
                cand = header_lines[i + 2].split()
                if cand and all(c.startswith("(") and c.endswith(")") for c in cand):
                    units = [c.strip("()") for c in cand]
            break

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if not names or len(names) != data.shape[1]:
        names = ["Time"] + [f"col{j}" for j in range(1, data.shape[1])]
    if len(units) != len(names):
        units = [""] * len(names)
    return names, units, data


def as_dict(names, data):
    """{nom_espece: colonne} + vecteur temps."""
    t = data[:, 0]
    return t, {names[j]: data[:, j] for j in range(1, len(names))}


def metrics(t_ref, y_ref, t_oth, y_oth):
    """Interpole y_oth sur t_ref (plage commune) -> dict de metriques (oth - ref)."""
    lo, hi = max(t_ref[0], t_oth[0]), min(t_ref[-1], t_oth[-1])
    m = (t_ref >= lo) & (t_ref <= hi)
    tr, yr = t_ref[m], y_ref[m]
    yo = np.interp(tr, t_oth, y_oth)
    err = yo - yr
    denom = max(np.abs(yr).max(), 1e-30)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.abs(err).max()),
        "bias": float(np.mean(err)),
        "rmse_rel": float(np.sqrt(np.mean(err ** 2)) / denom),
        "n_points": int(m.sum()),
    }


def main():
    for f in (FILE_A, FILE_B):
        if not os.path.isfile(f):
            raise SystemExit(f"Fichier introuvable : {f}")

    na, _, da = read_converge_out(FILE_A)
    nb, _, db = read_converge_out(FILE_B)
    ta, ya = as_dict(na, da)
    tb, yb = as_dict(nb, db)

    common = [s for s in na[1:] if s in yb]
    only_a = [s for s in na[1:] if s not in yb]
    only_b = [s for s in nb[1:] if s not in ya]
    if SPECIES:
        common = [s for s in SPECIES if s in common]
    if not common:
        raise SystemExit("Aucune espece commune a tracer.")

    print(f"{LABEL_A} : {FILE_A}")
    print(f"     {da.shape[0]} pas, t {ta[0]:.4g} -> {ta[-1]:.4g} s")
    print(f"{LABEL_B} : {FILE_B}")
    print(f"     {db.shape[0]} pas, t {tb[0]:.4g} -> {tb[-1]:.4g} s")
    print(f"{len(common)} especes communes")
    if only_a:
        print(f"  seulement dans {LABEL_A} : {', '.join(only_a)}")
    if only_b:
        print(f"  seulement dans {LABEL_B} : {', '.join(only_b)}")

    os.makedirs(OUTDIR, exist_ok=True)

    if REF == "b":
        lref, lother = LABEL_B, LABEL_A
        tref, yref, toth, yoth = tb, yb, ta, ya
    else:
        lref, lother = LABEL_A, LABEL_B
        tref, yref, toth, yoth = ta, ya, tb, yb
    print(f"reference metriques : {lref}   (erreur = {lother} - {lref})")

    # --- metriques ---
    rows, rmse_by_sp = [], {}
    for s in common:
        mt = metrics(tref, yref[s], toth, yoth[s])
        rmse_by_sp[s] = mt["rmse"]
        rows.append((s, mt["rmse"], mt["rmse_rel"], mt["mae"],
                     mt["max_abs_err"], mt["bias"], mt["n_points"]))
    csv_path = os.path.join(OUTDIR, "compare_species_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("species,rmse,rmse_rel,mae,max_abs_err,bias,n_points\n")
        for r in rows:
            f.write("{},{:.6e},{:.6e},{:.6e},{:.6e},{:+.6e},{}\n".format(*r))
    print(f"  ecrit {csv_path}")
    print(f"\n  {'espece':<8} {'RMSE':>12} {'RMSE_rel':>10} {'max|err|':>12} {'biais':>13}")
    for s, rm, rr, ma, mx, bi, _ in rows:
        print(f"  {s:<8} {rm:12.4e} {rr:10.3%} {mx:12.4e} {bi:+13.4e}")

    # --- grille de courbes ---
    n = len(common)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        s = common[k]
        ax.plot(ta, ya[s], lw=1.6, label=LABEL_A)
        ax.plot(tb, yb[s], lw=1.6, ls="--", label=LABEL_B)
        ax.set_title(f"{s}  (RMSE {rmse_by_sp[s]:.2e})", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y")
        has_pos = np.any(ya[s] > 0) or np.any(yb[s] > 0)
        if not LINEAR and has_pos:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"species_mass_frac : {LABEL_A} vs {LABEL_B}", y=1.0)
    fig.tight_layout()
    grid_png = os.path.join(OUTDIR, "compare_species_grid.png")
    fig.savefig(grid_png, dpi=150)
    print(f"  ecrit {grid_png}")

    # --- barres RMSE ---
    order = sorted(common, key=lambda s: rmse_by_sp[s], reverse=True)
    fig2, ax = plt.subplots(figsize=(max(6, 0.45 * len(order)), 5))
    ax.bar(range(len(order)), [rmse_by_sp[s] for s in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    if any(rmse_by_sp[s] > 0 for s in order):
        ax.set_yscale("log")
    ax.set_ylabel("RMSE (fraction massique)")
    ax.set_title(f"RMSE par espece : {lother} vs {lref} (ref {lref})")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig2.tight_layout()
    rmse_png = os.path.join(OUTDIR, "compare_species_rmse.png")
    fig2.savefig(rmse_png, dpi=150)
    print(f"  ecrit {rmse_png}")

    if SHOW:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
