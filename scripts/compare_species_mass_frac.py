#!/usr/bin/env python3
"""Compare deux fichiers CONVERGE ``species_mass_frac.out`` (ex. ANN vs SAGE).

Usage
-----
    python compare_species_mass_frac.py FICHIER_ANN.out FICHIER_SAGE.out \
        [-o DOSSIER] [--labels ANN SAGE] [--ref b] [--species N2 H2 NO ...] [--linear]

Genere dans le dossier de sortie :
  - compare_species_grid.png   : une sous-figure par espece, les 2 courbes
  - compare_species_rmse.png   : RMSE par espece (barres, echelle log)
  - compare_species_metrics.csv: RMSE, MAE, erreur max, biais par espece

Le fichier de reference des metriques est choisi par --ref (defaut : b, le 2e
argument, typiquement SAGE). L'autre fichier est interpole sur la grille de temps
de la reference, sur la plage de temps commune ; biais = (autre - ref).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def metrics(t_ref, y_ref, t_b, y_b):
    """Interpole y_b sur t_ref (plage commune) et renvoie un dict de metriques."""
    lo, hi = max(t_ref[0], t_b[0]), min(t_ref[-1], t_b[-1])
    m = (t_ref >= lo) & (t_ref <= hi)
    tr, yr = t_ref[m], y_ref[m]
    yb = np.interp(tr, t_b, y_b)
    err = yb - yr
    denom = np.maximum(np.abs(yr).max(), 1e-30)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.abs(err).max()),
        "bias": float(np.mean(err)),
        "rmse_rel": float(np.sqrt(np.mean(err ** 2)) / denom),
        "n_points": int(m.sum()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file_a", help="1er fichier (reference des metriques), ex. ANN")
    p.add_argument("file_b", help="2e fichier, ex. SAGE")
    p.add_argument("-o", "--outdir", default=".", help="dossier de sortie")
    p.add_argument("--labels", nargs=2, default=["A", "B"],
                   metavar=("LABEL_A", "LABEL_B"))
    p.add_argument("--ref", choices=["a", "b"], default="b",
                   help="fichier de reference pour les metriques (defaut : b, "
                        "typiquement SAGE ; erreur = autre - ref)")
    p.add_argument("--species", nargs="+", default=None,
                   help="sous-ensemble d'especes a tracer")
    p.add_argument("--linear", action="store_true", help="axe y lineaire (defaut : log)")
    p.add_argument("--show", action="store_true", help="afficher a l'ecran en plus")
    args = p.parse_args(argv)

    for f in (args.file_a, args.file_b):
        if not os.path.isfile(f):
            sys.exit(f"Fichier introuvable : {f}")

    la, lb = args.labels
    na, ua, da = read_converge_out(args.file_a)
    nb, ub, db = read_converge_out(args.file_b)
    ta, ya = as_dict(na, da)
    tb, yb = as_dict(nb, db)

    common = [s for s in na[1:] if s in yb]
    only_a = [s for s in na[1:] if s not in yb]
    only_b = [s for s in nb[1:] if s not in ya]
    if args.species:
        common = [s for s in args.species if s in common]
    if not common:
        sys.exit("Aucune espece commune a tracer.")

    print(f"{la} : {args.file_a}")
    print(f"     {da.shape[0]} pas, t {ta[0]:.4g} -> {ta[-1]:.4g} s")
    print(f"{lb} : {args.file_b}")
    print(f"     {db.shape[0]} pas, t {tb[0]:.4g} -> {tb[-1]:.4g} s")
    print(f"{len(common)} especes communes")
    if only_a:
        print(f"  seulement dans {la} : {', '.join(only_a)}")
    if only_b:
        print(f"  seulement dans {lb} : {', '.join(only_b)}")

    os.makedirs(args.outdir, exist_ok=True)

    if args.ref == "b":
        lref, lother = lb, la
        tref, yref, toth, yoth = tb, yb, ta, ya
    else:
        lref, lother = la, lb
        tref, yref, toth, yoth = ta, ya, tb, yb
    print(f"reference metriques : {lref}   (erreur = {lother} - {lref})")

    # --- metriques ---
    rows, rmse_by_sp = [], {}
    for s in common:
        mt = metrics(tref, yref[s], toth, yoth[s])
        rmse_by_sp[s] = mt["rmse"]
        rows.append((s, mt["rmse"], mt["rmse_rel"], mt["mae"],
                     mt["max_abs_err"], mt["bias"], mt["n_points"]))
    csv_path = os.path.join(args.outdir, "compare_species_metrics.csv")
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
        ax.plot(ta, ya[s], lw=1.6, label=la)
        ax.plot(tb, yb[s], lw=1.6, ls="--", label=lb)
        ax.set_title(f"{s}  (RMSE {rmse_by_sp[s]:.2e})", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y")
        has_pos = np.any(ya[s] > 0) or np.any(yb[s] > 0)
        if not args.linear and has_pos:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"species_mass_frac : {la} vs {lb}", y=1.0)
    fig.tight_layout()
    grid_png = os.path.join(args.outdir, "compare_species_grid.png")
    fig.savefig(grid_png, dpi=150)
    plt.close(fig)
    print(f"  ecrit {grid_png}")

    # --- barres RMSE ---
    order = sorted(common, key=lambda s: rmse_by_sp[s], reverse=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(order)), 5))
    ax.bar(range(len(order)), [rmse_by_sp[s] for s in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    if any(rmse_by_sp[s] > 0 for s in order):
        ax.set_yscale("log")
    ax.set_ylabel("RMSE (fraction massique)")
    ax.set_title(f"RMSE par espece : {lother} vs {lref} (ref {lref})")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    rmse_png = os.path.join(args.outdir, "compare_species_rmse.png")
    fig.savefig(rmse_png, dpi=150)
    plt.close(fig)
    print(f"  ecrit {rmse_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
