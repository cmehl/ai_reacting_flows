#!/usr/bin/env python3
"""Lit un fichier CONVERGE ``species_mass_frac.out`` (ou tout fichier ``*.out``
du meme format : entete en lignes ``#``, colonne 1 = Time) et en fait des plots.

Usage
-----
    python plot_species_mass_frac.py [FICHIER.out] [-o DOSSIER_SORTIE]
                                     [--species N2 H2 NO ...] [--linear] [--show]

Sans argument, utilise le chemin par defaut ci-dessous (a adapter).
Genere dans le dossier de sortie :
  - species_mass_frac_all.png      : toutes les especes sur un graphe (y log)
  - species_mass_frac_grid.png     : une sous-figure par espece
Et ecrit species_mass_frac.csv (donnees parsees, pratique pour Excel/pandas).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FILE = (
    "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/"
    "REDUCED_MECH_A_AI_MULTIRESEAU/outputs_original_log1e14_renorm/stream0/"
    "species_mass_frac.out"
)


def read_converge_out(path):
    """Parse un fichier de sortie CONVERGE.

    Retourne (names, units, data) ou :
      - names : liste des noms de colonnes (col 1 incluse, typiquement 'Time')
      - units : liste des unites (meme longueur), '' si absente
      - data  : ndarray (n_rows, n_cols)
    """
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

    # Lignes d'entete utiles : celle des numeros de colonne ("column 1 2 3 ..."),
    # suivie des noms, puis des unites entre parentheses.
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
        # Fallback : noms generiques
        names = ["Time"] + [f"col{j}" for j in range(1, data.shape[1])]
    if len(units) != len(names):
        units = [""] * len(names)

    return names, units, data


def plot_all(t, names, data, out_png, linear=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(1, data.shape[1]):
        ax.plot(t, data[:, j], label=names[j], lw=1.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass fraction")
    if not linear:
        ax.set_yscale("log")
    ax.set_title("CONVERGE species_mass_frac")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  ecrit {out_png}")


def plot_grid(t, names, data, out_png, linear=False):
    n = data.shape[1] - 1
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        j = k + 1
        ax.plot(t, data[:, j], lw=1.4)
        ax.set_title(names[j])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y")
        if not linear:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("CONVERGE species_mass_frac (par espece)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  ecrit {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", default=DEFAULT_FILE,
                   help="fichier species_mass_frac.out")
    p.add_argument("-o", "--outdir", default=None,
                   help="dossier de sortie (defaut : a cote du fichier)")
    p.add_argument("--species", nargs="+", default=None,
                   help="sous-ensemble d'especes a tracer")
    p.add_argument("--linear", action="store_true", help="axe y lineaire (defaut : log)")
    p.add_argument("--show", action="store_true", help="afficher a l'ecran en plus")
    args = p.parse_args(argv)

    path = args.file
    if not os.path.isfile(path):
        sys.exit(f"Fichier introuvable : {path}")

    names, units, data = read_converge_out(path)
    t = data[:, 0]
    print(f"Lu {path}")
    print(f"  {data.shape[0]} pas de temps, {data.shape[1] - 1} especes")
    print(f"  especes : {', '.join(names[1:])}")
    print(f"  temps : {t[0]:.6g} -> {t[-1]:.6g} s")

    if args.species:
        keep = [0] + [names.index(s) for s in args.species if s in names]
        missing = [s for s in args.species if s not in names]
        if missing:
            print(f"  (ignorees, absentes : {', '.join(missing)})")
        names = [names[i] for i in keep]
        data = data[:, keep]

    outdir = args.outdir or os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)

    # CSV pour reutilisation
    csv_path = os.path.join(outdir, "species_mass_frac.csv")
    np.savetxt(csv_path, data, delimiter=",", header=",".join(names), comments="")
    print(f"  ecrit {csv_path}")

    plot_all(t, names, data, os.path.join(outdir, "species_mass_frac_all.png"),
             linear=args.linear)
    plot_grid(t, names, data, os.path.join(outdir, "species_mass_frac_grid.png"),
              linear=args.linear)

    if args.show:
        fig, ax = plt.subplots(figsize=(10, 6))
        for j in range(1, data.shape[1]):
            ax.plot(t, data[:, j], label=names[j], lw=1.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass fraction")
        if not args.linear:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.show()


if __name__ == "__main__":
    main()
