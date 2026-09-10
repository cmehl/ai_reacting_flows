#!/usr/bin/env python3
"""Lit un fichier CONVERGE ``species_mass_frac.out`` (entete en lignes ``#``,
colonne 1 = Time) et en fait des plots.

Tout se configure dans le bloc CONFIG ci-dessous, puis :  python plot_species_mass_frac.py

Genere dans OUTDIR :
  - species_mass_frac_all.png   : toutes les especes sur un graphe
  - species_mass_frac_grid.png  : une sous-figure par espece
  - species_mass_frac.csv       : donnees parsees (pour pandas/Excel)
"""

import os

import numpy as np
import matplotlib

# ============================ CONFIG ============================
# Fichier CONVERGE a lire
FILE = (
    "/ifpengpfs/scratch/ifpen/kotlarcm/CONVERGE/CAS_AI/CAS_LEWIS_UNITAIRE/"
    "REDUCED_MECH_A_AI_MULTIRESEAU/outputs_original_log1e14_renorm/stream0/"
    "species_mass_frac.out"
)

# Dossier de sortie ; None = a cote du fichier d'entree
OUTDIR = None

# Especes a tracer : None (ou []) = toutes
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


def plot_all(t, names, data, out_png):
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(1, data.shape[1]):
        ax.plot(t, data[:, j], label=names[j], lw=1.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass fraction")
    if not LINEAR and np.any(data[:, 1:] > 0):
        ax.set_yscale("log")
    ax.set_title("CONVERGE species_mass_frac")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  ecrit {out_png}")


def plot_grid(t, names, data, out_png):
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
        if not LINEAR and np.any(data[:, j] > 0):
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("CONVERGE species_mass_frac (par espece)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"  ecrit {out_png}")


def main():
    if not os.path.isfile(FILE):
        raise SystemExit(f"Fichier introuvable : {FILE}")

    names, _, data = read_converge_out(FILE)
    t = data[:, 0]
    print(f"Lu {FILE}")
    print(f"  {data.shape[0]} pas de temps, {data.shape[1] - 1} especes")
    print(f"  especes : {', '.join(names[1:])}")
    print(f"  temps : {t[0]:.6g} -> {t[-1]:.6g} s")

    if SPECIES:
        keep = [0] + [names.index(s) for s in SPECIES if s in names]
        missing = [s for s in SPECIES if s not in names]
        if missing:
            print(f"  (ignorees, absentes : {', '.join(missing)})")
        names = [names[i] for i in keep]
        data = data[:, keep]

    outdir = OUTDIR or os.path.dirname(os.path.abspath(FILE))
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "species_mass_frac.csv")
    np.savetxt(csv_path, data, delimiter=",", header=",".join(names), comments="")
    print(f"  ecrit {csv_path}")

    plot_all(t, names, data, os.path.join(outdir, "species_mass_frac_all.png"))
    plot_grid(t, names, data, os.path.join(outdir, "species_mass_frac_grid.png"))

    if SHOW:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
