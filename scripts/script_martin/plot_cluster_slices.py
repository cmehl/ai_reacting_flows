"""For each model and each testing slice, plot which spatial zone of the
slice was routed to which cluster (i.e. which per-cluster sub-network).

Scans ``.idea/<CASE>/testing_results_*_csv.h5`` -- the outputs of
CFDSnapshotTester's ``csv_slices`` mode (see cfd_snapshot_testing.py). Each
such file already stores, per sampled cell: its slice-plane coordinates
(XCEN_X/Y/Z), which testing slice it came from (slice_id), and which cluster
the ANN routed it to (cluster; -1 = below T_threshold, no ANN prediction).
No CVODE/ANN re-run is needed -- this only reads and plots existing results.

For every slice_id in every matching file, writes one PNG (2D scatter,
colored by cluster, in-plane axes chosen from the slice's cut orientation)
to ``<CASE>/figs_clusters/<model>/<slice_name>.png``.

Usage:
    python plot_cluster_slices.py [--root PATH_TO_.idea] [--case CASE_NAME]
"""

import argparse
import glob
import os

import h5py
import numpy as np
import oyaml as yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# axis (as used in test_ann_vs_cvode*.yaml: X/Y/Z) -> the two in-plane
# coordinate datasets (XCEN_X/XCEN_Y/XCEN_Z) to use for the 2D scatter.
IN_PLANE_AXES = {
    "X": (("XCEN_Y", "$y$ [m]"), ("XCEN_Z", "$z$ [m]")),
    "Y": (("XCEN_X", "$x$ [m]"), ("XCEN_Z", "$z$ [m]")),
    "Z": (("XCEN_X", "$x$ [m]"), ("XCEN_Y", "$y$ [m]")),
}


def model_name_from_h5(path):
    base = os.path.basename(path)
    base = base[len("testing_results_"):] if base.startswith("testing_results_") else base
    for suffix in ("_csv.h5", ".h5"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def plot_one_slice(ax, x, y, cluster, xlabel, ylabel, title):
    clusters_present = sorted(int(c) for c in np.unique(cluster))

    if -1 in clusters_present:
        below = cluster == -1
        ax.scatter(x[below], y[below], s=0.6, c="lightgray", linewidths=0,
                   rasterized=True, label=f"< T_threshold ({below.sum()})")

    real_clusters = [c for c in clusters_present if c >= 0]
    cmap = plt.get_cmap("tab10")
    for c in real_clusters:
        mask = cluster == c
        ax.scatter(x[mask], y[mask], s=0.6, c=[cmap(c % 10)], linewidths=0,
                   rasterized=True, label=f"cluster {c} ({mask.sum()})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=9)
    leg = ax.legend(markerscale=15, fontsize=7, loc="best")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)


def process_file(h5_path, out_root):

    model = model_name_from_h5(h5_path)
    print(f">> {h5_path} (model={model})")

    with h5py.File(h5_path, "r") as f:
        if "slices" not in f.attrs:
            print("   no 'slices' attr (not a csv_slices/slices run) -- skipping")
            return

        slices_meta = yaml.safe_load(f.attrs["slices"])
        test_data_file = str(f.attrs.get("test_data_file", ""))
        slice_files = test_data_file.split(";") if test_data_file else []

        slice_id = f["slice_id"][()]
        cluster = f["cluster"][()]
        coords = {k: f[k][()] for k in ("XCEN_X", "XCEN_Y", "XCEN_Z")}

    out_dir = os.path.join(out_root, model)
    os.makedirs(out_dir, exist_ok=True)

    n_slices = int(slice_id.max()) + 1 if slice_id.size else 0
    for i in range(n_slices):
        mask = slice_id == i
        if not mask.any():
            continue

        meta = slices_meta[i] if i < len(slices_meta) else {}
        axis = str(meta.get("axis", "Z")).upper()
        center = meta.get("center", float("nan"))

        if i < len(slice_files):
            slice_name = os.path.splitext(os.path.basename(slice_files[i]))[0]
        else:
            slice_name = f"slice_{axis}_{center:g}"

        (xk, xlabel), (yk, ylabel) = IN_PLANE_AXES.get(axis, IN_PLANE_AXES["Z"])
        x, y = coords[xk][mask], coords[yk][mask]
        c = cluster[mask]

        fig, ax = plt.subplots(figsize=(6, 5))
        plot_one_slice(
            ax, x, y, c, xlabel, ylabel,
            title=f"{model}\n{slice_name}  (axis={axis}, center={center:g})",
        )
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{slice_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"   [{i}] {slice_name}: {mask.sum()} cells -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=None,
        help="Path to the .idea directory (default: ../../.idea relative to this script)",
    )
    parser.add_argument(
        "--case", default=None,
        help="Only process this case subdirectory of .idea (default: all cases)",
    )
    args = parser.parse_args()

    root = args.root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".idea"
    )
    root = os.path.abspath(root)

    pattern = os.path.join(root, args.case or "*", "testing_results_*_csv.h5")
    h5_files = sorted(glob.glob(pattern))
    print(f">> Found {len(h5_files)} result file(s) matching {pattern}")

    for h5_path in h5_files:
        case_dir = os.path.dirname(h5_path)
        out_root = os.path.join(case_dir, "figs_clusters")
        process_file(h5_path, out_root)

    print(">> Done")


if __name__ == "__main__":
    main()
