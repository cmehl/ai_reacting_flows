"""Compare a full-CFD SAGE (reference chemistry) run against an ANN-
accelerated CFD run on a single planar slice (X=0, Y=0 or Z=0), across
every available timestep, for Temperature and all species mass fractions.

Inputs are raw CONVERGE post-processing snapshots (STREAM_00/CELL_CENTER_DATA)
under two sibling ``output/`` directories -- one from the SAGE (reference
chemistry) run, one from the ANN-accelerated run, with matching
``post*_+<time>.h5`` filenames on each side.

Cell ordering differs between the two runs (same mesh, different cell
numbering/partitioning) even though cell counts and coordinates match
exactly -- so ANN cells are re-indexed to the SAGE cell order via a
coordinate KDTree (nearest-neighbor distance is checked and should be ~0)
before any comparison.

If the two runs don't reach the same final time (one still running, or
stopped earlier), only the timesteps present on *both* sides are compared
-- i.e. the smallest common time range -- matched by the time encoded in
each filename, not by position.

Outputs, written under ``--out-dir`` (default: ``comparison_<axis>0_slice``
next to the two ``output/`` dirs):
    stats_<axis>0_slice.csv             - per-timestep, per-field error stats
    stats_<axis>0_slice_aggregated.csv  - per-field stats aggregated over all timesteps
    figs/error_vs_time_<field>.png      - RMSE / relative-RMSE vs time, one per field
    figs/<field>.png                    - final-timestep 3-panel (SAGE, ANN, ANN-SAGE) slice plot

Usage:
    python compare_ann_sage_x0.py \\
        --sage-dir .idea/Output_CFD/REDUCED/ANN_FROM_CFD/outputs_original/output \\
        --ann-dir  .idea/Output_CFD/REDUCED/ANN_FROM_CFD/outputs_original_ANN/output \\
        --slice-axis Y

Run with no arguments to use the REDUCED case's default paths (X=0 slice).
"""

import argparse
import glob
import os
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SPECIES = [
    "MASSFRAC_H", "MASSFRAC_H2", "MASSFRAC_H2O", "MASSFRAC_HNO", "MASSFRAC_HO2",
    "MASSFRAC_N", "MASSFRAC_N2", "MASSFRAC_N2H2", "MASSFRAC_N2O", "MASSFRAC_NH",
    "MASSFRAC_NH2", "MASSFRAC_NH3", "MASSFRAC_NNH", "MASSFRAC_NO", "MASSFRAC_NO2",
    "MASSFRAC_O", "MASSFRAC_O2", "MASSFRAC_OH",
]
FIELDS = ["TEMPERATURE"] + SPECIES

# Which coordinate is held ~constant for each slice axis, and which two free
# coordinates go on the plot's horizontal/vertical axes (index into the
# [X, Y, Z] coords array, a +1/-1 sign, and the axis label). X-slice keeps
# the plot's existing (Y, Z) orientation; Y-slice is rotated so the free
# coordinates (X, Z) are plotted as (-Z horizontal, X vertical).
SLICE_AXES = {
    "X": dict(idx=0, h=(1, 1, "Y [m]"), v=(2, 1, "Z [m]")),
    "Y": dict(idx=1, h=(2, -1, "-Z [m]"), v=(0, 1, "X [m]")),
    "Z": dict(idx=2, h=(0, 1, "X [m]"), v=(1, 1, "Y [m]")),
}


def field_label(field):
    return "Temperature" if field == "TEMPERATURE" else field.replace("MASSFRAC_", "")


def load(fp, fields):
    with h5py.File(fp, "r") as f:
        g = f["STREAM_00/CELL_CENTER_DATA"]
        coords = np.stack(
            [g["XCEN_X"][:], g["XCEN_Y"][:], g["XCEN_Z"][:]], axis=1
        ).astype(np.float64)
        data = {k: g[k][:].astype(np.float64) for k in fields}
        t = float(f.attrs["OUTPUT_TIME"][0])
    return coords, data, t


def main():
    default_base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        ".idea", "Output_CFD", "REDUCED", "ANN_FROM_CFD",
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sage-dir", default=os.path.join(default_base, "outputs_original", "output"),
                         help="Directory of SAGE (reference chemistry) post*.h5 snapshots")
    parser.add_argument("--ann-dir", default=os.path.join(default_base, "outputs_original_ANN", "output"),
                         help="Directory of ANN-accelerated post*.h5 snapshots")
    parser.add_argument("--out-dir", default=None,
                         help="Output directory for stats CSVs and figures "
                              "(default: comparison_<axis>0_slice next to the two output/ dirs)")
    parser.add_argument("--slice-axis", choices=["X", "Y", "Z"], default="X",
                         help="Coordinate held ~constant to define the slice plane (default: X)")
    parser.add_argument("--slice-halfwidth", type=float, default=0.001,
                         help="Half-width [m] of the |coord| < value slab used as the <axis>=0 slice")
    args = parser.parse_args()

    axis = args.slice_axis
    axcfg = SLICE_AXES[axis]

    sage_dir = os.path.abspath(args.sage_dir)
    ann_dir = os.path.abspath(args.ann_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(default_base, f"comparison_{axis}0_slice")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    stats_name = f"stats_{axis}0_slice.csv"
    stats_agg_name = f"stats_{axis}0_slice_aggregated.csv"

    # Match snapshots by the time encoded in the filename (post<idx>_+<time>.h5)
    # rather than by position -- the two runs can have different file counts
    # (e.g. one still running, or stopped earlier) even when every timestep
    # they DO share matches exactly. Using the intersection naturally limits
    # the comparison to the smallest common time range between the two runs.
    name_re = re.compile(r"^post\d+_(?P<time>[+-][0-9.eE+-]+)\.h5$")

    def index_by_time(directory):
        by_time = {}
        for fp in glob.glob(os.path.join(directory, "post*.h5")):
            m = name_re.match(os.path.basename(fp))
            if m:
                by_time[m.group("time")] = fp
        return by_time

    sage_by_time = index_by_time(sage_dir)
    ann_by_time = index_by_time(ann_dir)
    common = sorted(set(sage_by_time) & set(ann_by_time), key=float)
    sage_only = sorted(set(sage_by_time) - set(ann_by_time), key=float)
    ann_only = sorted(set(ann_by_time) - set(sage_by_time), key=float)
    assert common, f"No matching timesteps between {sage_dir} and {ann_dir}"
    print(f"Common time range: t={float(common[0]):.3e}s to t={float(common[-1]):.3e}s "
          f"({len(common)} matching timesteps)")
    if sage_only or ann_only:
        print(f"  NOTE: {len(sage_only)} SAGE-only timestep(s), {len(ann_only)} ANN-only "
              f"timestep(s) beyond the common range are skipped: "
              f"SAGE-only={sage_only[:3]}{'...' if len(sage_only) > 3 else ''}, "
              f"ANN-only={ann_only[:3]}{'...' if len(ann_only) > 3 else ''}")

    sage_files = [sage_by_time[t] for t in common]
    ann_files = [ann_by_time[t] for t in common]

    records = []
    last_slice = None

    for i, (sf, af) in enumerate(zip(sage_files, ann_files), start=1):
        coords_s, data_s, t_s = load(sf, FIELDS)
        coords_a, data_a, t_a = load(af, FIELDS)
        assert abs(t_s - t_a) < 1e-9, (t_s, t_a)

        tree = cKDTree(coords_s)
        dist, idx = tree.query(coords_a, k=1)
        if dist.max() > 1e-9:
            print(f"  WARNING t={t_s:.3e}: max nearest-neighbor dist = {dist.max():.3e} (mesh mismatch?)")

        data_a_aligned = {}
        for k, v in data_a.items():
            aligned = np.empty_like(v)
            aligned[idx] = v
            data_a_aligned[k] = aligned

        mask = np.abs(coords_s[:, axcfg["idx"]]) < args.slice_halfwidth
        h_idx, h_sign, _ = axcfg["h"]
        v_idx, v_sign, _ = axcfg["v"]
        h_coord = h_sign * coords_s[mask, h_idx]
        v_coord = v_sign * coords_s[mask, v_idx]
        print(f"  t={t_s:.3e}s ({i}/{len(sage_files)}): {mask.sum()} cells in {axis}=0 slice")

        for field in FIELDS:
            sv = data_s[field][mask]
            av = data_a_aligned[field][mask]
            diff = av - sv
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            maxerr = float(np.max(np.abs(diff)))
            mean_ref = float(np.mean(np.abs(sv)))
            rel_rmse = rmse / mean_ref if mean_ref > 0 else np.nan
            records.append(
                dict(
                    timestep=i, time=t_s, field=field_label(field), n_cells=int(mask.sum()),
                    rmse=rmse, mae=mae, max_abs_err=maxerr, mean_sage=mean_ref, rel_rmse=rel_rmse,
                )
            )

        if i == len(sage_files):
            last_slice = dict(
                h=h_coord, v=v_coord, time=t_s,
                sage={f: data_s[f][mask] for f in FIELDS},
                ann={f: data_a_aligned[f][mask] for f in FIELDS},
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(out_dir, stats_name), index=False)

    agg = (
        df.groupby("field")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_max=("rmse", "max"),
            mae_mean=("mae", "mean"),
            max_abs_err_max=("max_abs_err", "max"),
            rel_rmse_mean=("rel_rmse", "mean"),
            rel_rmse_max=("rel_rmse", "max"),
        )
        .reset_index()
        .sort_values("rel_rmse_mean", ascending=False)
    )
    agg.to_csv(os.path.join(out_dir, stats_agg_name), index=False)

    print(f"\n=== Aggregated error (ANN vs SAGE), {axis}=0 slice, {len(sage_files)} timesteps ===")
    print(agg.to_string(index=False))

    # --- RMSE / rel-RMSE vs time, one figure per field ---
    for field in FIELDS:
        label = field_label(field)
        sub = df[df["field"] == label]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(sub["time"], sub["rmse"], "o-", color="tab:blue", label="RMSE (abs)")
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("RMSE (absolute)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(sub["time"], sub["rel_rmse"] * 100, "s--", color="tab:red", label="relative RMSE (%)")
        ax2.set_ylabel("relative RMSE [%]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        fig.suptitle(f"ANN vs SAGE error on {axis}=0 slice — {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"error_vs_time_{label}.png"), dpi=150)
        plt.close(fig)

    # --- final-timestep 3-panel (SAGE, ANN, ERROR) plots ---
    h_coord, v_coord = last_slice["h"], last_slice["v"]
    h_label, v_label = axcfg["h"][2], axcfg["v"][2]
    for field in FIELDS:
        label = field_label(field)
        sv = last_slice["sage"][field]
        av = last_slice["ann"][field]
        diff = av - sv

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        vmin, vmax = min(sv.min(), av.min()), max(sv.max(), av.max())

        sc0 = axes[0].scatter(h_coord, v_coord, c=sv, cmap="inferno", vmin=vmin, vmax=vmax, s=5)
        axes[0].set_title("SAGE")
        plt.colorbar(sc0, ax=axes[0])

        sc1 = axes[1].scatter(h_coord, v_coord, c=av, cmap="inferno", vmin=vmin, vmax=vmax, s=5)
        axes[1].set_title("ANN")
        plt.colorbar(sc1, ax=axes[1])

        absmax = np.max(np.abs(diff)) or 1e-30
        sc2 = axes[2].scatter(h_coord, v_coord, c=diff, cmap="coolwarm", vmin=-absmax, vmax=absmax, s=5)
        axes[2].set_title("ANN - SAGE (error)")
        plt.colorbar(sc2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel(h_label)
            ax.set_ylabel(v_label)
            ax.set_aspect("equal")

        fig.suptitle(f"{axis}=0 slice at t={last_slice['time']:.3e}s — {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{label}.png"), dpi=150)
        plt.close(fig)

    print(f"\nWrote {len(FIELDS)} final-slice figures + {len(FIELDS)} error-vs-time figures to {fig_dir}")
    print(f"Wrote stats to {os.path.join(out_dir, stats_name)} and {os.path.join(out_dir, stats_agg_name)}")


if __name__ == "__main__":
    main()
