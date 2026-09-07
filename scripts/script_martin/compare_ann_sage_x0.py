"""Compare a full-CFD SAGE (reference chemistry) run against one or two
accelerated CFD runs -- an ANN run, and optionally a hybrid run (ANN
prediction, falling back to SAGE where the ANN error exceeds a threshold)
-- on a single planar slice (X=0, Y=0 or Z=0), across every available
timestep, for Temperature and all species mass fractions.

Inputs are raw CONVERGE post-processing snapshots (STREAM_00/CELL_CENTER_DATA)
under sibling ``output/`` directories -- one per run -- with matching
``post*_+<time>.h5`` filenames on each side.

Cell ordering can differ between runs (same mesh, different cell
numbering/partitioning) even though cell counts and coordinates match
exactly -- so each accelerated run's cells are re-indexed to the SAGE cell
order via a coordinate KDTree (nearest-neighbor distance is checked and
should be ~0) before any comparison.

If the runs don't all reach the same final time (one still running, or
stopped earlier), only the timesteps present on *every* side are compared
-- i.e. the smallest common time range -- matched by the time encoded in
each filename, not by position.

Outputs, written under ``--out-dir`` (default: ``comparison_<axis>0_slice``
next to the SAGE/ANN ``output/`` dirs):
    stats_<axis>0_slice.csv             - per-timestep, per-field, per-model error stats (vs SAGE)
    stats_<axis>0_slice_aggregated.csv  - per-field, per-model stats aggregated over all timesteps
    values_<axis>0_slice.csv            - per-timestep, per-field, per-run (SAGE + each model) raw value stats
    figs/error_vs_time_<field>.png      - RMSE / relative-RMSE vs time, one per field, all models overlaid
    figs/value_vs_time_<field>.png      - raw mean/max value vs time, SAGE + every model on the same axes
    figs/<field>.png                    - final-timestep slice plot: SAGE, each model, each model's error vs SAGE

Usage:
    python compare_ann_sage_x0.py \\
        --sage-dir   .../outputs_original_COVDE/output \\
        --ann-dir    .../outputs_original_ANN/output \\
        --hybrid-dir .../outputs_original_ANN_Hybrid/output \\
        --slice-axis Y

Run with no arguments to use the REDUCED case's default paths (X=0 slice, ANN only).
"""

import argparse
import glob
import itertools
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

# Consistent color per model across figures; extra models beyond this list
# cycle through matplotlib's default palette.
MODEL_COLORS = {"SAGE": "black", "ANN": "tab:blue", "Hybrid": "tab:green", "ANN-Renorm": "tab:green"}
FALLBACK_COLORS = itertools.cycle(
    ["tab:purple", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]
)


def model_color(name):
    if name not in MODEL_COLORS:
        MODEL_COLORS[name] = next(FALLBACK_COLORS)
    return MODEL_COLORS[name]


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


def index_by_time(directory, name_re):
    by_time = {}
    for fp in glob.glob(os.path.join(directory, "post*.h5")):
        m = name_re.match(os.path.basename(fp))
        if m:
            by_time[m.group("time")] = fp
    return by_time


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
    parser.add_argument("--hybrid-dir", default=None,
                         help="Optional directory of hybrid (ANN, falling back to SAGE above an error "
                              "threshold) post*.h5 snapshots -- compared alongside ANN if given")
    parser.add_argument("--ann-label", default="ANN",
                         help="Legend/column label for the --ann-dir run (default: ANN)")
    parser.add_argument("--hybrid-label", default="Hybrid",
                         help="Legend/column label for the --hybrid-dir run (default: Hybrid)")
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=DIR",
                         help="Additional accelerated run to compare, beyond --ann-dir/--hybrid-dir -- "
                              "repeatable, e.g. --model ANN_512=.../output --model ANN_512_Renorm=.../output")
    parser.add_argument("--out-dir", default=None,
                         help="Output directory for stats CSVs and figures "
                              "(default: comparison_<axis>0_slice next to the SAGE/ANN output/ dirs)")
    parser.add_argument("--slice-axis", choices=["X", "Y", "Z"], default="X",
                         help="Coordinate held ~constant to define the slice plane (default: X)")
    parser.add_argument("--slice-halfwidth", type=float, default=0.001,
                         help="Half-width [m] of the |coord| < value slab used as the <axis>=0 slice")
    args = parser.parse_args()

    axis = args.slice_axis
    axcfg = SLICE_AXES[axis]

    sage_dir = os.path.abspath(args.sage_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(default_base, f"comparison_{axis}0_slice")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    stats_name = f"stats_{axis}0_slice.csv"
    stats_agg_name = f"stats_{axis}0_slice_aggregated.csv"

    models = [(args.ann_label, os.path.abspath(args.ann_dir))]
    if args.hybrid_dir:
        models.append((args.hybrid_label, os.path.abspath(args.hybrid_dir)))
    for spec in args.model:
        label, _, mdir = spec.partition("=")
        assert mdir, f"--model expects LABEL=DIR, got {spec!r}"
        models.append((label, os.path.abspath(mdir)))

    # Match snapshots by the time encoded in the filename (post<idx>_+<time>.h5)
    # rather than by position -- the runs can have different file counts (e.g.
    # one still running, or stopped earlier) even when every timestep they DO
    # share matches exactly. Using the intersection across all runs naturally
    # limits the comparison to the smallest common time range.
    name_re = re.compile(r"^post\d+_(?P<time>[+-][0-9.eE+-]+)\.h5$")

    sage_by_time = index_by_time(sage_dir, name_re)
    model_by_time = {name: index_by_time(d, name_re) for name, d in models}

    common = set(sage_by_time)
    for name, by_time in model_by_time.items():
        common &= set(by_time)
    common = sorted(common, key=float)
    assert common, f"No timestep is common to all runs (SAGE + {[n for n, _ in models]})"
    print(f"Common time range: t={float(common[0]):.3e}s to t={float(common[-1]):.3e}s "
          f"({len(common)} matching timesteps, models: SAGE + {[n for n, _ in models]})")

    common_set = set(common)
    all_by_time = {"SAGE": sage_by_time, **model_by_time}
    for name, by_time in all_by_time.items():
        only = sorted(set(by_time) - common_set, key=float)
        if only:
            print(f"  NOTE: {len(only)} {name}-only timestep(s) beyond the common range are skipped: "
                  f"{only[:3]}{'...' if len(only) > 3 else ''}")

    sage_files = [sage_by_time[t] for t in common]
    model_files = {name: [by_time[t] for t in common] for name, by_time in model_by_time.items()}

    records = []
    value_records = []
    last_slice = None

    n_steps = len(sage_files)
    for i in range(n_steps):
        coords_s, data_s, t_s = load(sage_files[i], FIELDS)
        tree = cKDTree(coords_s)

        mask = np.abs(coords_s[:, axcfg["idx"]]) < args.slice_halfwidth
        h_idx, h_sign, _ = axcfg["h"]
        v_idx, v_sign, _ = axcfg["v"]
        h_coord = h_sign * coords_s[mask, h_idx]
        v_coord = v_sign * coords_s[mask, v_idx]
        print(f"  t={t_s:.3e}s ({i + 1}/{n_steps}): {mask.sum()} cells in {axis}=0 slice")

        sv_by_field = {field: data_s[field][mask] for field in FIELDS}
        for field in FIELDS:
            sv = sv_by_field[field]
            value_records.append(
                dict(timestep=i + 1, time=t_s, model="SAGE", field=field_label(field),
                     value_mean=float(sv.mean()), value_max=float(sv.max()), value_min=float(sv.min()))
            )

        aligned_by_model = {}
        for name, files in model_files.items():
            coords_m, data_m, t_m = load(files[i], FIELDS)
            assert abs(t_s - t_m) < 1e-9, (name, t_s, t_m)

            dist, idx = tree.query(coords_m, k=1)
            if dist.max() > 1e-9:
                print(f"  WARNING t={t_s:.3e} [{name}]: max nearest-neighbor dist = {dist.max():.3e} "
                      f"(mesh mismatch?)")

            aligned = {}
            for k, v in data_m.items():
                a = np.empty_like(v)
                a[idx] = v
                aligned[k] = a
            aligned_by_model[name] = aligned

            for field in FIELDS:
                sv = sv_by_field[field]
                mv = aligned[field][mask]
                diff = mv - sv
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
                maxerr = float(np.max(np.abs(diff)))
                mean_ref = float(np.mean(np.abs(sv)))
                rel_rmse = rmse / mean_ref if mean_ref > 0 else np.nan
                records.append(
                    dict(
                        timestep=i + 1, time=t_s, model=name, field=field_label(field), n_cells=int(mask.sum()),
                        rmse=rmse, mae=mae, max_abs_err=maxerr, mean_sage=mean_ref, rel_rmse=rel_rmse,
                    )
                )
                value_records.append(
                    dict(timestep=i + 1, time=t_s, model=name, field=field_label(field),
                         value_mean=float(mv.mean()), value_max=float(mv.max()), value_min=float(mv.min()))
                )

        if i == n_steps - 1:
            last_slice = dict(
                h=h_coord, v=v_coord, time=t_s,
                sage={f: data_s[f][mask] for f in FIELDS},
                models={name: {f: aligned_by_model[name][f][mask] for f in FIELDS} for name in aligned_by_model},
            )

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(out_dir, stats_name), index=False)

    values_name = f"values_{axis}0_slice.csv"
    df_values = pd.DataFrame.from_records(value_records)
    df_values.to_csv(os.path.join(out_dir, values_name), index=False)

    agg = (
        df.groupby(["model", "field"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_max=("rmse", "max"),
            mae_mean=("mae", "mean"),
            max_abs_err_max=("max_abs_err", "max"),
            rel_rmse_mean=("rel_rmse", "mean"),
            rel_rmse_max=("rel_rmse", "max"),
        )
        .reset_index()
        .sort_values(["model", "rel_rmse_mean"], ascending=[True, False])
    )
    agg.to_csv(os.path.join(out_dir, stats_agg_name), index=False)

    for name, _ in models:
        print(f"\n=== Aggregated error ({name} vs SAGE), {axis}=0 slice, {n_steps} timesteps ===")
        print(agg[agg["model"] == name].drop(columns="model").to_string(index=False))

    # --- RMSE / rel-RMSE vs time, one figure per field, all models overlaid ---
    for field in FIELDS:
        label = field_label(field)
        sub_field = df[df["field"] == label]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()
        for name, _ in models:
            sub = sub_field[sub_field["model"] == name]
            color = model_color(name)
            ax1.plot(sub["time"], sub["rmse"], "o-", color=color, label=f"{name} RMSE (abs)")
            ax2.plot(sub["time"], sub["rel_rmse"] * 100, "s--", color=color, alpha=0.6,
                      label=f"{name} relative RMSE (%)")
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("RMSE (absolute)")
        ax2.set_ylabel("relative RMSE [%]")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
        fig.suptitle(f"Error vs SAGE on {axis}=0 slice — {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"error_vs_time_{label}.png"), dpi=150)
        plt.close(fig)

    # --- raw value (mean/max over the slice) vs time, SAGE + every model on the same axes ---
    for field in FIELDS:
        label = field_label(field)
        sub_field = df_values[df_values["field"] == label]
        fig, (ax_mean, ax_max) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        for name in ["SAGE"] + [n for n, _ in models]:
            sub = sub_field[sub_field["model"] == name]
            color = model_color(name)
            lw = 2.5 if name == "SAGE" else 1.5
            ax_mean.plot(sub["time"], sub["value_mean"], "-", color=color, lw=lw, label=name)
            ax_max.plot(sub["time"], sub["value_max"], "-", color=color, lw=lw, label=name)
        ax_mean.set_ylabel(f"mean {label} over slice")
        ax_mean.legend(fontsize=8, loc="best")
        ax_mean.set_title("Mean over slice")
        ax_max.set_xlabel("time [s]")
        ax_max.set_ylabel(f"max {label} over slice")
        ax_max.set_title("Max over slice")
        fig.suptitle(f"{label}: SAGE vs {', '.join(n for n, _ in models)} on {axis}=0 slice")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"value_vs_time_{label}.png"), dpi=150)
        plt.close(fig)

    # --- final-timestep slice plots: SAGE, each model, each model's error ---
    h_coord, v_coord = last_slice["h"], last_slice["v"]
    h_label, v_label = axcfg["h"][2], axcfg["v"][2]
    model_names = [name for name, _ in models]
    n_models = len(model_names)
    for field in FIELDS:
        label = field_label(field)
        sv = last_slice["sage"][field]
        mvals = {name: last_slice["models"][name][field] for name in model_names}
        diffs = {name: mvals[name] - sv for name in model_names}

        n_panels = 1 + 2 * n_models
        fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5))

        vmin = min([sv.min()] + [v.min() for v in mvals.values()])
        vmax = max([sv.max()] + [v.max() for v in mvals.values()])
        absmax = max(np.max(np.abs(d)) for d in diffs.values()) or 1e-30

        sc0 = axes[0].scatter(h_coord, v_coord, c=sv, cmap="inferno", vmin=vmin, vmax=vmax, s=5)
        axes[0].set_title("SAGE")
        plt.colorbar(sc0, ax=axes[0])

        for j, name in enumerate(model_names):
            ax = axes[1 + j]
            sc = ax.scatter(h_coord, v_coord, c=mvals[name], cmap="inferno", vmin=vmin, vmax=vmax, s=5)
            ax.set_title(name)
            plt.colorbar(sc, ax=ax)

        for j, name in enumerate(model_names):
            ax = axes[1 + n_models + j]
            sc = ax.scatter(h_coord, v_coord, c=diffs[name], cmap="coolwarm", vmin=-absmax, vmax=absmax, s=5)
            ax.set_title(f"{name} - SAGE (error)")
            plt.colorbar(sc, ax=ax)

        for ax in axes:
            ax.set_xlabel(h_label)
            ax.set_ylabel(v_label)
            ax.set_aspect("equal")

        fig.suptitle(f"{axis}=0 slice at t={last_slice['time']:.3e}s — {label}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{label}.png"), dpi=150)
        plt.close(fig)

    print(f"\nWrote {len(FIELDS)} final-slice figures + {len(FIELDS)} error-vs-time figures "
          f"+ {len(FIELDS)} value-vs-time figures to {fig_dir}")
    print(f"Wrote stats to {os.path.join(out_dir, stats_name)}, {os.path.join(out_dir, stats_agg_name)} "
          f"and {os.path.join(out_dir, values_name)}")


if __name__ == "__main__":
    main()
