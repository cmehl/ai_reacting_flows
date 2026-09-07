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
    stats_<axis>0_slice.csv             - per-timestep, per-field, per-model error stats (vs SAGE),
                                           both unweighted (rmse/mae/rel_rmse) and mass-weighted
                                           (rmse_mw/mae_mw/rel_rmse_mw, weight = SAGE cell DENSITY*VOLUME)
    stats_<axis>0_slice_aggregated.csv  - per-field, per-model stats aggregated over all timesteps
    values_<axis>0_slice.csv            - per-timestep, per-field, per-run (SAGE + each model) raw value stats
    flame_position_<axis>0_slice.csv    - per-timestep, per-model mass-weighted centroid position (along
                                           --flame-axis) of cells within --flame-halfwidth of --flame-tref,
                                           a simple flame-front proxy -- reveals a spatial lag/lead of the
                                           accelerated runs' flame relative to SAGE
    mass_conservation_<axis>0_slice.csv - per-model, sampled-cell (every --scatter-stride-th timestep,
                                           --scatter-cells random cells) e_T (Eq. 13 of Mehl & Aubagnac-Karkar,
                                           Phys. Fluids 2023) vs e_Sigma = |sum_k Y_k - 1| (Eq. 10, ibid.),
                                           the model's own mass-fraction-sum conservation violation
    figs/error_vs_time_<field>.png      - RMSE / relative-RMSE vs time, one per field, all models overlaid
    figs/mass_weighted/error_vs_time_<field>.png - same, mass-weighted RMSE/relative-RMSE
    figs/value_vs_time_<field>.png      - raw mean/max value vs time, SAGE + every model on the same axes
    figs/<field>.png                    - final-timestep slice plot: SAGE, each model, each model's error vs SAGE
    figs/flame_position_vs_time.png     - flame-front position vs time (top) and offset from SAGE (bottom)
    figs/mass_conservation_scatter.png  - e_Sigma vs e_T scatter per model, colored by SAGE temperature

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
# Loaded alongside FIELDS on every read but not treated as a compared field:
# DENSITY*VOLUME gives the physical cell mass, used both to mass-weight the
# error stats and as the weight for the flame-position centroid.
EXTRA_FIELDS = ["DENSITY", "VOLUME"]

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
    parser.add_argument("--flame-axis", choices=["X", "Y", "Z"], default="Z",
                         help="Coordinate along which the flame-front centroid is tracked (default: Z, "
                              "the combustor's axial direction)")
    parser.add_argument("--flame-tref", type=float, default=1200.0,
                         help="Reference temperature [K] defining the flame band (default: 1200)")
    parser.add_argument("--flame-halfwidth", type=float, default=150.0,
                         help="Half-width [K] of the |T - flame-tref| band used for the flame-front "
                              "centroid (default: 150)")
    parser.add_argument("--scatter-stride", type=int, default=20,
                         help="Sample the e_Sigma-vs-e_T scatter every Nth timestep (default: 20)")
    parser.add_argument("--scatter-cells", type=int, default=3000,
                         help="Random cells sampled per included timestep for the scatter (default: 3000)")
    parser.add_argument("--scatter-seed", type=int, default=0, help="RNG seed for scatter cell sampling")
    args = parser.parse_args()

    axis = args.slice_axis
    axcfg = SLICE_AXES[axis]
    flame_idx = SLICE_AXES[args.flame_axis]["idx"]
    rng = np.random.default_rng(args.scatter_seed)

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
    flame_records = []
    scatter_records = []
    last_slice = None
    load_fields = FIELDS + EXTRA_FIELDS

    n_steps = len(sage_files)
    for i in range(n_steps):
        coords_s, data_s, t_s = load(sage_files[i], load_fields)
        tree = cKDTree(coords_s)

        mask = np.abs(coords_s[:, axcfg["idx"]]) < args.slice_halfwidth
        h_idx, h_sign, _ = axcfg["h"]
        v_idx, v_sign, _ = axcfg["v"]
        h_coord = h_sign * coords_s[mask, h_idx]
        v_coord = v_sign * coords_s[mask, v_idx]
        flame_coord = coords_s[mask, flame_idx]
        cell_mass = data_s["DENSITY"][mask] * data_s["VOLUME"][mask]
        print(f"  t={t_s:.3e}s ({i + 1}/{n_steps}): {mask.sum()} cells in {axis}=0 slice")

        sv_by_field = {field: data_s[field][mask] for field in FIELDS}
        for field in FIELDS:
            sv = sv_by_field[field]
            value_records.append(
                dict(timestep=i + 1, time=t_s, model="SAGE", field=field_label(field),
                     value_mean=float(sv.mean()), value_max=float(sv.max()), value_min=float(sv.min()))
            )

        def flame_centroid(temperature):
            band = np.abs(temperature - args.flame_tref) < args.flame_halfwidth
            if not band.any():
                return np.nan, 0
            w = cell_mass[band]
            return float(np.sum(w * flame_coord[band]) / np.sum(w)), int(band.sum())

        z_sage, n_sage = flame_centroid(sv_by_field["TEMPERATURE"])
        flame_records.append(dict(timestep=i + 1, time=t_s, model="SAGE", flame_pos=z_sage, n_band_cells=n_sage))

        do_scatter = (i % args.scatter_stride) == 0
        if do_scatter:
            n_pick = min(args.scatter_cells, mask.sum())
            pick = rng.choice(mask.sum(), size=n_pick, replace=False)
            t_sage_pick = sv_by_field["TEMPERATURE"][pick]

        aligned_by_model = {}
        for name, files in model_files.items():
            coords_m, data_m, t_m = load(files[i], load_fields)
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
                # Mass-weighted counterparts (weight = SAGE cell mass = DENSITY*VOLUME):
                # an unweighted per-cell average over-represents the AMR-refined flame
                # zone (many small cells) relative to the coarse downstream/burnt region,
                # which physically holds most of the domain's mass.
                w_sum = float(cell_mass.sum())
                rmse_mw = float(np.sqrt(np.sum(cell_mass * diff ** 2) / w_sum))
                mae_mw = float(np.sum(cell_mass * np.abs(diff)) / w_sum)
                mean_ref_mw = float(np.sum(cell_mass * np.abs(sv)) / w_sum)
                rel_rmse_mw = rmse_mw / mean_ref_mw if mean_ref_mw > 0 else np.nan
                records.append(
                    dict(
                        timestep=i + 1, time=t_s, model=name, field=field_label(field), n_cells=int(mask.sum()),
                        rmse=rmse, mae=mae, max_abs_err=maxerr, mean_sage=mean_ref, rel_rmse=rel_rmse,
                        rmse_mw=rmse_mw, mae_mw=mae_mw, mean_sage_mw=mean_ref_mw, rel_rmse_mw=rel_rmse_mw,
                    )
                )
                value_records.append(
                    dict(timestep=i + 1, time=t_s, model=name, field=field_label(field),
                         value_mean=float(mv.mean()), value_max=float(mv.max()), value_min=float(mv.min()))
                )

            z_model, n_model = flame_centroid(aligned["TEMPERATURE"][mask])
            flame_records.append(
                dict(timestep=i + 1, time=t_s, model=name, flame_pos=z_model, n_band_cells=n_model,
                     delta_pos=(z_model - z_sage) if np.isfinite(z_model) and np.isfinite(z_sage) else np.nan)
            )

            if do_scatter:
                t_model_pick = aligned["TEMPERATURE"][mask][pick]
                y_sum_pick = np.zeros(n_pick)
                for sp in SPECIES:
                    y_sum_pick += aligned[sp][mask][pick]
                e_t = 100.0 * np.abs(t_model_pick - t_sage_pick) / t_sage_pick
                e_sigma = np.abs(y_sum_pick - 1.0)
                for e_t_i, e_sigma_i, t_i in zip(e_t, e_sigma, t_sage_pick):
                    scatter_records.append(
                        dict(model=name, timestep=i + 1, time=t_s,
                             e_T=float(e_t_i), e_Sigma=float(e_sigma_i), T_sage=float(t_i))
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

    flame_name = f"flame_position_{axis}0_slice.csv"
    df_flame = pd.DataFrame.from_records(flame_records)
    df_flame.to_csv(os.path.join(out_dir, flame_name), index=False)

    scatter_name = f"mass_conservation_{axis}0_slice.csv"
    df_scatter = pd.DataFrame.from_records(scatter_records)
    df_scatter.to_csv(os.path.join(out_dir, scatter_name), index=False)

    agg = (
        df.groupby(["model", "field"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_max=("rmse", "max"),
            mae_mean=("mae", "mean"),
            max_abs_err_max=("max_abs_err", "max"),
            rel_rmse_mean=("rel_rmse", "mean"),
            rel_rmse_max=("rel_rmse", "max"),
            rmse_mw_mean=("rmse_mw", "mean"),
            rmse_mw_max=("rmse_mw", "max"),
            mae_mw_mean=("mae_mw", "mean"),
            rel_rmse_mw_mean=("rel_rmse_mw", "mean"),
            rel_rmse_mw_max=("rel_rmse_mw", "max"),
        )
        .reset_index()
        .sort_values(["model", "rel_rmse_mean"], ascending=[True, False])
    )
    agg.to_csv(os.path.join(out_dir, stats_agg_name), index=False)

    for name, _ in models:
        print(f"\n=== Aggregated error ({name} vs SAGE), {axis}=0 slice, {n_steps} timesteps ===")
        print(agg[agg["model"] == name].drop(columns="model").to_string(index=False))

    # --- RMSE / rel-RMSE vs time, one figure per field, all models overlaid ---
    # (also produced mass-weighted, in a separate subdir -- see below)
    def plot_error_vs_time(rmse_col, rel_col, out_subdir, title_suffix):
        os.makedirs(out_subdir, exist_ok=True)
        for field in FIELDS:
            label = field_label(field)
            sub_field = df[df["field"] == label]
            fig, ax1 = plt.subplots(figsize=(7, 4.5))
            ax2 = ax1.twinx()
            for name, _ in models:
                sub = sub_field[sub_field["model"] == name]
                color = model_color(name)
                ax1.plot(sub["time"], sub[rmse_col], "o-", color=color, label=f"{name} RMSE (abs)")
                ax2.plot(sub["time"], sub[rel_col] * 100, "s--", color=color, alpha=0.6,
                          label=f"{name} relative RMSE (%)")
            ax1.set_xlabel("time [s]")
            ax1.set_ylabel("RMSE (absolute)")
            ax2.set_ylabel("relative RMSE [%]")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
            fig.suptitle(f"Error vs SAGE on {axis}=0 slice — {label}{title_suffix}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_subdir, f"error_vs_time_{label}.png"), dpi=150)
            plt.close(fig)

    plot_error_vs_time("rmse", "rel_rmse", fig_dir, "")
    plot_error_vs_time("rmse_mw", "rel_rmse_mw", os.path.join(fig_dir, "mass_weighted"),
                        " (mass-weighted)")

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

    # --- flame-front position vs time: raw centroid (top) + offset from SAGE (bottom) ---
    # flame_coord is the raw (unsigned) coordinate along --flame-axis, independent of the
    # h/v sign conventions used for the slice-plane figures above.
    flame_label = f"{args.flame_axis} [m]"
    fig, (ax_pos, ax_delta) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for name in ["SAGE"] + [n for n, _ in models]:
        sub = df_flame[df_flame["model"] == name]
        color = model_color(name)
        lw = 2.5 if name == "SAGE" else 1.5
        ax_pos.plot(sub["time"], sub["flame_pos"], "-", color=color, lw=lw, label=name)
        if name != "SAGE":
            ax_delta.plot(sub["time"], sub["delta_pos"], "-", color=color, lw=lw, label=name)
    ax_pos.set_ylabel(f"flame centroid, {flame_label}")
    ax_pos.set_title(f"Flame-front position (mass-weighted centroid of |T-{args.flame_tref:g}K|"
                      f"<{args.flame_halfwidth:g}K)")
    ax_pos.legend(fontsize=8, loc="best")
    ax_delta.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax_delta.set_xlabel("time [s]")
    ax_delta.set_ylabel(f"model - SAGE, {flame_label}")
    ax_delta.set_title("Spatial offset of the flame front vs SAGE (0 = no shift)")
    ax_delta.legend(fontsize=8, loc="best")
    fig.suptitle(f"Flame position vs time on {axis}=0 slice")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "flame_position_vs_time.png"), dpi=150)
    plt.close(fig)

    # --- e_Sigma (mass-fraction-sum conservation error) vs e_T (temperature error), per model,
    # colored by SAGE temperature -- same diagnostic as Fig. 11/12 of Mehl & Aubagnac-Karkar,
    # Phys. Fluids 35, 067115 (2023), applied a posteriori to this simulation's slice states.
    model_names_all = [n for n, _ in models]
    n_m = len(model_names_all)
    fig, axes = plt.subplots(1, n_m, figsize=(5.5 * n_m, 5), squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, model_names_all):
        sub = df_scatter[df_scatter["model"] == name]
        sc = ax.scatter(sub["e_T"], sub["e_Sigma"], c=sub["T_sage"], cmap="inferno", s=4, alpha=0.5)
        ax.set_yscale("log")
        ax.set_xlabel(r"$e_T$ [%]")
        ax.set_ylabel(r"$e_\Sigma = |\sum_k Y_k - 1|$")
        ax.set_title(name)
        plt.colorbar(sc, ax=ax, label="SAGE T [K]")
    fig.suptitle(f"Mass-conservation error vs temperature error, {axis}=0 slice "
                 f"(every {args.scatter_stride}th timestep, {args.scatter_cells} cells/timestep)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "mass_conservation_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"\nWrote {len(FIELDS)} final-slice figures + {len(FIELDS)} error-vs-time figures "
          f"(x2, unweighted + mass-weighted) + {len(FIELDS)} value-vs-time figures "
          f"+ flame_position_vs_time.png + mass_conservation_scatter.png to {fig_dir}")
    print(f"Wrote stats to {os.path.join(out_dir, stats_name)}, {os.path.join(out_dir, stats_agg_name)}, "
          f"{os.path.join(out_dir, values_name)}, {os.path.join(out_dir, flame_name)} "
          f"and {os.path.join(out_dir, scatter_name)}")


if __name__ == "__main__":
    main()
