"""Animate the temporal evolution of a full-CFD SAGE (reference chemistry) run
against an ANN run and an ANN-Hybrid run (ANN prediction, falling back to SAGE
where the ANN error exceeds a threshold), on a single planar slice
(X=0, Y=0 or Z=0), for Temperature and every species mass fraction.

For each field one animation (GIF) is written with a 2x3 panel layout, one
frame per shared timestep:

    +-----------+-----------+-----------------+
    |   SAGE    |    ANN    |   ANN Hybrid    |   <- raw field, shared color scale
    +-----------+-----------+-----------------+
    |  (blank / |  ANN      |  ANN Hybrid     |
    |   caption)|  - SAGE   |  - SAGE         |   <- error vs SAGE, shared +/- scale
    +-----------+-----------+-----------------+

The value color scale (top row) and the symmetric error color scale (bottom
row) are fixed across the whole animation -- computed once from robust
percentiles over every frame and every model -- so brightness/contrast is
comparable frame to frame.

Inputs are raw CONVERGE post-processing snapshots (STREAM_00/CELL_CENTER_DATA)
under sibling ``output/`` directories -- one per run -- with matching
``post*_+<time>.h5`` filenames on each side. Cell ordering can differ between
runs (same mesh, different numbering/partitioning); each accelerated run's
cells are re-indexed to the SAGE cell order via a coordinate KDTree before
comparison. Only timesteps present on every side are animated, matched by the
time encoded in the filename (not by position).

Usage:
    python animate_ann_sage_hybrid.py \\
        --sage-dir   .../outputs_original_COVDE/output \\
        --ann-dir    .../outputs_original_ANN/output \\
        --hybrid-dir .../outputs_original_ANN_Hybrid/output \\
        --slice-axis Y \\
        --out-dir    .../comparison_Y0_slice_hybrid/anim \\
        --stride 1 --fps 12

Only pillow (GIF) is required as an animation writer -- no ffmpeg needed.
"""

import argparse
import glob
import os
import re
import time as _time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
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
# [X, Y, Z] coords array, a +1/-1 sign, and the axis label). Mirrors
# compare_ann_sage_x0.py so the two tools show the same orientation.
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


def index_by_time(directory, name_re):
    by_time = {}
    for fp in glob.glob(os.path.join(directory, "post*.h5")):
        m = name_re.match(os.path.basename(fp))
        if m:
            by_time[m.group("time")] = fp
    return by_time


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sage-dir", required=True,
                        help="Directory of SAGE (reference chemistry) post*.h5 snapshots")
    parser.add_argument("--ann-dir", required=True,
                        help="Directory of ANN-accelerated post*.h5 snapshots")
    parser.add_argument("--hybrid-dir", required=True,
                        help="Directory of ANN-Hybrid post*.h5 snapshots")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for the per-field GIF animations")
    parser.add_argument("--slice-axis", choices=["X", "Y", "Z"], default="Y",
                        help="Coordinate held ~constant to define the slice plane (default: Y)")
    parser.add_argument("--slice-halfwidth", type=float, default=0.001,
                        help="Half-width [m] of the |coord| < value slab used as the <axis>=0 slice")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth shared timestep (default: 1 = all)")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second")
    parser.add_argument("--dpi", type=int, default=90, help="Animation raster resolution")
    parser.add_argument("--marker-size", type=float, default=6.0, help="Scatter marker size")
    parser.add_argument("--clip-percentile", type=float, default=99.5,
                        help="Percentile (p and 100-p) of the SAGE field for the fixed value color scale")
    parser.add_argument("--err-percentile", type=float, default=99.0,
                        help="Percentile of |error| for the fixed symmetric error color scale")
    parser.add_argument("--vmin", type=float, default=None, help="Override value color-scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Override value color-scale maximum")
    parser.add_argument("--emax", type=float, default=None,
                        help="Override symmetric error color-scale half-range (+/- this)")
    parser.add_argument("--fields", nargs="+", default=None,
                        help="Subset of field labels to animate (e.g. Temperature NO OH); default: all")
    args = parser.parse_args()

    axis = args.slice_axis
    axcfg = SLICE_AXES[axis]
    h_idx, h_sign, h_label = axcfg["h"]
    v_idx, v_sign, v_label = axcfg["v"]

    sage_dir = os.path.abspath(args.sage_dir)
    ann_dir = os.path.abspath(args.ann_dir)
    hybrid_dir = os.path.abspath(args.hybrid_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    want = set(args.fields) if args.fields else None
    fields = [f for f in FIELDS if want is None or field_label(f) in want]
    assert fields, f"No field matched {args.fields}; valid: {[field_label(f) for f in FIELDS]}"

    name_re = re.compile(r"^post\d+_(?P<time>[+-][0-9.eE+-]+)\.h5$")
    sage_by_time = index_by_time(sage_dir, name_re)
    ann_by_time = index_by_time(ann_dir, name_re)
    hybrid_by_time = index_by_time(hybrid_dir, name_re)

    common = sorted(set(sage_by_time) & set(ann_by_time) & set(hybrid_by_time), key=float)
    assert common, "No timestep is common to SAGE + ANN + Hybrid runs"
    common = common[:: args.stride]
    n = len(common)
    print(f"Common time range: t={float(common[0]):.3e}s to t={float(common[-1]):.3e}s "
          f"({n} frames after stride {args.stride})", flush=True)

    # ---- pass 1: extract the slice for every field/model at every frame ----
    # Stored as [n_frames, n_cells] float32 arrays keyed by field label.
    sage = {field_label(f): None for f in fields}
    ann = {field_label(f): None for f in fields}
    hyb = {field_label(f): None for f in fields}
    times = np.empty(n, dtype=np.float64)
    h_coord = v_coord = None
    n_cells = None

    t0 = _time.time()
    for i, tkey in enumerate(common):
        coords_s, data_s, t_s = load(sage_by_time[tkey], fields)
        tree = cKDTree(coords_s)
        mask = np.abs(coords_s[:, axcfg["idx"]]) < args.slice_halfwidth
        times[i] = t_s

        if h_coord is None:
            h_coord = h_sign * coords_s[mask, h_idx]
            v_coord = v_sign * coords_s[mask, v_idx]
            n_cells = int(mask.sum())
            for d in (sage, ann, hyb):
                for lbl in d:
                    d[lbl] = np.empty((n, n_cells), dtype=np.float32)
        elif int(mask.sum()) != n_cells:
            raise RuntimeError(
                f"slice cell count changed at t={t_s:.3e}s "
                f"({int(mask.sum())} vs {n_cells}) -- adaptive mesh not supported"
            )

        for other_dir_by_time, store in (
            (ann_by_time, ann), (hybrid_by_time, hyb)
        ):
            coords_m, data_m, t_m = load(other_dir_by_time[tkey], fields)
            assert abs(t_s - t_m) < 1e-9, (t_s, t_m)
            dist, idx = tree.query(coords_m, k=1)
            if dist.max() > 1e-9:
                print(f"  WARNING t={t_s:.3e}: max nearest-neighbor dist = {dist.max():.3e}",
                      flush=True)
            for f in fields:
                lbl = field_label(f)
                a = np.empty_like(data_m[f])
                a[idx] = data_m[f]
                store[lbl][i] = a[mask].astype(np.float32)

        for f in fields:
            lbl = field_label(f)
            sage[lbl][i] = data_s[f][mask].astype(np.float32)

        if (i + 1) % 25 == 0 or i == n - 1:
            el = _time.time() - t0
            print(f"  loaded frame {i + 1}/{n}  ({el:.0f}s, {el / (i + 1):.2f}s/frame)",
                  flush=True)

    print(f"pass 1 done: {n} frames x {n_cells} slice cells x {len(fields)} fields", flush=True)

    # ---- pass 2: one GIF per field ----
    writer = manimation.PillowWriter(fps=args.fps)
    for f in fields:
        lbl = field_label(f)
        sv, av, hv = sage[lbl], ann[lbl], hyb[lbl]
        err_a = av - sv
        err_h = hv - sv

        # Value scale from the SAGE reference only -- ANN can numerically
        # diverge (T -> 1e4+ K in a blow-up cell); letting that set the scale
        # would wash out every other frame. Such excursions just saturate.
        lo_p = 100.0 - args.clip_percentile
        vmin = args.vmin if args.vmin is not None else float(np.percentile(sv, lo_p))
        vmax = args.vmax if args.vmax is not None else float(np.percentile(sv, args.clip_percentile))
        if vmax <= vmin:
            vmax = vmin + 1e-30
        # Error scale: robust percentile of |err| over all frames of both
        # models -- a blow-up in a handful of late frames stays a small
        # fraction of the whole dataset so it does not dominate.
        if args.emax is not None:
            emax = args.emax
        else:
            emax = float(np.percentile(np.abs(np.concatenate([err_a, err_h])), args.err_percentile))
            # Never resolve an error wider than the reference field's own full
            # span -- a diverged ANN cell just saturates deep red/blue.
            emax = min(emax, float(sv.max() - sv.min()) or emax)
        emax = emax or 1e-30

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.06, wspace=0.25, hspace=0.15)
        (ax_s, ax_a, ax_h), (ax_blank, ax_ea, ax_eh) = axes

        sc_kw = dict(s=args.marker_size, cmap="inferno", vmin=vmin, vmax=vmax)
        ec_kw = dict(s=args.marker_size, cmap="coolwarm", vmin=-emax, vmax=emax)
        pc_s = ax_s.scatter(h_coord, v_coord, c=sv[0], **sc_kw)
        pc_a = ax_a.scatter(h_coord, v_coord, c=av[0], **sc_kw)
        pc_h = ax_h.scatter(h_coord, v_coord, c=hv[0], **sc_kw)
        pc_ea = ax_ea.scatter(h_coord, v_coord, c=err_a[0], **ec_kw)
        pc_eh = ax_eh.scatter(h_coord, v_coord, c=err_h[0], **ec_kw)

        ax_s.set_title("SAGE (reference)")
        ax_a.set_title("ANN")
        ax_h.set_title("ANN Hybrid")
        ax_ea.set_title("ANN - SAGE")
        ax_eh.set_title("ANN Hybrid - SAGE")
        ax_blank.axis("off")
        caption = ax_blank.text(0.5, 0.5, "", ha="center", va="center",
                                fontsize=13, transform=ax_blank.transAxes)

        for ax in (ax_s, ax_a, ax_h, ax_ea, ax_eh):
            ax.set_xlabel(h_label)
            ax.set_ylabel(v_label)
            ax.set_aspect("equal")

        cax_v = fig.add_axes([0.92, 0.55, 0.015, 0.33])
        cax_e = fig.add_axes([0.92, 0.10, 0.015, 0.33])
        fig.colorbar(pc_s, cax=cax_v, label=f"{lbl} (SAGE-scaled)", extend="both")
        fig.colorbar(pc_ea, cax=cax_e, label=f"{lbl} error", extend="both")

        suptitle = fig.suptitle("", fontsize=15)

        def update(k):
            pc_s.set_array(sv[k])
            pc_a.set_array(av[k])
            pc_h.set_array(hv[k])
            pc_ea.set_array(err_a[k])
            pc_eh.set_array(err_h[k])
            suptitle.set_text(
                f"{axis}=0 slice  --  {lbl}  --  t = {times[k]:.3e} s   (frame {k + 1}/{n})"
            )
            caption.set_text(
                f"{lbl}\n\nt = {times[k]:.3e} s\nframe {k + 1} / {n}\n\n"
                f"color scale fixed\n[{vmin:.3g}, {vmax:.3g}]\n"
                f"error scale +/- {emax:.3g}"
            )
            return pc_s, pc_a, pc_h, pc_ea, pc_eh, suptitle, caption

        anim = manimation.FuncAnimation(fig, update, frames=n, blit=False)
        out_fp = os.path.join(out_dir, f"anim_{lbl}.gif")
        ta = _time.time()
        anim.save(out_fp, writer=writer, dpi=args.dpi)
        plt.close(fig)
        print(f"  wrote {out_fp}  ({_time.time() - ta:.0f}s)", flush=True)

    print(f"\nDone: {len(fields)} animations in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
