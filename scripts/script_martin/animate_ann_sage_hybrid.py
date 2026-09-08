"""Animate the temporal evolution of a full-CFD SAGE (reference chemistry) run
against one or more accelerated runs (typically an ANN run, optionally a
second one -- Hybrid, a different architecture, etc.), on a single planar
slice (X=0, Y=0 or Z=0), for Temperature and every species mass fraction.

For each field one animation (GIF) is written with a 2x(1+N) panel layout
(N = number of accelerated runs -- 1 for a single --ann-dir, 2 with
--hybrid-dir/--model too), one frame per shared timestep:

    +-----------+-----------+-----+
    |   SAGE    |    ANN    | ... |   <- raw field, shared color scale
    +-----------+-----------+-----+
    |  (blank / |  ANN      | ... |
    |   caption)|  - SAGE   |     |   <- error vs SAGE, shared +/- scale
    +-----------+-----------+-----+

The value color scale (top row, from the SAGE field) and the symmetric error
color scale (bottom row) are fixed across the whole animation so brightness /
contrast is comparable frame to frame.

The slice cells are binned once onto a regular pixel grid; every frame is then
an ``imshow`` data swap rather than a 30k-point scatter redraw, which is what
makes animating hundreds of frames for 19 fields tractable.

Inputs are raw CONVERGE post-processing snapshots (STREAM_00/CELL_CENTER_DATA)
under sibling ``output/`` directories -- one per run -- with matching
``post*_+<time>.h5`` filenames on each side. Cell ordering can differ between
runs (same mesh, different numbering/partitioning); each accelerated run's
cells are re-indexed to the SAGE cell order via a coordinate KDTree before
comparison. Only timesteps present on every side are animated, matched by the
time encoded in the filename (not by position).

Two-stage use (recommended for the full 19-field run): build the slice cache
once, then render fields in parallel (e.g. a SLURM array) off the cache:

    python animate_ann_sage_hybrid.py ... --cache slices.npz --cache-only
    python animate_ann_sage_hybrid.py ... --cache slices.npz --fields Temperature
    python animate_ann_sage_hybrid.py ... --cache slices.npz --fields NO OH ...

Single-shot use:

    python animate_ann_sage_hybrid.py \\
        --sage-dir   .../outputs_original_COVDE/output \\
        --ann-dir    .../outputs_original_ANN/output \\
        --hybrid-dir .../outputs_original_ANN_Hybrid/output \\
        --slice-axis Y --out-dir .../anim --stride 1 --fps 12

Only pillow (GIF) is required as an animation writer -- no ffmpeg needed.
"""

import argparse
import glob
import os
import re
import time as _time
from copy import copy

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


def slice_indices(coords, axcfg, slab, res):
    """Indices of the cells forming the <axis>=0 slice. The mesh is AMR: a fixed
    |coord| < slab threshold either misses the coarse downstream region (too
    thin) or stacks several refined layers upstream (too thick). Instead: keep a
    generous slab, bucket its cells on the two in-plane coordinates at
    resolution ``res``, and within each bucket keep the single cell closest to
    the plane. That yields a clean one-cell-thick slice over the *whole* domain
    at every refinement level."""
    c = coords[:, axcfg["idx"]]
    cand = np.where(np.abs(c) < slab)[0]
    h = coords[cand, axcfg["h"][0]]
    v = coords[cand, axcfg["v"][0]]
    hb = np.round(h / res).astype(np.int64)
    vb = np.round(v / res).astype(np.int64)
    key = hb * 4_000_003 + vb
    order = np.argsort(np.abs(c[cand]), kind="stable")   # nearest-to-plane first
    _, first = np.unique(key[order], return_index=True)
    return np.sort(cand[order[first]])


def build_cache(args, models):
    """One pass over the shared timesteps: extract every field on the slice for
    SAGE and every accelerated run in ``models`` (list of (label, dir)),
    re-indexing each onto SAGE cell order. Returns a dict of [n_frames, n_cells]
    float32 arrays plus slice coords. Stored .npz-flat: model arrays go under
    model_0, model_1, ... (order matches the model_labels array) since npz
    cannot nest a dict of arrays."""
    axis = args.slice_axis
    axcfg = SLICE_AXES[axis]
    h_idx, h_sign, _ = axcfg["h"]
    v_idx, v_sign, _ = axcfg["v"]

    name_re = re.compile(r"^post\d+_(?P<time>[+-][0-9.eE+-]+)\.h5$")
    sage_by_time = index_by_time(os.path.abspath(args.sage_dir), name_re)
    model_by_time = [(name, index_by_time(os.path.abspath(d), name_re)) for name, d in models]

    common = set(sage_by_time)
    for _, by_time in model_by_time:
        common &= set(by_time)
    common = sorted(common, key=float)
    assert common, f"No timestep is common to SAGE + {[n for n, _ in models]}"
    common = common[:: args.stride]
    n = len(common)
    print(f"Common time range: t={float(common[0]):.3e}s to t={float(common[-1]):.3e}s "
          f"({n} frames after stride {args.stride})", flush=True)

    labels = [field_label(f) for f in FIELDS]
    sage = None
    model_arrays = [None] * len(models)
    times = np.empty(n, dtype=np.float64)
    h_coord = v_coord = None
    n_cells = None

    t0 = _time.time()
    sel = None
    for i, tkey in enumerate(common):
        coords_s, data_s, t_s = load(sage_by_time[tkey], FIELDS)
        tree = cKDTree(coords_s)
        times[i] = t_s

        if h_coord is None:
            sel = slice_indices(coords_s, axcfg, args.slice_halfwidth, args.slice_res)
            h_coord = h_sign * coords_s[sel, h_idx]
            v_coord = v_sign * coords_s[sel, v_idx]
            n_cells = sel.size
            print(f"  slice: {n_cells} cells, "
                  f"{axcfg['h'][2]} in [{h_coord.min():.3f}, {h_coord.max():.3f}], "
                  f"{axcfg['v'][2]} in [{v_coord.min():.3f}, {v_coord.max():.3f}]", flush=True)
            sage = np.empty((len(FIELDS), n, n_cells), dtype=np.float32)
            model_arrays = [np.empty_like(sage) for _ in models]
        elif coords_s.shape[0] != sel_nmesh:
            raise RuntimeError(
                f"mesh cell count changed at t={t_s:.3e}s "
                f"({coords_s.shape[0]} vs {sel_nmesh}) -- adaptive mesh not supported"
            )
        sel_nmesh = coords_s.shape[0]

        for fj, f in enumerate(FIELDS):
            sage[fj, i] = data_s[f][sel].astype(np.float32)

        for (name, by_time), store in zip(model_by_time, model_arrays):
            coords_m, data_m, t_m = load(by_time[tkey], FIELDS)
            assert abs(t_s - t_m) < 1e-9, (name, t_s, t_m)
            dist, idx = tree.query(coords_m, k=1)
            if dist.max() > 1e-9:
                print(f"  WARNING t={t_s:.3e} [{name}]: max nearest-neighbor dist = {dist.max():.3e}",
                      flush=True)
            for fj, f in enumerate(FIELDS):
                a = np.empty_like(data_m[f])
                a[idx] = data_m[f]
                store[fj, i] = a[sel].astype(np.float32)

        if (i + 1) % 25 == 0 or i == n - 1:
            el = _time.time() - t0
            print(f"  loaded frame {i + 1}/{n}  ({el:.0f}s, {el / (i + 1):.2f}s/frame)", flush=True)

    print(f"pass 1 done: {n} frames x {n_cells} slice cells x {len(FIELDS)} fields "
          f"x {len(models)} model(s)", flush=True)
    cache = dict(
        labels=np.array(labels), times=times, h_coord=h_coord, v_coord=v_coord,
        sage=sage, axis=np.array(axis), model_labels=np.array([n for n, _ in models]),
    )
    for mi, arr in enumerate(model_arrays):
        cache[f"model_{mi}"] = arr
    return cache


def make_binner(h_coord, v_coord, grid, fill_radius):
    """Map every pixel of a regular grid to its nearest slice cell once (KDTree
    on the 2-D slice coords). ``to_img`` then turns a per-cell vector into a
    [ny, nx] image by plain fancy-indexing -- no per-frame averaging, no gaps
    inside the meshed region. Pixels farther than ``fill_radius`` from any cell
    (the empty background outside the flow wedge) are left NaN."""
    h0, h1 = float(h_coord.min()), float(h_coord.max())
    v0, v1 = float(v_coord.min()), float(v_coord.max())
    nx = int(grid)
    ny = max(1, int(round(grid * (v1 - v0) / (h1 - h0))))
    hc = h0 + (np.arange(nx) + 0.5) * (h1 - h0) / nx
    vc = v0 + (np.arange(ny) + 0.5) * (v1 - v0) / ny
    hh, vv = np.meshgrid(hc, vc)
    dist, near = cKDTree(np.c_[h_coord, v_coord]).query(np.c_[hh.ravel(), vv.ravel()], k=1)
    bad = dist > fill_radius

    def to_img(vals):
        img = vals[near].astype(np.float64)
        img[bad] = np.nan
        return img.reshape(ny, nx)

    return dict(extent=[h0, h1, v0, v1], nx=nx, ny=ny, to_img=to_img)


def render_field(lbl, times, sv, mvals, binner, axis, args, out_dir, model_labels):
    """mvals: list of [n_frames, n_cells] arrays, one per accelerated model,
    matching model_labels (order and length)."""
    n_models = len(mvals)
    errs = [mv - sv for mv in mvals]
    n = sv.shape[0]

    lo_p = 100.0 - args.clip_percentile
    vmin = args.vmin if args.vmin is not None else float(np.percentile(sv, lo_p))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(sv, args.clip_percentile))
    if vmax <= vmin:
        vmax = vmin + 1e-30
    if args.emax is not None:
        emax = args.emax
    else:
        emax = float(np.percentile(np.abs(np.concatenate(errs)), args.err_percentile))
        # never resolve an error wider than the reference field's own span --
        # a diverged ANN cell just saturates deep red / blue
        emax = min(emax, float(sv.max() - sv.min()) or emax)
    emax = emax or 1e-30

    to_img = binner["to_img"]
    val_cmap = copy(matplotlib.colormaps["inferno"])
    val_cmap.set_bad("#dddddd")
    err_cmap = copy(matplotlib.colormaps["coolwarm"])
    err_cmap.set_bad("#dddddd")
    im_kw = dict(extent=binner["extent"], origin="lower", interpolation="nearest", aspect="equal")

    # size the figure to the slice aspect so the equal-aspect panels fill it
    n_cols = 1 + n_models
    e = binner["extent"]
    panel_ar = (e[1] - e[0]) / (e[3] - e[2])           # width / height of one panel
    pw = 5.4
    fig_w = n_cols * pw + 2.4
    fig_h = 2 * (pw / panel_ar) + 1.7
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.88, bottom=0.1, wspace=0.2, hspace=0.35)
    top_row, bottom_row = axes
    ax_s, model_axes = top_row[0], top_row[1:]
    ax_blank, err_axes = bottom_row[0], bottom_row[1:]

    im_s = ax_s.imshow(to_img(sv[0]), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw)
    ax_s.set_title("SAGE (reference)")
    im_models = []
    for ax, mv, name in zip(model_axes, mvals, model_labels):
        im_models.append(ax.imshow(to_img(mv[0]), cmap=val_cmap, vmin=vmin, vmax=vmax, **im_kw))
        ax.set_title(name)
    im_errs = []
    for ax, err, name in zip(err_axes, errs, model_labels):
        im_errs.append(ax.imshow(to_img(err[0]), cmap=err_cmap, vmin=-emax, vmax=emax, **im_kw))
        ax.set_title(f"{name} - SAGE")

    ax_blank.axis("off")
    caption = ax_blank.text(0.5, 0.5, "", ha="center", va="center", fontsize=13,
                            transform=ax_blank.transAxes)

    axcfg = SLICE_AXES[axis]
    for ax in [ax_s] + list(model_axes) + list(err_axes):
        ax.set_xlabel(axcfg["h"][2])
        ax.set_ylabel(axcfg["v"][2])

    fig.colorbar(im_s, ax=[ax_s] + list(model_axes), fraction=0.015, pad=0.02,
                 label=f"{lbl} (SAGE-scaled)", extend="both")
    if im_errs:
        fig.colorbar(im_errs[0], ax=list(err_axes), fraction=0.015, pad=0.02,
                     label=f"{lbl} error", extend="both")
    suptitle = fig.suptitle("", fontsize=15)

    def update(k):
        im_s.set_data(to_img(sv[k]))
        for im, mv in zip(im_models, mvals):
            im.set_data(to_img(mv[k]))
        for im, err in zip(im_errs, errs):
            im.set_data(to_img(err[k]))
        suptitle.set_text(
            f"{axis}=0 slice  --  {lbl}  --  t = {times[k]:.3e} s   (frame {k + 1}/{n})"
        )
        caption.set_text(
            f"{lbl}\n\nt = {times[k]:.3e} s\nframe {k + 1} / {n}\n\n"
            f"value scale (SAGE)\n[{vmin:.3g}, {vmax:.3g}]\nerror scale +/- {emax:.3g}"
        )
        return [im_s] + im_models + im_errs + [suptitle, caption]

    anim = manimation.FuncAnimation(fig, update, frames=n, blit=False)
    out_fp = os.path.join(out_dir, f"anim_{lbl}.gif")
    ta = _time.time()
    anim.save(out_fp, writer=manimation.PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    print(f"  wrote {out_fp}  ({_time.time() - ta:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sage-dir", help="Directory of SAGE (reference chemistry) post*.h5 snapshots")
    parser.add_argument("--ann-dir", help="Directory of ANN-accelerated post*.h5 snapshots")
    parser.add_argument("--hybrid-dir", default=None,
                        help="Optional second accelerated run's post*.h5 snapshots (e.g. ANN-Hybrid) -- "
                             "animated alongside --ann-dir if given, omit for a single-model animation")
    parser.add_argument("--ann-label", default="ANN",
                        help="Panel/title label for the --ann-dir run (default: ANN)")
    parser.add_argument("--hybrid-label", default="ANN Hybrid",
                        help="Panel/title label for the --hybrid-dir run (default: ANN Hybrid)")
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=DIR",
                        help="Additional accelerated run to animate, beyond --ann-dir/--hybrid-dir -- "
                             "repeatable")
    parser.add_argument("--out-dir", required=True, help="Output directory for the per-field GIFs")
    parser.add_argument("--cache", default=None,
                        help="Path to an .npz slice cache: loaded if it exists (skips pass 1), "
                             "else built from the --*-dir inputs and written here")
    parser.add_argument("--cache-only", action="store_true",
                        help="Build the --cache file and exit without rendering")
    parser.add_argument("--slice-axis", choices=["X", "Y", "Z"], default="Y",
                        help="Coordinate held ~constant to define the slice plane (default: Y)")
    parser.add_argument("--slice-halfwidth", type=float, default=0.005,
                        help="Half-width [m] of the |coord| < value slab pre-filter; within it only "
                             "the cell closest to the plane is kept per --slice-res bucket, so this "
                             "just needs to exceed the coarsest cell's half-size (it is NOT the "
                             "final slice thickness)")
    parser.add_argument("--slice-res", type=float, default=0.0005,
                        help="In-plane bucket size [m] for de-duplicating the slab down to one "
                             "cell-thick; ~ the finest cell size (smaller keeps more upstream detail)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth shared timestep (default: 1 = all)")
    parser.add_argument("--grid", type=int, default=320,
                        help="Horizontal pixel resolution the slice cells are binned onto")
    parser.add_argument("--fill-radius", type=float, default=0.006,
                        help="Pixels farther than this [m] from any slice cell stay blank "
                             "(should exceed the coarsest cell spacing so the domain fills solid)")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second")
    parser.add_argument("--dpi", type=int, default=85, help="Animation raster resolution")
    parser.add_argument("--clip-percentile", type=float, default=99.5,
                        help="Percentile (p and 100-p) of the SAGE field for the fixed value scale")
    parser.add_argument("--err-percentile", type=float, default=99.0,
                        help="Percentile of |error| for the fixed symmetric error scale")
    parser.add_argument("--vmin", type=float, default=None, help="Override value color-scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Override value color-scale maximum")
    parser.add_argument("--emax", type=float, default=None,
                        help="Override symmetric error color-scale half-range (+/- this)")
    parser.add_argument("--fields", nargs="+", default=None,
                        help="Subset of field labels to animate (e.g. Temperature NO OH); default: all")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    models = [(args.ann_label, args.ann_dir)] if args.ann_dir else []
    if args.hybrid_dir:
        models.append((args.hybrid_label, args.hybrid_dir))
    for spec in args.model:
        label, _, mdir = spec.partition("=")
        assert mdir, f"--model expects LABEL=DIR, got {spec!r}"
        models.append((label, mdir))

    if args.cache and os.path.exists(args.cache):
        print(f"loading slice cache {args.cache}", flush=True)
        c = np.load(args.cache, allow_pickle=False)
        cache = {k: c[k] for k in c.files}
    else:
        assert args.sage_dir and models, (
            "--sage-dir and at least one of --ann-dir/--hybrid-dir/--model are required "
            "when no --cache file exists"
        )
        cache = build_cache(args, models)
        if args.cache:
            print(f"writing slice cache {args.cache}", flush=True)
            np.savez(args.cache, **cache)

    if args.cache_only:
        print("cache-only: done", flush=True)
        return

    axis = str(cache["axis"])
    times = cache["times"]
    labels = list(cache["labels"])
    model_labels = [str(lbl) for lbl in cache["model_labels"]]
    n_models = len(model_labels)
    binner = make_binner(cache["h_coord"], cache["v_coord"], args.grid, args.fill_radius)
    print(f"binned {cache['h_coord'].size} slice cells onto {binner['nx']}x{binner['ny']} grid",
          flush=True)

    want = set(args.fields) if args.fields else None
    todo = [(j, lbl) for j, lbl in enumerate(labels) if want is None or lbl in want]
    assert todo, f"No field matched {args.fields}; valid: {labels}"

    for j, lbl in todo:
        mvals = [cache[f"model_{mi}"][j] for mi in range(n_models)]
        render_field(lbl, times, cache["sage"][j], mvals, binner, axis, args, out_dir, model_labels)

    print(f"\nDone: {len(todo)} animations in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
