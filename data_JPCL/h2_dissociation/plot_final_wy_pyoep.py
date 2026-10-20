#!/usr/bin/env python3
"""Make the plot_final.py journal figure with WY and PyOEP overlaid."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import plot_final as original


INPUT = Path("final_merged_dou_sig.csv")
def solver_frame(data: pd.DataFrame, solver: str) -> pd.DataFrame:
    """Expose one solver's prefixed columns under plot_final's field names."""
    frame = data[
        data["density_type"].astype(str).str.strip().str.upper() == "FCI"
    ].copy()
    frame["R_input"] = pd.to_numeric(
        frame["R_bohr"] if "R_bohr" in frame else frame["R"], errors="coerce"
    )
    frame["R_plot"] = frame["R_input"]

    prefix = f"{solver}_"
    for column in data.columns:
        if column.startswith(prefix):
            frame[column[len(prefix):]] = data.loc[frame.index, column]

    frame = (
        frame.sort_values("R_input")
        .drop_duplicates("R_input")
        .replace([np.inf, -np.inf], np.nan)
    )
    frame["SCE+rmZPE"] = pd.to_numeric(
        frame["total_sce_rmzpe_cyl"], errors="coerce"
    )
    return original.reconstruct_cylindrical_ac(frame)


def format_axes(ax):
    """Apply the same journal axes to combined and independent figures."""
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$E$ ")
    ax.set_xlim(0.0, 7.7)
    ax.set_ylim(-1.50, -0.75)
    ax.set_xticks([0.0, 2.0, 4.0, 6.0, 7.7])
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


def add_residual_filtered_columns(frame, curves, threshold=0.001):
    """Mask isolated energy spikes without interpolation or smoothing."""
    filtered = {}
    print(f"\nEnergy-residual filtering at {threshold:.6f} Ha:")
    for label, column in curves.items():
        output_column = f"{column}_residual_filtered"
        clean, rejected = original.filter_isolated_oscillations(
            frame["R_plot"],
            pd.to_numeric(frame[column], errors="coerce"),
            start_x=4.5,
            absolute_tolerance=threshold,
            scale_factor=0.0,
            max_passes=20,
        )
        frame[output_column] = clean
        filtered[label] = output_column
        rejected_r = frame.loc[rejected, "R_plot"].to_numpy(dtype=float)
        print(
            f"  {label:11s}: masked {len(rejected_r):3d} point(s); "
            f"no interpolation"
        )
    return filtered


def mask_sustained_nonconverged_tail(
    frame, curves, search_start=4.5, run_length=2
):
    """Stop curves at the first sustained optimizer-convergence failure.

    Isolated failures can occur on an otherwise usable scan, but once two
    consecutive geometries fail, later isolated ``converged=True`` points no
    longer define a trustworthy continuous potential-energy curve.  The raw
    columns remain unchanged; only plotting columns are masked.
    """
    if "converged" not in frame.columns:
        print("  PyOEP reliability cutoff: unavailable (no convergence flag)")
        return curves

    ordered = frame[["R_plot", "converged"]].copy()
    ordered["R_plot"] = pd.to_numeric(ordered["R_plot"], errors="coerce")
    ordered["converged"] = (
        ordered["converged"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    ordered = ordered.dropna(subset=["R_plot"]).sort_values("R_plot")
    ordered = ordered[ordered["R_plot"] >= search_start]

    failures = (~ordered["converged"]).to_numpy(dtype=bool)
    cutoff = None
    for start in range(0, failures.size - run_length + 1):
        if failures[start : start + run_length].all():
            cutoff = float(ordered.iloc[start]["R_plot"])
            break

    if cutoff is None:
        print("  PyOEP reliability cutoff: no sustained failure found")
        return curves, None

    tail = pd.to_numeric(frame["R_plot"], errors="coerce") >= cutoff
    masked_curves = {}
    for label, column in curves.items():
        output_column = f"{column}_converged_tail"
        frame[output_column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[tail, output_column] = np.nan
        masked_curves[label] = output_column

    print(
        f"  PyOEP reliability cutoff: R = {cutoff:.5f} bohr; "
        f"masked {int(tail.sum())} tail point(s) after {run_length} "
        "consecutive optimizer failures"
    )
    return masked_curves, cutoff


def save_independent_plot(frame, solver, curves, tail_curves=None, cutoff=None):
    """Save one figure containing quantities from exactly one OEP method.

    ``tail_curves``/``cutoff``, when given, add the residual-filtered-only
    (i.e. NOT cut for sustained non-convergence) segment beyond ``cutoff``
    in a visually distinct, faded style, so the unconverged tail is shown
    rather than hidden -- explicitly flagged as unreliable, not
    interpolated/extrapolated/smoothed.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    reference = frame[["R_plot", "E_ref"]].dropna()
    ax.plot(reference["R_plot"], reference["E_ref"], label="FCI",
            **original.STYLE["FCI"])

    for label, column in curves.items():
        # Preserve NaNs so rejected points appear as gaps rather than being
        # silently joined by a straight line.
        valid = frame[["R_plot", column]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna(subset=["R_plot"])
        ax.plot(
            valid["R_plot"], valid[column], label=label,
            markersize=5.5, markeredgecolor="black", markeredgewidth=0.45,
            markevery=3, **original.STYLE[label],
        )

    if tail_curves is not None and cutoff is not None:
        r_all = pd.to_numeric(frame["R_plot"], errors="coerce")
        for label, tail_column in tail_curves.items():
            tail_vals = pd.to_numeric(frame[tail_column], errors="coerce")
            tail_frame = pd.DataFrame({"R_plot": r_all, "v": tail_vals})
            tail_frame = tail_frame[tail_frame["R_plot"] >= cutoff].dropna()
            if tail_frame.empty:
                continue
            base_style = dict(original.STYLE[label])
            base_style["ls"] = ":"
            base_style.pop("marker", None)
            ax.plot(
                tail_frame["R_plot"], tail_frame["v"],
                color=base_style["color"], lw=base_style["lw"] * 0.85,
                ls=":", alpha=0.40, marker="x", markersize=4.5,
                zorder=base_style.get("zorder", 1) - 0.5,
            )
        ax.axvline(cutoff, color="0.5", lw=1.2, ls=(0, (4, 3)), zorder=1)

    format_axes(ax)

    if tail_curves is not None and cutoff is not None:
        ylo, _ = ax.get_ylim()
        ax.text(
            cutoff + 0.1, ylo + 0.02,
            f"PyOEP unconverged\nbeyond $R={cutoff:.1f}$",
            fontsize=13, color="0.35", va="bottom", ha="left",
        )
    solver_title = "PyOEP" if solver == "PYOEP" else solver
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(0.98, 0.98), title=None,
        ncol=2,
        frameon=True,
        framealpha=1.0, edgecolor="black", facecolor="white",
        handlelength=1.8, columnspacing=0.9, labelspacing=0.32,
        borderpad=0.50,
    )
    legend.get_title().set_fontsize(19)
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_linewidth(1.1)
    fig.tight_layout()

    output = Path(f"Fig_H2_total_energy_journal_cylAC_{solver}")
    fig.savefig(output.with_suffix(".eps"), format="eps")
    plt.close(fig)
    print(f"Saved {output.with_suffix('.eps')}")


def main():
    original.configure_publication_style()
    data = pd.read_csv(INPUT, comment="#", skip_blank_lines=True)
    frames = {name: solver_frame(data, name) for name in ("PYOEP",)}

    base_curves = {
        "GL2": "E_gl2_tot",
        "SPL": "E_spl_cyl",
        "genISI2": "E_gisi2_cyl",
        "SCE": "total_sce_cyl",
        "SCE+rmZPE": "SCE+rmZPE",
    }
    # Filter each inversion independently. These plotting-only columns mask
    # points whose leave-one-out residual exceeds 0.01 Ha (applied across
    # the full R range, R=7-10 included). No rejected point is
    # interpolated, smoothed, or extrapolated.
    solver_curves = {}
    solver_tails = {}
    solver_cutoffs = {}
    for solver, frame in frames.items():
        filtered_curves = add_residual_filtered_columns(
            frame, base_curves, 0.01
        )
        if solver == "PYOEP":
            masked_curves, cutoff = mask_sustained_nonconverged_tail(
                frame, filtered_curves
            )
            solver_curves[solver] = masked_curves
            # The unconverged tail keeps only the softer residual filter
            # (no hard sustained-non-convergence cut) -- shown, not hidden,
            # but visually flagged in save_independent_plot.
            solver_tails[solver] = filtered_curves
            solver_cutoffs[solver] = cutoff
        else:
            solver_curves[solver] = filtered_curves

    # Tail-overlay capability (solver_tails/solver_cutoffs) is kept in
    # main()/save_independent_plot() for future use, but the R>7.7 PyOEP
    # tail is not plotted here: the non-convergence there has a known,
    # fixable cause (no RI-fit auxiliary basis for the OEP potential --
    # see PyOEP's own tutorials/Advanced.ipynb), so showing it as
    # "unreliable data" is less useful than simply re-running with the
    # correct basis once that fix lands. Revert to the clean R=0-7.7 cut.
    show_tail = False
    for solver, frame in frames.items():
        save_independent_plot(
            frame, solver, solver_curves[solver],
            tail_curves=solver_tails.get(solver) if show_tail else None,
            cutoff=solver_cutoffs.get(solver) if show_tail else None,
        )

    print(f"Input range: {data['R_bohr'].min():.4f}--{data['R_bohr'].max():.4f} bohr")


if __name__ == "__main__":
    main()
