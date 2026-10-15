



#!/usr/bin/env python3
"""Plot H2 total-energy curves with SPL/genISI2 reconstructed from 3D SCE.

The reconstructed adiabatic-connection curves use the consistent cylindrical
strong-coupling pair

    W_inf  = winf_cyl
    Wp_inf = winf_prime_rmzpe_cyl

For a two-electron singlet, exact exchange on the FCI density is constructed
directly from the FCI-density Hartree energy,

    E_xx_FCI = -J[n_FCI] / 2.

The stored W0 is used only as a validation value.  The reconstructed formulas
use E_xx_FCI explicitly and are assembled as

    E_SPL     = E_base_x + E_c_SPL
    E_genISI2 = E_base_x + E_c_genISI2

Because spl_correlation() and genisi2_correlation() return correlation
integrals, the preferred total-energy base is the explicit HF total

    E_base_x = E_hf_tot.

For older CSV files, E_ref - E_c is retained as the equivalent fallback.  This
is the same total-energy convention used when the production CSV created
E_spl and E_gisi2.  It must not be replaced by an interacting-kinetic
T[FCI]+V_ne+J+E_nn+E_x expression, which mixes the interacting kinetic energy
with a Kohn--Sham adiabatic-connection correlation integral.

Important: W1_exact - E_c is not exact exchange.  W1_exact is the physical
lambda=1 potential contribution V_ee-J, whereas E_c in this CSV is the total
FCI-HF correlation energy and contains a kinetic contribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from scipy.integrate import IntegrationWarning
from scipy.signal import savgol_filter


BOHR_TO_ANGSTROM = 1#0.529177210903


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct cylindrical-SCE SPL/genISI2 curves and plot them "
            "together with FCI, GL2, SCE, rmZPE, and ACM curves."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="final_merged_dou_sig.csv",
        help="Input merged CSV.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default="Fig_H2_total_energy_journal_cylAC",
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--input-unit",
        choices=("Bohr", "Angstrom"),
        default="Bohr",
        help="Unit stored in the CSV R/R_bohr column.",
    )
    parser.add_argument(
        "--x-unit",
        choices=("Bohr", "Angstrom"),
        default="Bohr",
        help="Unit displayed on the horizontal axis.",
    )
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-min", type=float, default=-1.50)
    parser.add_argument("--y-max", type=float, default=-0.75)
    parser.add_argument(
        "--diagnostic-r",
        type=float,
        default=10.0,
        help="Bond length in input units used for the numerical check.",
    )
    parser.add_argument(
        "--include-mp2",
        action="store_true",
        help="Plot MP2 when MP2-density rows are available.",
    )
    parser.add_argument(
        "--include-stored-ac",
        action="store_true",
        help="Also plot the legacy E_spl and E_gisi2 columns for comparison.",
    )
    parser.add_argument(
        "--save-reconstructed-csv",
        default=None,
        help=(
            "Optional path for an FCI-row CSV containing the reconstructed "
            "columns and all original columns."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figures without opening an interactive window.",
    )
    parser.add_argument(
        "--no-oscillation-filter",
        action="store_true",
        help="Disable removal of isolated long-range plotting artifacts.",
    )
    parser.add_argument(
        "--oscillation-start",
        type=float,
        default=4.5,
        help=(
            "Apply the plotting-only artifact filter at/after this R value "
            "in the displayed x-axis unit."
        ),
    )
    parser.add_argument(
        "--oscillation-abs-tol",
        type=float,
        default=0.003,
        help=(
            "Minimum local-interpolation residual required to reject an "
            "isolated point, in Hartree."
        ),
    )
    parser.add_argument(
        "--oscillation-scale",
        type=float,
        default=4.0,
        help=(
            "Multiplier applied to the normal local point-to-point variation "
            "when setting the artifact threshold."
        ),
    )
    parser.add_argument(
        "--oscillation-passes",
        type=int,
        default=4,
        help="Maximum number of isolated-point rejection passes.",
    )
    parser.add_argument(
        "--oscillation-window",
        type=int,
        default=11,
        help=(
            "Odd Savitzky-Golay window used only on the long-range plotted "
            "tails after isolated spikes are removed."
        ),
    )
    return parser.parse_args()


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column aligned to df, or all-NaN if it is absent."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(
            "The cylindrical reconstruction requires these missing columns: "
            + ", ".join(missing)
        )


def spl_correlation(W0: float, gl2: float, W_inf: float) -> float:
    """SPL correlation energy obtained by integrating W_lambda - W0."""
    if not np.all(np.isfinite([W0, gl2, W_inf])):
        return np.nan
    if abs(gl2) < 1.0e-12 or abs(W_inf - W0) < 1.0e-12:
        return 0.0

    chi = 2.0 * gl2 / (W_inf - W0)
    argument = 1.0 + 2.0 * chi
    if argument <= 0.0:
        return np.nan

    return float(
        (W0 - W_inf)
        * ((np.sqrt(argument) - 1.0 - chi) / chi)
    )


def genisi2_correlation(
    W0: float,
    gl2: float,
    W_inf: float,
    Wp_inf: float,
) -> float:
    """genISI2 correlation energy obtained from numerical lambda integration."""
    if not np.all(np.isfinite([W0, gl2, W_inf, Wp_inf])):
        return np.nan
    if abs(gl2) < 1.0e-12:
        return 0.0
    if abs(W0) < 1.0e-30 or abs(W_inf) < 1.0e-30:
        return np.nan
    if Wp_inf <= 1.0e-30:
        return np.nan

    d = 3.5
    l1 = 10.65
    l2 = 3.6
    b = (W0 - W_inf) * (1.0 + d)
    c = b**2 / (4.0 * Wp_inf**2)
    W0_prime = 2.0 * gl2
    p = W0_prime / W0
    ratio_cubed = (W0 / W_inf) ** 3
    r1 = l1 * ratio_cubed
    r2 = l2 * ratio_cubed

    def integrand(lam):
        sqrt_term = np.sqrt(1.0 + c * lam)
        u_isi = (
            W_inf
            + b
            * (2.0 + c * lam + 2.0 * d * sqrt_term)
            / (2.0 * sqrt_term * (d + sqrt_term) ** 2)
        )

        p_lambda = p * lam
        a1 = W0
        a2 = (
            -(u_isi - W0) / p_lambda
            if abs(p_lambda) > 1.0e-30
            else 0.0
        )

        W_a = (a1 * p_lambda) / (1.0 + r1 * p_lambda) ** 3
        W_b = (a2 * p_lambda) / (1.0 + r2 * p_lambda) ** 3
        return u_isi + W_a + W_b - W0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            value, _ = integrate.quad(
                integrand,
                0.0,
                1.0,
                limit=300,
                epsabs=1.0e-10,
                epsrel=1.0e-10,
            )
        return float(value)
    except (IntegrationWarning, FloatingPointError, ValueError, ZeroDivisionError):
        return np.nan


def construct_fci_exact_exchange(
    df: pd.DataFrame,
    tolerance: float = 1.0e-8,
) -> pd.Series:
    """Construct and validate exact exchange for singlet H2.

    The FCI density is the density represented by each selected FCI row.
    For a two-electron singlet, exact exchange is rigorously -J/2.
    """
    _require_columns(df, ("J", "W0"))

    E_xx_from_fci_density = -0.5 * _numeric(df, "J")
    if "E_xx_FCI" in df.columns:
        supplied = _numeric(df, "E_xx_FCI")
        supplied_error = (supplied - E_xx_from_fci_density).abs()
        finite_supplied_error = supplied_error[
            np.isfinite(supplied_error)
        ]
        if (
            not finite_supplied_error.empty
            and float(finite_supplied_error.max()) > tolerance
        ):
            raise ValueError(
                "The supplied E_xx_FCI is inconsistent with -J/2: "
                f"max error={finite_supplied_error.max():.3e} Hartree."
            )
        E_xx_FCI = supplied.combine_first(E_xx_from_fci_density)
    else:
        E_xx_FCI = E_xx_from_fci_density

    stored_error = (_numeric(df, "W0") - E_xx_FCI).abs()
    finite_stored_error = stored_error[np.isfinite(stored_error)]
    maximum_stored_error = (
        float(finite_stored_error.max())
        if not finite_stored_error.empty
        else np.nan
    )
    print(
        "Maximum |stored W0 - E_xx_FCI(-J/2)| = "
        f"{maximum_stored_error:.3e} Hartree"
    )
    if (
        np.isfinite(maximum_stored_error)
        and maximum_stored_error > tolerance
    ):
        raise ValueError(
            "Stored W0 is not the exact exchange of the FCI density: "
            f"max error={maximum_stored_error:.3e} Hartree."
        )

    df["E_xx_FCI"] = E_xx_FCI
    df["W0_stored_minus_E_xx_FCI"] = (
        _numeric(df, "W0") - E_xx_FCI
    )

    # Diagnostic only: demonstrate why W1_exact-E_c must not be substituted
    # for exact exchange.
    if {"W1_exact", "E_c"}.issubset(df.columns):
        false_exchange = _numeric(df, "W1_exact") - _numeric(df, "E_c")
        df["W1_exact_minus_E_c"] = false_exchange
        df["W1_exact_minus_E_c_minus_E_xx_FCI"] = (
            false_exchange - E_xx_FCI
        )
        false_error = (
            df["W1_exact_minus_E_c_minus_E_xx_FCI"].abs().dropna()
        )
        if not false_error.empty:
            print(
                "Maximum |(W1_exact-E_c)-E_xx_FCI| = "
                f"{false_error.max():.3e} Hartree "
                "(therefore W1_exact-E_c is not exchange)"
            )

    return E_xx_FCI


def construct_exchange_including_base(
    df: pd.DataFrame,
    E_xx_FCI: pd.Series,
) -> pd.Series:
    """Build the exchange-including base for a correlation-integral model.

    Priority:
      1. E_hf_tot, explicitly stored in the corrected HF/FCI CSV.
      2. E_ref-E_c, the equivalent fallback for older post-HF ACM data.
      3. E_dft_base from a production script (already includes exchange).
      4. T_s_WY + E_ne + J + E_nn + E_xx_FCI.

    Interacting E_kin/ts_cyl fallbacks are deliberately excluded: adding an
    AC correlation integral to an interacting-kinetic base double counts the
    kinetic-correlation bookkeeping.
    """
    base = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series("", index=df.index, dtype=object)

    candidates: list[tuple[str, pd.Series]] = []

    if "E_hf_tot" in df.columns:
        candidates.append(
            (
                "stored HF total: E_hf_tot",
                _numeric(df, "E_hf_tot"),
            )
        )

    if {"E_ref", "E_c"}.issubset(df.columns):
        candidates.append(
            (
                "post-HF ACM base: E_ref - E_c",
                _numeric(df, "E_ref") - _numeric(df, "E_c"),
            )
        )

    if "E_dft_base" in df.columns:
        candidates.append(("E_dft_base", _numeric(df, "E_dft_base")))

    if (
        "T_s_WY" in df.columns
        and ("E_ne" in df.columns or "E_vne" in df.columns)
        and {"J", "E_nn"}.issubset(df.columns)
    ):
        E_ne = (
            _numeric(df, "E_ne")
            if "E_ne" in df.columns
            else _numeric(df, "E_vne")
        )
        candidates.append(
            (
                "exact-KS base: T_s_WY + E_ne + J + E_nn + E_xx_FCI",
                _numeric(df, "T_s_WY")
                + E_ne
                + _numeric(df, "J")
                + _numeric(df, "E_nn")
                + E_xx_FCI,
            )
        )

    for label, candidate in candidates:
        use = base.isna() & candidate.notna()
        base.loc[use] = candidate.loc[use]
        source.loc[use] = label

    if base.notna().sum() == 0:
        raise KeyError(
            "Could not construct the exchange-including base. Provide "
            "E_ref/E_c, E_dft_base, or T_s_WY/E_ne/J/E_nn."
        )

    df["E_ac_base_source"] = source
    return base


def reconstruct_cylindrical_ac(df: pd.DataFrame) -> pd.DataFrame:
    """Add SPL/genISI2 columns based only on the cylindrical SCE pair."""
    _require_columns(
        df,
        (
            "W0",
            "winf_cyl",
            "winf_prime_rmzpe_cyl",
        ),
    )

    for column in (
        "W0",
        "gl2",
        "W0_prime",
        "winf_cyl",
        "winf_prime_rmzpe_cyl",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "gl2" in df.columns:
        gl2 = _numeric(df, "gl2")
    elif "W0_prime" in df.columns:
        gl2 = 0.5 * _numeric(df, "W0_prime")
        print("Reconstruction: gl2 was recovered as W0_prime / 2.")
    else:
        raise KeyError("The reconstruction requires either 'gl2' or 'W0_prime'.")

    E_xx_FCI = construct_fci_exact_exchange(df)
    df["E_ac_base_cyl"] = construct_exchange_including_base(
        df,
        E_xx_FCI,
    )
    df["E_c_spl_cyl"] = [
        spl_correlation(W0, g2, W_inf)
        for W0, g2, W_inf in zip(
            E_xx_FCI,
            gl2,
            _numeric(df, "winf_cyl"),
        )
    ]
    df["E_c_gisi2_cyl"] = [
        genisi2_correlation(W0, g2, W_inf, Wp_inf)
        for W0, g2, W_inf, Wp_inf in zip(
            E_xx_FCI,
            gl2,
            _numeric(df, "winf_cyl"),
            _numeric(df, "winf_prime_rmzpe_cyl"),
        )
    ]

    df["E_spl_cyl"] = df["E_ac_base_cyl"] + df["E_c_spl_cyl"]
    df["E_gisi2_cyl"] = df["E_ac_base_cyl"] + df["E_c_gisi2_cyl"]
    df["delta_E_spl_cyl_FCI"] = (
        df["E_spl_cyl"] - _numeric(df, "E_ref")
    )
    df["delta_E_gisi2_cyl_FCI"] = (
        df["E_gisi2_cyl"] - _numeric(df, "E_ref")
    )

    source_counts = (
        df.loc[df["E_ac_base_cyl"].notna(), "E_ac_base_source"]
        .value_counts()
        .to_dict()
    )
    print(f"AC base construction used: {source_counts}")
    return df


def load_data(csv_path: str, input_unit: str, output_unit: str):
    # comment="#" supports sectioned calculation CSVs. Repeated section
    # headers are removed below.
    df_all = pd.read_csv(csv_path, comment="#", skip_blank_lines=True)

    if "density_type" not in df_all.columns:
        raise KeyError("The CSV does not contain the required 'density_type' column.")

    density_text = df_all["density_type"].astype(str).str.strip()
    df_all = df_all[density_text.str.lower() != "density_type"].copy()

    r_source = "R_bohr" if "R_bohr" in df_all.columns else "R"
    if r_source not in df_all.columns:
        raise KeyError("The CSV contains neither 'R_bohr' nor 'R'.")

    df_all["R_input"] = pd.to_numeric(df_all[r_source], errors="coerce")

    if input_unit == output_unit:
        df_all["R_plot"] = df_all["R_input"]
    elif input_unit == "Bohr" and output_unit == "Angstrom":
        df_all["R_plot"] = df_all["R_input"] * BOHR_TO_ANGSTROM
    else:
        df_all["R_plot"] = df_all["R_input"] / BOHR_TO_ANGSTROM

    df_mp2 = df_all[
        df_all["density_type"].astype(str).str.strip().str.upper() == "MP2"
    ].copy()
    df_mp2 = (
        df_mp2.sort_values("R_input")
        .drop_duplicates(subset=["R_input"], keep="first")
    )

    df = df_all[
        df_all["density_type"].astype(str).str.strip().str.upper() == "FCI"
    ].copy()
    df = (
        df.sort_values("R_input")
        .drop_duplicates(subset=["R_input"], keep="first")
    )

    if df.empty:
        raise ValueError("No FCI-density rows were found in the CSV.")

    if not df_mp2.empty:
        mp2_map = df_mp2[["R_input", "E_ref"]].rename(
            columns={"E_ref": "E_mp2_tot"}
        )
        df = df.merge(mp2_map, on="R_input", how="left")
    else:
        df["E_mp2_tot"] = np.nan

    if "total_sce_rmzpe_cyl" in df.columns:
        df["SCE+rmZPE"] = pd.to_numeric(
            df["total_sce_rmzpe_cyl"], errors="coerce"
        )
    elif {
        "total_sce_cyl",
        "winf_prime_rmzpe_cyl",
    }.issubset(df.columns):
        df["SCE+rmZPE"] = (
            pd.to_numeric(df["total_sce_cyl"], errors="coerce")
            + pd.to_numeric(df["winf_prime_rmzpe_cyl"], errors="coerce")
        )
    else:
        df["SCE+rmZPE"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_plot"])
    return reconstruct_cylindrical_ac(df)


def make_curve_map(include_mp2: bool, include_stored_ac: bool):
    curves = {
        "FCI": "E_ref",
        # "Exact ACM": "E_acm_exact",
    }
    if include_mp2:
        curves["MP2"] = "E_mp2_tot"

    curves.update(
        {
            "GL2": "E_gl2_tot",
            "SPL": "E_spl_cyl",
            # "SPL-ACM": "E_spl_acm",
            # "SPL": "E_spl_acm",

            "genISI2": "E_gisi2_cyl",
            # "genISI2-ACM": "E_gisi2_acm",
            "SCE": "total_sce_cyl",
            "SCE+rmZPE": "SCE+rmZPE",
        }
    )

    if include_stored_ac:
        curves["SPL (stored)"] = "E_spl"
        curves["genISI2 (stored)"] = "E_gisi2"
    return curves


def configure_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 40,
            "axes.titlesize": 40,
            "xtick.labelsize": 34,
            "ytick.labelsize": 34,
            "legend.fontsize": 19,
            "axes.linewidth": 2.5,
            "xtick.major.width": 2.5,
            "ytick.major.width": 2.5,
            "xtick.minor.width": 1.5,
            "ytick.minor.width": 1.5,
            "xtick.major.size": 12,
            "ytick.major.size": 12,
            "xtick.minor.size": 6,
            "ytick.minor.size": 6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "hatch.linewidth": 1.8,
            "savefig.dpi": 800,
            "savefig.bbox": "tight",
        }
    )


STYLE = {
    "FCI": dict(color="black", lw=3.2, ls="-", marker=None, zorder=20),
    "Exact ACM": dict(
        color="#666666", lw=1.8, ls="--", marker="o", zorder=18
    ),
    "MP2": dict(color="#009E73", lw=2.2, ls="--", marker="v", zorder=6),
    "GL2": dict(color="#0072B2", lw=2.2, ls="-.", marker="^", zorder=5),
    "SPL": dict(color="#E69F00", lw=2.7, ls="-", marker="D", zorder=12),
    "SPL-ACM": dict(
        color="#7B2CBF", lw=2.4, ls="--", marker="P", zorder=13
    ),
    "genISI2": dict(
        color="#D55E00", lw=2.7, ls=":", marker="<", zorder=10
    ),
    "genISI2-ACM": dict(
        color="#56B4E9", lw=2.4, ls="-.", marker=">", zorder=11
    ),
    "SCE": dict(color="#CC79A7", lw=2.4, ls="--", marker="s", zorder=4),
    "SCE+rmZPE": dict(color="red", lw=2.6, ls="-", marker="X", zorder=14),
    "SPL (stored)": dict(
        color="#B8860B", lw=1.5, ls=(0, (3, 2)), marker=None, zorder=3
    ),
    "genISI2 (stored)": dict(
        color="#8B4513", lw=1.5, ls=(0, (1, 2)), marker=None, zorder=3
    ),
}


def _interpolate_column(df: pd.DataFrame, column: str, target_r: float) -> float:
    valid = df[["R_input", column]].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid.sort_values("R_input").drop_duplicates("R_input")
    if len(valid) < 2:
        return np.nan
    if not (valid["R_input"].min() <= target_r <= valid["R_input"].max()):
        return np.nan
    return float(PchipInterpolator(valid["R_input"], valid[column])(target_r))


def print_diagnostic(df: pd.DataFrame, target_r: float, input_unit: str):
    columns = (
        "E_ref",
        "W0",
        "E_xx_FCI",
        "W0_stored_minus_E_xx_FCI",
        "W1_exact_minus_E_c",
        "W1_exact_minus_E_c_minus_E_xx_FCI",
        "gl2",
        "winf_cyl",
        "winf_prime_rmzpe_cyl",
        "E_ac_base_cyl",
        "E_c_spl_cyl",
        "E_spl_cyl",
        "delta_E_spl_cyl_FCI",
        "E_c_gisi2_cyl",
        "E_gisi2_cyl",
        "delta_E_gisi2_cyl_FCI",
    )

    print("\n" + "=" * 74)
    print(
        f"CYLINDRICAL SPL/genISI2 CHECK AT R = {target_r:.6f} {input_unit}"
    )
    print("=" * 74)
    for column in columns:
        if column in df.columns:
            value = _interpolate_column(df, column, target_r)
            print(f"{column:29s} = {value: .12f}")

    gisi2 = _interpolate_column(df, "E_gisi2_cyl", target_r)
    if np.isfinite(gisi2):
        print(f"{'E_gisi2_cyl - (-0.95)':29s} = {gisi2 + 0.95: .12f}")
    print("=" * 74)


def filter_isolated_oscillations(
    x: pd.Series,
    y: pd.Series,
    start_x: float = 4.5,
    absolute_tolerance: float = 0.003,
    scale_factor: float = 4.0,
    max_passes: int = 4,
) -> tuple[pd.Series, pd.Series]:
    """Mask isolated numerical spikes without modifying the raw data.

    A candidate is compared with a leave-one-out PCHIP estimate formed from
    two valid neighbours on each side.  It is rejected only when it is an
    isolated local overshoot/undershoot and its residual exceeds both the
    absolute tolerance and the scaled normal local variation.
    """
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    clean = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float).copy()
    rejected = np.zeros(clean.size, dtype=bool)

    if max_passes < 1:
        return (
            pd.Series(clean, index=y.index, dtype=float),
            pd.Series(rejected, index=y.index, dtype=bool),
        )

    order = np.argsort(x_values, kind="mergesort")

    for _ in range(max_passes):
        valid_order = order[
            np.isfinite(x_values[order]) & np.isfinite(clean[order])
        ]
        reject_this_pass: list[int] = []

        for position in range(2, len(valid_order) - 2):
            index = int(valid_order[position])
            if x_values[index] < start_x:
                continue

            left_1 = int(valid_order[position - 1])
            right_1 = int(valid_order[position + 1])
            is_local_extreme = (
                (clean[index] - clean[left_1])
                * (clean[index] - clean[right_1])
                > 0.0
            )
            if not is_local_extreme:
                continue

            neighbour_indices = np.array(
                [
                    valid_order[position - 2],
                    valid_order[position - 1],
                    valid_order[position + 1],
                    valid_order[position + 2],
                ],
                dtype=int,
            )
            neighbour_x = x_values[neighbour_indices]
            neighbour_y = clean[neighbour_indices]

            if (
                np.unique(neighbour_x).size != neighbour_x.size
                or not np.all(np.diff(neighbour_x) > 0.0)
            ):
                continue

            prediction = float(
                PchipInterpolator(neighbour_x, neighbour_y)(x_values[index])
            )
            residual = abs(clean[index] - prediction)
            local_variation = float(np.median(np.abs(np.diff(neighbour_y))))
            threshold = max(
                absolute_tolerance,
                scale_factor * max(local_variation, 1.0e-12),
            )

            if residual > threshold:
                reject_this_pass.append(index)

        if not reject_this_pass:
            break

        clean[reject_this_pass] = np.nan
        rejected[reject_this_pass] = True

    return (
        pd.Series(clean, index=y.index, dtype=float),
        pd.Series(rejected, index=y.index, dtype=bool),
    )


def interpolate_and_smooth_long_range(
    x: pd.Series,
    y: pd.Series,
    start_x: float,
    window_length: int = 11,
    polynomial_order: int = 2,
) -> pd.Series:
    """Fill rejected spikes and smooth only the long-range plotted tail."""
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float).copy()
    output = values.copy()

    order = np.argsort(x_values, kind="mergesort")
    long_order = order[
        np.isfinite(x_values[order]) & (x_values[order] >= start_x)
    ]
    if long_order.size < polynomial_order + 3:
        return pd.Series(output, index=y.index, dtype=float)

    long_x = x_values[long_order]
    long_y = values[long_order]
    finite = np.isfinite(long_y)
    if finite.sum() < polynomial_order + 3:
        return pd.Series(output, index=y.index, dtype=float)

    interpolator = PchipInterpolator(
        long_x[finite],
        long_y[finite],
        extrapolate=False,
    )
    filled = long_y.copy()
    missing = ~finite
    filled[missing] = interpolator(long_x[missing])

    usable = np.isfinite(filled)
    if usable.sum() < polynomial_order + 3:
        return pd.Series(output, index=y.index, dtype=float)

    usable_positions = np.flatnonzero(usable)
    first = int(usable_positions[0])
    last = int(usable_positions[-1]) + 1
    block = filled[first:last]

    window = max(int(window_length), polynomial_order + 2)
    if window % 2 == 0:
        window += 1
    if window > block.size:
        window = block.size if block.size % 2 == 1 else block.size - 1
    if window <= polynomial_order:
        output[long_order] = filled
        return pd.Series(output, index=y.index, dtype=float)

    filled[first:last] = savgol_filter(
        block,
        window_length=window,
        polyorder=polynomial_order,
        mode="interp",
    )
    output[long_order] = filled
    return pd.Series(output, index=y.index, dtype=float)


def add_clean_plot_columns(
    df: pd.DataFrame,
    curves: dict[str, str],
    start_x: float,
    absolute_tolerance: float,
    scale_factor: float,
    max_passes: int,
    smoothing_window: int,
) -> dict[str, str]:
    """Create plotting-only cleaned columns while retaining raw columns."""
    clean_curves = dict(curves)
    filter_labels = ("SPL", "genISI2", "SCE", "SCE+rmZPE")

    print("\nLong-range plotting-artifact filtering:")
    for label in filter_labels:
        column = curves.get(label)
        if column is None or column not in df.columns:
            continue

        clean_column = f"{column}_plot_clean"
        mask_column = f"{column}_artifact_mask"
        clean_series, artifact_mask = filter_isolated_oscillations(
            df["R_plot"],
            df[column],
            start_x=start_x,
            absolute_tolerance=absolute_tolerance,
            scale_factor=scale_factor,
            max_passes=max_passes,
        )
        df[clean_column] = interpolate_and_smooth_long_range(
            df["R_plot"],
            clean_series,
            start_x=start_x,
            window_length=smoothing_window,
            polynomial_order=2,
        )
        df[mask_column] = artifact_mask
        clean_curves[label] = clean_column

        removed_r = df.loc[artifact_mask, "R_plot"].to_numpy(dtype=float)
        removed_text = (
            ", ".join(f"{value:.4f}" for value in removed_r)
            if removed_r.size
            else "none"
        )
        print(
            f"  {label:11s}: removed {removed_r.size:2d} point(s); "
            f"R = {removed_text}"
        )

    return clean_curves


def main():
    args = parse_args()
    configure_publication_style()

    df = load_data(args.input, args.input_unit, args.x_unit)
    curves = make_curve_map(args.include_mp2, args.include_stored_ac)

    for column in curves.values():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            print(f"Warning: column '{column}' was not found.")

    if not args.no_oscillation_filter:
        curves = add_clean_plot_columns(
            df,
            curves,
            start_x=args.oscillation_start,
            absolute_tolerance=args.oscillation_abs_tol,
            scale_factor=args.oscillation_scale,
            max_passes=args.oscillation_passes,
            smoothing_window=args.oscillation_window,
        )

    print_diagnostic(df, args.diagnostic_r, args.input_unit)

    if args.save_reconstructed_csv:
        csv_output = Path(args.save_reconstructed_csv)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_output, index=False)
        print(f"Saved reconstructed data: {csv_output}")

    df_filtered = df[
        (df["R_plot"] >= args.x_min) & (df["R_plot"] <= args.x_max)
    ].copy()

    curve_columns = [
        column for column in curves.values() if column in df_filtered.columns
    ]
    df_filtered = df_filtered.dropna(subset=curve_columns, how="all")

    print("\nFirst rows used for cylindrical SPL/genISI2 plotting:")
    diagnostic_columns = [
        "R_input",
        "R_plot",
        "E_ref",
        "winf_cyl",
        "winf_prime_rmzpe_cyl",
        "E_spl_cyl",
        "E_gisi2_cyl",
        "E_acm_exact",
    ]
    diagnostic_columns = [
        column for column in diagnostic_columns if column in df_filtered.columns
    ]
    print(df_filtered[diagnostic_columns].head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 10))

    plotted = []
    for label, column in curves.items():
        if column not in df_filtered.columns:
            continue
        valid = df_filtered[["R_plot", column]].dropna()
        if valid.empty:
            print(f"Warning: no finite data for {label} ({column}).")
            continue

        ax.plot(
            valid["R_plot"],
            valid[column],
            label=label,
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.45,
            markerfacecolor="white" if label == "Exact ACM" else None,
            markevery=3,
            alpha=1.0,
            **STYLE[label],
        )
        plotted.append(label)

    unit_label = r"\mathrm{\AA}" if args.x_unit == "Angstrom" else r"a_0"
    ax.set_xlabel(rf"$R$")
    ax.set_ylabel(r"$E$ ")

    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.47, 0.040),
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        handlelength=1.8,
        columnspacing=0.8,
        labelspacing=0.32,
        borderpad=0.50,
    )
    legend.get_frame().set_linewidth(1.1)

    plt.tight_layout()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    eps_path = output_prefix.with_suffix(".eps")
    png_path = output_prefix.with_suffix(".png")

    fig.savefig(eps_path, format="eps")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"\nPlotted curves: {', '.join(plotted)}")
    print(f"Saved: {eps_path}")
    print(f"Saved: {png_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()


