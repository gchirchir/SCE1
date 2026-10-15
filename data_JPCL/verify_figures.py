#!/usr/bin/env python3
"""Compare every regenerated figure with the copy used in the manuscript.

Each EPS produced by run_all.sh (in the system folders) is rasterized with
Ghostscript at 100 dpi, together with the matching file in reference_figures/
(the exact EPS files included in main.tex / SI.tex), and the two images are
compared pixel by pixel. Exit status is 0 only if every figure is identical.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent

# manuscript figure -> (folder that regenerates it, where it appears)
FIGURES = {
    "Fig_MARE_Hook_FCI_PYOEP_RED_GREEN_R2SCAN.eps": ("hookium", "main text, Figure 1"),
    "Hook_MARE_SPL_genISI2_PYOEP.eps": ("hookium", "main text, Figure 2"),
    "HOOK_FCI_PYOEP_SPL_genISI2_PCME_combined.eps": ("hookium", "main text, Figure 3"),
    "HE_SERIES_1_PYOEP.eps": ("he_series", "main text, Figure 4"),
    "He_series_MARE_SPL_genISI2_PYOEP.eps": ("he_series", "main text, Figure 5"),
    "He_iso_FCI_PYOEP_genISI2_SPL_PCME_combined.eps": ("he_series", "main text, Figure 6"),
    "Fig_H2_total_energy_journal_cylAC_PYOEP.eps": ("h2_dissociation", "main text, Figure 7"),
    "H2_delta_Winf_all_models_split.eps": ("h2_dissociation", "main text, Figure 8"),
    "H2_Wprime_inf_enhanced.eps": ("h2_dissociation", "main text, Figure 9"),
    "neutral_1_PYOEP.eps": ("atoms_ions", "main text, Figure 10a"),
    "neutral_MARE_SPL_genISI2_PYOEP.eps": ("atoms_ions", "main text, Figure 10b"),
    "toc_jpcl.eps": ("toc_graphic", "TOC/abstract graphic"),
    "Hook_delta_ingredients_PYOEP.eps": ("hookium", "SI, Figure S1"),
    "He_series_delta_ingredients_PYOEP.eps": ("he_series", "SI, Figure S2"),
    "atoms_model_errors_PYOEP.eps": ("atoms_ions", "SI, Figure S3"),
    "H2_models_AC_MAE_PYOEP.eps": ("h2_dissociation", "SI, Figure S4"),
    "HOOK_FCI_PYOEP_SPL_genISI2_PCME_ACME_MAE.eps": ("hookium", "SI, Figure S5"),
    "He_iso_FCI_PYOEP_SPL_genISI2_PCME_ACME_MAE.eps": ("he_series", "SI, Figure S6"),
}


def rasterize(eps: Path, png: Path) -> Image.Image:
    subprocess.run(
        ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop",
         "-sDEVICE=png16m", "-r100", f"-sOutputFile={png}", str(eps)],
        check=True,
    )
    return Image.open(png).convert("RGB")


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, (folder, where) in FIGURES.items():
            new = HERE / folder / name
            ref = HERE / "reference_figures" / name
            if not new.exists():
                print(f"MISSING    {name:48s} (run run_all.sh first)")
                failures += 1
                continue
            a = rasterize(ref, tmp / f"ref_{name}.png")
            b = rasterize(new, tmp / f"new_{name}.png")
            if a.size != b.size:
                print(f"SIZE DIFF  {name:48s} {a.size} vs {b.size}")
                failures += 1
                continue
            diff = np.asarray(ImageChops.difference(a, b)).max(axis=2)
            if diff.max() == 0:
                print(f"IDENTICAL  {name:48s} {where}")
            else:
                frac = 100 * (diff > 0).mean()
                print(f"DIFFERENT  {name:48s} {where}: {frac:.3f}% of pixels")
                failures += 1
    print(f"\n{len(FIGURES) - failures}/{len(FIGURES)} figures reproduce the manuscript exactly.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
