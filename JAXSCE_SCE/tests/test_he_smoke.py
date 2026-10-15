"""Quick smoke test: He from an HF density must reproduce the SCE values.

Run from the JAXSCE_SCE folder with   python -m pytest -q tests
(about 20-30 s on a CPU). Full production runs were done on the WCSS cluster.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_helium_hf_sce(tmp_path):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run(
        [sys.executable, str(ROOT / "drivers/atoms_ions/star.py"), "--atom", "He",
         "--basis", "aug-cc-pVTZ", "--density", "hf", "--N-grid", "257"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=900,
    ).stdout
    w_inf = float(re.search(r"W_inf\s+:\s+(-?\d+\.\d+)", out).group(1))
    wp_inf = float(re.search(r"T_ZP \(W'∞\)\s+:\s+(-?\d+\.\d+)", out).group(1))
    assert abs(w_inf - (-1.5008)) < 5e-3      # Seidl et al. / Mirtschink et al.
    assert abs(wp_inf - 0.621) < 5e-3         # Gori-Giorgi, Vignale, Seidl
