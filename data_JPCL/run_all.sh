#!/usr/bin/env bash
# Regenerate every figure of the manuscript and its Supporting Information from
# the data in this folder, then check each one against the manuscript copy in
# reference_figures/.  Usage:  ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg

run() {
  local dir="$1"; shift
  for script in "$@"; do
    printf '%-18s %-48s ' "$dir" "$script"
    if ( cd "$dir" && python3 "$script" >/dev/null 2>&1 ); then
      echo ok
    else
      echo FAILED; exit 1
    fi
  done
}

run hookium         plot_hook_mare_wy_pyoep.py plot_hook_mare_spl_genisi2_wy_pyoep.py \
                    plot_hook_wy_pyoep.py plot_hook_delta_ingredients.py plot_hook_mae_pcme_acme.py
run he_series       plot_he_series_winf_mare_wy_pyoep.py plot_he_series_mare_spl_genisi2_wy_pyoep.py \
                    plot_oep_wy.py plot_he_delta_ingredients.py plot_he_mae_pcme_acme.py
run atoms_ions      plot_he_series_winf_mare_wy_pyoep.py plot_he_series_mare_spl_genisi2_wy_pyoep.py \
                    plot_atoms_model_errors.py
run h2_dissociation plot_final_wy_pyoep.py 3.py 4.py plot_h2_models_ac.py
run toc_graphic     plot_toc_jpcl.py

echo
python3 verify_figures.py
