#!/usr/bin/env bash
# Minimal example: exact SCE W_inf and W'_inf for the He atom from a
# Hartree-Fock density (aug-cc-pVTZ). Runs in about 20 s on a laptop CPU.
# Expected: W_inf ~ -1.4993 E_H, W'_inf ~ 0.6207 E_H
#           (literature: -1.5008 and 0.621).
set -e
cd "$(dirname "$0")/../drivers/atoms_ions"
JAX_PLATFORMS=cpu python star.py --atom He --basis aug-cc-pVTZ --density hf --N-grid 257
