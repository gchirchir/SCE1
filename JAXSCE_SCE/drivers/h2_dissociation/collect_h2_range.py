#!/usr/bin/env python3
"""
Walk R_*/dou_sig.csv (the sectioned per-point output of star.py) and collect
every completed point into one flat CSV, restricted to an R window.

Run this ON THE CLUSTER, from the directory that holds the R_* folders
(e.g. ~/pyscf/gpu/jaxsce/H2_strect/wyyy_H2_strect), then copy the resulting
h2_range_7_10.csv back for plotting:

    python3 collect_h2_range.py --rmin 7.0 --rmax 10.0 -o h2_range_7_10.csv
"""
import argparse
import glob
import io
import os
import re

import pandas as pd


def parse_sectioned_csv(path):
    """Split a star.py dou_sig.csv into its '# ===== ... =====' blocks and
    return {block_name: DataFrame}."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    blocks = {}
    parts = re.split(r"^# =====\s*(.+?)\s*=====\s*$", text, flags=re.MULTILINE)
    # parts[0] is the leading comment header before the first block (discard)
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1]
        lines = [ln for ln in body.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not lines:
            continue
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines)))
        except Exception as exc:
            print(f"  WARN: could not parse block '{name}' in {path}: {exc}")
            continue
        blocks[name] = df
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmin", type=float, default=float("-inf"))
    ap.add_argument("--rmax", type=float, default=float("inf"))
    ap.add_argument("--glob", default="R_*/dou_sig.csv")
    ap.add_argument("-o", "--output", default="h2_range_merged.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    print(f"Found {len(paths)} candidate dou_sig.csv files")

    rows = []
    skipped = 0
    for path in paths:
        blocks = parse_sectioned_csv(path)
        for name, df in blocks.items():
            if "R_bohr" not in df.columns:
                continue
            df = df.copy()
            df["source_block"] = name
            df["source_file"] = path
            rows.append(df)

    if not rows:
        print("No usable rows found — nothing to write.")
        return

    merged = pd.concat(rows, ignore_index=True, sort=False)
    merged["R_bohr"] = pd.to_numeric(merged["R_bohr"], errors="coerce")
    before = len(merged)
    merged = merged[(merged["R_bohr"] >= args.rmin) & (merged["R_bohr"] <= args.rmax)]
    merged = merged.sort_values(["R_bohr", "source_block"])
    merged.to_csv(args.output, index=False)
    print(f"Kept {len(merged)}/{before} rows in R in [{args.rmin}, {args.rmax}]")
    print(f"Wrote {args.output}")
    if "R_bohr" in merged.columns and len(merged):
        print("R values covered:", sorted(merged["R_bohr"].unique()))


if __name__ == "__main__":
    main()
