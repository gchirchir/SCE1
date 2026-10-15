#!/usr/bin/env python3
"""Check that every sourced table value in table_sources.csv really appears,
after rounding to 6 decimals, on the stated line of the stated source file."""
import csv, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # data_JPCL/
num = re.compile(r"-?\d+\.\d{6,}")
ok = bad = missing = 0
for r in csv.DictReader(open(HERE / "table_sources.csv")):
    val = r["value_in_manuscript"].lstrip("-")
    if not r["source_file"]:
        missing += 1
        continue
    lines = open(ROOT / r["source_file"], errors="ignore").read().split("\n")
    line_no = int(r["source_line"])
    if r["source_file"].startswith("tables/tableS5_ne_pyoep/"):   # excerpts keep original numbering
        cand = [l for l in lines if l.startswith(f"{line_no:5d}:")]
        text = cand[0] if cand else ""
    else:
        text = lines[line_no - 1]
    if any(f"{abs(float(x)):.6f}" == val for x in num.findall(text)):
        ok += 1
    else:
        bad += 1
        print("MISMATCH:", r["table"], r["system"], r["basis_or_density"], r["quantity"], r["value_in_manuscript"])
print(f"{ok} values verified against their source line, {bad} mismatches, "
      f"{missing} values with no archived raw output.")
sys.exit(1 if bad else 0)
