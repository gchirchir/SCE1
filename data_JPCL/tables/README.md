# Sources of the values in Table 1 and SI Tables S4–S5

`table_sources.csv` lists every numerical value printed in manuscript Table 1 and
Supporting Information Tables S4–S5 (90 values). Tables S4 and S5 (the Ne
basis-set/correlation-level tables) were Tables 2 and 3 of the main text in the
first submission; they were moved to SI Section S13 in revision 1. For each one it gives the source file in this package, the line
number, and the source line itself. Run

```bash
python3 check_tables.py
```

to confirm that each sourced value appears (rounded to 6 decimals) on the stated
line. Current result: **70 values verified, 0 mismatches, 20 values with no
archived raw output** (listed below).

| Folder | Contents | Table values sourced |
|---|---|---|
| `table1_atoms/` | `ATOMS11.csv` (compiled per-system summary of W∞/W′∞ with basis and literature values); `merged_all_results11.csv` (full result rows, incl. Xe) | Table 1: 40 of 42 values |
| `tableS4_ne_ccsdt_runs/` | Ne CCSD(T) runs at 2–5ZaPa-NR-CV: `star.py`, SLURM scripts, output logs (`slurm-*.out`, e.g. `W_inf_SCE : -20.0501746823`), `batch_summary_pc4.csv`, and the CCSD(T) density matrices (`dm_ccsdt.npy`, `.npz`, `.json`) | Table S4 (formerly 2): the 8 CCSD(T) values |
| `tableS5_ne_pyoep/` | Excerpts of the cluster output for the PyOEP-inverted Ne densities (HF, CCSD, CCSD(T) targets), as pasted by the author into working sessions on 2026-09-09 and 2026-09-11 | Table S5 (formerly 3): 22 of 24 values |

Cluster usernames, host names and home-directory paths have been replaced by
`<user>` / `<cluster>` in the logs and scripts of `tableS4_ne_ccsdt_runs/` and
`tableS5_ne_pyoep/`; nothing else was changed.

## Values without archived raw output

The following 20 values are reported in the manuscript, but the raw program
output that produced them is not in this package (it was not found in the
authors' local archive and presumably remains on the computing cluster):

- **Table 1:** Zn, W∞ and W′∞ (CCSD density, aug-cc-pwCV5Z).
- **Table S4:** the HF and CCSD rows at all four bases (16 values).
- **Table S5:** HF target density at 5ZaPa-NR-CV, W∞ and W′∞.
