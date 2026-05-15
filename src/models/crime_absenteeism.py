"""Two-way FE panel + PySAL spatial spillover; FBI UCR fallback documented for years where CA DOJ is missing.

STATUS: Phase A scaffold. Implementation pending (Phase B).
Reads from:
  - processed/joined/master_panel.parquet
  - processed/canonical/ca_doj_crime.parquet
  - processed/canonical/fbi_ucr.parquet
Outputs: results/crime/twoway_fe_spillover.json
"""
from __future__ import annotations


def run(fast: bool = False) -> dict:
    raise NotImplementedError(
        "Phase A scaffold -- module not yet implemented. "
        "See README.md for the research question this module answers.")


if __name__ == "__main__":
    print("SCAFFOLD ONLY -- module not yet implemented")
