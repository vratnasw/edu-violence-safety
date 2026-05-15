"""Orchestrator: runs crime->absenteeism, FRPL-RD, and the headline assembler.

Pass --fast to keep the mediation bootstrap to 500 draws (vs 5000) — the
estimators themselves are unaffected.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("violence-orchestrator")


def _load_dotenv() -> None:
    """Same search order as preflight.py."""
    candidates = [
        REPO / ".env",
        REPO.parent / ".env",
        REPO.parent / "edu-data-pipeline" / ".env",
    ]
    for c in candidates:
        if c.exists():
            for line in c.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
            log.info("loaded .env from %s", c)
            return
    log.info("no .env found (will rely on already-exported env vars)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="500-draw mediation bootstrap (vs default 5000)")
    args = ap.parse_args()
    _load_dotenv()

    from src.models import crime_absenteeism, safety_investment, violence_paper_assembly

    log.info("=== STEP 1/3: crime_absenteeism (FE + spatial + mediation) ===")
    crime_absenteeism.run(fast=args.fast)
    log.info("=== STEP 2/3: safety_investment (FRPL-share fuzzy RD) ===")
    safety_investment.run(fast=args.fast)
    log.info("=== STEP 3/3: violence_paper_assembly (headline JSON) ===")
    summary = violence_paper_assembly.run(fast=args.fast)
    log.info("=== DONE === wrote results/violence_paper_summary.json")
    log.info("crime_coef = %s, suspension_rd = %s, absenteeism_rd = %s",
             summary["crime_absenteeism"]["crime_coef"],
             summary["safety_investment"]["suspension_rd_estimate"],
             summary["safety_investment"]["absenteeism_rd_estimate"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
