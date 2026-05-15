"""Assemble crime->absenteeism + FRPL-RD into the paper headline JSON.

Reads `results/crime_absenteeism/results.json` and
`results/safety_investment/results.json` produced by the two estimators and
emits `results/violence_paper_summary.json`. The summary is the file
downstream consumers (anchors, Layer 4) should pull from.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CRIME_PATH = REPO / "results" / "crime_absenteeism" / "results.json"
SAFETY_PATH = REPO / "results" / "safety_investment" / "results.json"
SUMMARY_PATH = REPO / "results" / "violence_paper_summary.json"


def _read_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"required input missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def run(fast: bool = False) -> dict:
    log.info("assembly: reading crime_absenteeism results from %s", CRIME_PATH)
    crime = _read_json(CRIME_PATH)
    log.info("assembly: reading safety_investment results from %s", SAFETY_PATH)
    safety = _read_json(SAFETY_PATH)

    # Per-10-unit rate-increase scaling makes the crime coefficient easier
    # to read alongside the underlying rate distribution (mean ~6, max ~24).
    crime_coef = float(crime.get("crime_coef", float("nan")))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "crime_absenteeism": {
            "primary_outcome": crime.get("primary_outcome"),
            "crime_coef": crime_coef,
            "crime_se": crime.get("crime_se"),
            "crime_p": crime.get("crime_p"),
            "crime_coef_per_10unit_rate_increase": crime_coef * 10.0,
            "spatial_spillover_coef": crime.get("spatial_lag_coef"),
            "spatial_spillover_se": crime.get("spatial_lag_se"),
            "prop_mediated_injury": crime.get("prop_mediated_injury"),
            "n_obs": crime.get("n_obs"),
            "n_districts": crime.get("n_districts"),
            "treatment_source": crime.get("treatment_source"),
            "secondary_outcome": crime.get("secondary_outcome"),
        },
        "safety_investment": {
            "suspension_rd_estimate": safety.get("suspension_rd", {}).get("estimate"),
            "suspension_rd_ci": safety.get("suspension_rd", {}).get("ci"),
            "suspension_rd_bandwidth": safety.get("suspension_rd", {}).get("bandwidth"),
            "absenteeism_rd_estimate": safety.get("absenteeism_rd", {}).get("estimate"),
            "absenteeism_rd_ci": safety.get("absenteeism_rd", {}).get("ci"),
            "absenteeism_rd_bandwidth": safety.get("absenteeism_rd", {}).get("bandwidth"),
            "threshold_used": safety.get("threshold_used"),
            "estimator": safety.get("estimator"),
        },
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float),
                            encoding="utf-8")
    log.info("assembly: wrote %s", SUMMARY_PATH)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    run()
