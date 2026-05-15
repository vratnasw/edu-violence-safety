"""Smoke + validation tests.

The smoke layer (test_modules_import) verifies every model module is
importable and exposes run(). The validation layer runs after the
orchestrator has produced results/violence_paper_summary.json and checks
that the panel coefficients are real floats and the RD bandwidth is > 0.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Smoke layer (Phase A holdover)
# --------------------------------------------------------------------------

def test_modules_import():
    from src.models import crime_absenteeism
    from src.models import safety_investment
    from src.models import violence_paper_assembly
    assert hasattr(crime_absenteeism, "run"), "crime_absenteeism.run missing"
    assert hasattr(safety_investment, "run"), "safety_investment.run missing"
    assert hasattr(violence_paper_assembly, "run"), "violence_paper_assembly.run missing"


# --------------------------------------------------------------------------
# Validation layer (runs against the JSON the orchestrator emits)
# --------------------------------------------------------------------------

SUMMARY = REPO / "results" / "violence_paper_summary.json"
CRIME = REPO / "results" / "crime_absenteeism" / "results.json"
SAFETY = REPO / "results" / "safety_investment" / "results.json"


def _need(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"orchestrator output missing — run scripts/run_violence.py first ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


def test_panel_coefs_are_floats():
    crime = _need(CRIME)
    coef = crime.get("crime_coef")
    se = crime.get("crime_se")
    spatial = crime.get("spatial_lag_coef")
    assert isinstance(coef, float), f"crime_coef should be float, got {type(coef).__name__}"
    assert isinstance(se, float), f"crime_se should be float, got {type(se).__name__}"
    assert isinstance(spatial, float), f"spatial_lag_coef should be float, got {type(spatial).__name__}"
    assert not math.isnan(coef), "crime_coef should be finite (NaN means estimator failed)"
    assert se > 0, "crime_se should be strictly positive"


def test_rd_bandwidth_positive():
    safety = _need(SAFETY)
    susp_bw = safety["suspension_rd"]["bandwidth"]
    abs_bw = safety["absenteeism_rd"]["bandwidth"]
    assert isinstance(susp_bw, float) and susp_bw > 0, \
        f"suspension RD bandwidth must be > 0, got {susp_bw}"
    assert isinstance(abs_bw, float) and abs_bw > 0, \
        f"absenteeism RD bandwidth must be > 0, got {abs_bw}"


def test_summary_has_required_keys():
    s = _need(SUMMARY)
    assert "crime_absenteeism" in s and "safety_investment" in s
    ca = s["crime_absenteeism"]
    for k in ("crime_coef", "crime_coef_per_10unit_rate_increase",
              "spatial_spillover_coef", "prop_mediated_injury",
              "treatment_source"):
        assert k in ca, f"missing crime_absenteeism.{k}"
    si = s["safety_investment"]
    for k in ("suspension_rd_estimate", "suspension_rd_ci",
              "absenteeism_rd_estimate", "absenteeism_rd_ci",
              "threshold_used"):
        assert k in si, f"missing safety_investment.{k}"
    assert isinstance(si["suspension_rd_ci"], list) and len(si["suspension_rd_ci"]) == 2
    assert isinstance(si["absenteeism_rd_ci"], list) and len(si["absenteeism_rd_ci"]) == 2
