"""Smoke test: every model stub must import + expose run().

Phase A only verifies the scaffold parses; Phase B will add real tests.
"""
from __future__ import annotations


def test_modules_import():
    from src.models import crime_absenteeism
    from src.models import safety_investment
    from src.models import violence_paper_assembly
    assert hasattr(crime_absenteeism, 'run'), 'crime_absenteeism.run missing'
    assert hasattr(safety_investment, 'run'), 'safety_investment.run missing'
    assert hasattr(violence_paper_assembly, 'run'), 'violence_paper_assembly.run missing'
