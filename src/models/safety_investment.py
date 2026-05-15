"""FRPL-share fuzzy RD as an E-Rate Category 1 proxy.

The pure E-Rate IV — eligibility for Category 1 priority funding — sits at
75% free-and-reduced-price-meal share (FRPL). E-Rate disbursement data is not
yet on R2, so we estimate the reduced-form discontinuity directly: for each
outcome, fit a local-linear regression with triangular kernel on
`frpl_share = pct_frl / enrollment` around the 0.75 cutoff. The threshold
itself has substantive content because it doubles as the federal Title-I
high-poverty threshold, but the estimated coefficient should be read as
"FRPL-threshold-RD," not pure E-Rate effect.

Outputs `results/safety_investment/results.json`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CUTOFF = 0.75  # FRPL share threshold ~= E-Rate Category 1 + Title I high-poverty
OUTCOMES = ("suspension_rate_pct", "chronic_absenteeism_rate")

# rdrobust is in requirements but we keep a hand-rolled fallback for
# Windows / install-failure resilience.
try:
    from rdrobust import rdrobust as _rdrobust
    HAVE_RDROBUST = True
except Exception:  # noqa: BLE001
    HAVE_RDROBUST = False
    log.warning("rdrobust unavailable — falling back to hand-rolled local-linear RD")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _load_panel() -> pd.DataFrame:
    """Pull the master panel and compute frpl_share from counts."""
    import sys
    sys.path.insert(0, str(REPO / "config"))
    import r2_client  # type: ignore
    df = r2_client.download("processed/joined/master_panel.parquet")
    df = df.copy()
    # `pct_frl` in the master panel is the FRPL student COUNT (max ~500k);
    # the share is FRPL / enrollment.
    for c in ("pct_frl", "enrollment", *OUTCOMES):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    m = df["enrollment"].fillna(0) > 0
    df["frpl_share"] = np.where(m, df["pct_frl"] / df["enrollment"], np.nan)
    # Clamp pathological values >1 (occasional roll-up errors in the source)
    df.loc[df["frpl_share"] > 1.0, "frpl_share"] = np.nan
    return df


# --------------------------------------------------------------------------
# Imbens-Kalyanaraman bandwidth (plug-in)
# --------------------------------------------------------------------------

def _ik_bandwidth(y: np.ndarray, x: np.ndarray, c: float) -> float:
    """Imbens-Kalyanaraman 2012 plug-in optimal bandwidth.

    Two-stage: (1) pilot bandwidth from Silverman's rule of thumb,
    (2) fit local quadratic on each side of cutoff, plug variance and
    second-derivative estimates into IK formula. If anything fails (e.g.
    too few obs), falls back to Silverman.
    """
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    n = len(x)
    if n < 50:
        return 0.1  # safe wide-ish default
    sigma_x = float(np.std(x, ddof=1))
    # Silverman pilot
    h_pilot = 1.84 * sigma_x * n ** (-0.2)
    h_pilot = max(h_pilot, 0.02)

    in_pilot = np.abs(x - c) <= h_pilot
    if in_pilot.sum() < 20:
        return h_pilot

    # Estimate density at cutoff
    f_c = float(in_pilot.sum() / (2.0 * h_pilot * n))

    # Local quadratics on each side
    left_mask = (x < c) & in_pilot
    right_mask = (x >= c) & in_pilot
    if left_mask.sum() < 10 or right_mask.sum() < 10:
        return h_pilot
    p_left = np.polyfit(x[left_mask] - c, y[left_mask], 2)
    p_right = np.polyfit(x[right_mask] - c, y[right_mask], 2)
    m2_left = 2.0 * p_left[0]  # second derivative
    m2_right = 2.0 * p_right[0]

    var_left = float(np.var(y[left_mask], ddof=1))
    var_right = float(np.var(y[right_mask], ddof=1))

    numer = (var_left + var_right) / max(f_c, 1e-6)
    denom = (m2_right - m2_left) ** 2 + 1e-8
    h_ik = 1.84 * (numer / denom) ** 0.2 * n ** (-0.2)
    # Constrain to a reasonable range so we don't blow up on flat outcomes
    return float(np.clip(h_ik, 0.02, 0.3))


# --------------------------------------------------------------------------
# Local-linear RD (fallback when rdrobust call fails)
# --------------------------------------------------------------------------

def _local_linear_rd(y: np.ndarray, x: np.ndarray, c: float = 0.0,
                     h: Optional[float] = None) -> dict:
    """Triangular-kernel local-linear RD with HC1-robust SEs."""
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    if h is None:
        h = _ik_bandwidth(y, x, c)
    in_band = np.abs(x - c) <= h
    yb, xb = y[in_band], x[in_band]
    n_left = int((xb < c).sum())
    n_right = int((xb >= c).sum())
    if min(n_left, n_right) < 10:
        return {
            "estimator": "local_linear_fallback",
            "estimate": float("nan"), "se": float("nan"),
            "ci_95_lo": float("nan"), "ci_95_hi": float("nan"),
            "bandwidth": float(h), "n_left": n_left, "n_right": n_right,
            "note": "insufficient obs in bandwidth",
        }
    w = np.maximum(1.0 - np.abs(xb - c) / h, 0.0)
    treat = (xb >= c).astype(float)
    z = xb - c
    X = np.column_stack([np.ones_like(z), treat, z, treat * z])
    Wsqrt = np.sqrt(w).reshape(-1, 1)
    beta, *_ = np.linalg.lstsq(X * Wsqrt, yb * np.sqrt(w), rcond=None)
    resid = yb - X @ beta
    XtWX_inv = np.linalg.pinv((X * w[:, None]).T @ X)
    sigma2 = float(np.sum(w * resid ** 2) / max(1, len(yb) - X.shape[1]))
    se_beta = np.sqrt(np.maximum(np.diag(XtWX_inv) * sigma2, 0.0))
    coef = float(beta[1])
    se = float(se_beta[1])
    return {
        "estimator": "local_linear_fallback",
        "estimate": coef, "se": se,
        "ci_95_lo": coef - 1.96 * se,
        "ci_95_hi": coef + 1.96 * se,
        "bandwidth": float(h),
        "n_left": n_left, "n_right": n_right,
    }


def _rdrobust_fit(y: np.ndarray, x: np.ndarray, c: float = CUTOFF,
                  h: Optional[float] = None) -> dict:
    """rdrobust wrapper with graceful fallback."""
    if not HAVE_RDROBUST:
        return _local_linear_rd(y, x, c=c, h=h)
    kwargs = dict(y=y, x=x, c=c, kernel="triangular", p=1)
    if h is not None:
        kwargs["h"] = h
    try:
        res = _rdrobust(**kwargs)
    except (ZeroDivisionError, ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        log.warning("rdrobust failed (%s); using local-linear fallback",
                    type(e).__name__)
        return _local_linear_rd(y, x, c=c, h=h)
    coef = float(res.coef.iloc[0, 0]) if hasattr(res, "coef") else float("nan")
    se = float(res.se.iloc[0, 0]) if hasattr(res, "se") else float("nan")
    ci_lo = float(res.ci.iloc[0, 0]) if hasattr(res, "ci") else float("nan")
    ci_hi = float(res.ci.iloc[0, 1]) if hasattr(res, "ci") else float("nan")
    pv = float(res.pv.iloc[0, 0]) if hasattr(res, "pv") else float("nan")
    bw = float(res.bws.iloc[0, 0]) if hasattr(res, "bws") else float("nan")
    n_l = float(res.N_h[0]) if getattr(res, "N_h", None) is not None else float("nan")
    n_r = float(res.N_h[1]) if getattr(res, "N_h", None) is not None else float("nan")
    return {
        "estimator": "rdrobust",
        "estimate": coef, "se": se, "p_value": pv,
        "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
        "bandwidth": bw,
        "n_left": int(n_l) if not np.isnan(n_l) else None,
        "n_right": int(n_r) if not np.isnan(n_r) else None,
    }


# --------------------------------------------------------------------------
# Outcome runner: IK-optimal h + h/2, 2h robustness
# --------------------------------------------------------------------------

def _run_outcome(panel: pd.DataFrame, outcome: str) -> dict:
    """Run primary (IK-optimal) + robustness (0.5h, 2h) RD for one outcome."""
    sub = panel.dropna(subset=[outcome, "frpl_share"]).copy()
    x = sub["frpl_share"].to_numpy(dtype=float)
    y = sub[outcome].to_numpy(dtype=float)
    primary = _rdrobust_fit(y, x, c=CUTOFF)
    primary["spec"] = "primary"

    h_primary = primary.get("bandwidth")
    robustness: list[dict] = []
    if h_primary is not None and not np.isnan(h_primary) and h_primary > 0:
        for mult, label in ((0.5, "half"), (2.0, "double")):
            r = _rdrobust_fit(y, x, c=CUTOFF, h=h_primary * mult)
            r["spec"] = "robustness"
            r["h_label"] = label
            r["h_mult"] = mult
            robustness.append(r)

    return {
        "outcome": outcome,
        "estimate": primary["estimate"],
        "ci_95": [primary["ci_95_lo"], primary["ci_95_hi"]],
        "se": primary["se"],
        "bandwidth": primary["bandwidth"],
        "n_left": primary.get("n_left"),
        "n_right": primary.get("n_right"),
        "primary": primary,
        "robustness": robustness,
    }


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

def run(fast: bool = False) -> dict:
    """Run the fuzzy-RD pipeline. `fast` is currently a no-op — these
    estimators all run in seconds — but is accepted for orchestrator
    compatibility."""
    log.info("safety_investment: loading panel + computing frpl_share")
    panel = _load_panel()

    suspension = _run_outcome(panel, "suspension_rate_pct")
    absenteeism = _run_outcome(panel, "chronic_absenteeism_rate")

    out = {
        "threshold_used": (
            "pct_frl/enrollment >= 0.75 (proxy for E-Rate Category 1 "
            "discount tier AND federal Title I high-poverty cutoff; the "
            "actual E-Rate disbursement panel is not yet on R2)"),
        "running_variable": "frpl_share = pct_frl / enrollment",
        "cutoff": CUTOFF,
        "kernel": "triangular",
        "polynomial_order": 1,
        "estimator": "rdrobust" if HAVE_RDROBUST else "local_linear_fallback",
        "suspension_rd": {
            "estimate": suspension["estimate"],
            "ci": suspension["ci_95"],
            "bandwidth": suspension["bandwidth"],
            "n_left": suspension["n_left"],
            "n_right": suspension["n_right"],
        },
        "absenteeism_rd": {
            "estimate": absenteeism["estimate"],
            "ci": absenteeism["ci_95"],
            "bandwidth": absenteeism["bandwidth"],
            "n_left": absenteeism["n_left"],
            "n_right": absenteeism["n_right"],
        },
        "suspension_full": suspension,
        "absenteeism_full": absenteeism,
    }

    out_dir = REPO / "results" / "safety_investment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float),
                        encoding="utf-8")
    log.info("safety_investment: wrote %s", out_path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    run()
