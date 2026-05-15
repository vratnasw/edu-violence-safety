"""Two-way fixed-effects panel + spatial spillover + injury mediation.

Estimates the effect of county-level juvenile arrest activity on district
chronic absenteeism (and suspension as secondary outcome). The CA DOJ
OpenJustice crime panel that was specified in the original design (Model 20)
is not yet available on R2, so we substitute the kidsdata juvenile-arrest-rate
series (`kidsdata_gang_arrest_rate_mean`, county-year, broadcast to districts
within each county). The reported coefficient is therefore the effect of
juvenile-arrest intensity, not full violent crime.

Three specifications are produced for each outcome:

  1. Baseline two-way FE:
        y_it = alpha_i + lambda_t + beta * crime_it + gamma' X_it + e_it
     where alpha_i is a district FE, lambda_t is a year FE, X are economic
     controls (county child-poverty, county unemployment). Standard errors are
     clustered at the county level.
  2. Spatial-spillover augmentation:
        y_it = ... + rho * (W * crime)_it
     The spatial lag uses the Layer-3 sparse weight matrix from
     `edu-spatial-rl/results/spatiotemporal/edge_index.npy`, row-normalized.
     Districts outside the 92-district spatial network have zero spillover.
  3. Mediation through injury hospitalization rate:
        crime -> kd_health_injury_injury_hospitalization_rate_mean -> y
     using `pingouin.mediation_analysis` with 5000 bootstrap replicates.

Outputs `results/crime_absenteeism/results.json`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Two outcomes; the first is the headline result, the second is reported as
# a secondary specification.
PRIMARY_OUTCOME = "chronic_absenteeism_rate"
SECONDARY_OUTCOME = "suspension_rate_pct"
TREATMENT = "kidsdata_gang_arrest_rate_mean"
MEDIATOR = "kd_health_injury_injury_hospitalization_rate_mean"
CONTROLS = ("census_saipe_child_poverty_rate", "bls_unemployment_rate")

REPO = Path(__file__).resolve().parents[2]
EDGE_INDEX_PATH = Path(
    "c:/Users/visha/Downloads/r_and_d/edu-spatial-rl/results/spatiotemporal/edge_index.npy"
)
EDGE_IDS_PATH = Path(
    "c:/Users/visha/Downloads/r_and_d/edu-spatial-rl/results/spatiotemporal/edge_ids.json"
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _load_panel() -> pd.DataFrame:
    """Pull the master panel from R2 (in-memory; no temp file)."""
    import sys
    sys.path.insert(0, str(REPO / "config"))
    import r2_client  # type: ignore
    df = r2_client.download("processed/joined/master_panel.parquet")
    df = df.copy()
    # Numeric coercion guards against the occasional string-as-numeric column
    for col in (PRIMARY_OUTCOME, SECONDARY_OUTCOME, TREATMENT, MEDIATOR, *CONTROLS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year_num"] = pd.to_numeric(df["year_num"], errors="coerce").astype("Int64")
    return df


def _load_spatial_weights() -> tuple[np.ndarray, list[str]]:
    """Load the Layer-3 sparse edge index and node id list."""
    if not EDGE_INDEX_PATH.exists():
        raise FileNotFoundError(f"edge_index.npy not at {EDGE_INDEX_PATH}")
    if not EDGE_IDS_PATH.exists():
        raise FileNotFoundError(f"edge_ids.json not at {EDGE_IDS_PATH}")
    ei = np.load(EDGE_INDEX_PATH)
    ids = json.loads(EDGE_IDS_PATH.read_text(encoding="utf-8"))
    return ei, ids


def _row_normalized_W(edge_index: np.ndarray, n: int) -> "csr_matrix":
    """Build a row-stochastic sparse weight matrix from edge_index (2 x m).

    edge_index[0] are sources, edge_index[1] are destinations; we treat the
    edges as already directed so that (W * x)[i] is the mean of x over i's
    out-neighbors. Self-loops, if any, are dropped first.
    """
    from scipy.sparse import csr_matrix

    src = edge_index[0]
    dst = edge_index[1]
    mask = src != dst
    src, dst = src[mask], dst[mask]
    data = np.ones_like(src, dtype=float)
    W = csr_matrix((data, (src, dst)), shape=(n, n))
    # Row-normalize
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = 1.0 / row_sums
    from scipy.sparse import diags
    W = diags(inv) @ W
    return W


def _spatial_lag_column(panel: pd.DataFrame, treatment_col: str,
                        edge_index: np.ndarray, ids: list[str]) -> pd.Series:
    """For each district-year row, compute the row-normalized weighted average
    of `treatment_col` across the district's spatial neighbors in the same year.

    Districts that are not in the 92-node spatial network get a zero spatial
    lag — they contribute no spillover information.
    """
    W = _row_normalized_W(edge_index, n=len(ids))
    id_to_idx = {cds: i for i, cds in enumerate(ids)}
    out = np.zeros(len(panel), dtype=float)
    panel_idx = panel.index.to_numpy()
    panel_cds = panel["cds"].to_numpy()
    panel_year = panel["year_num"].to_numpy()
    treat = panel[treatment_col].to_numpy(dtype=float)
    network_mask = np.array([c in id_to_idx for c in panel_cds])

    # Build x_t vectors per year over the 92 network districts and apply W.
    for year in np.unique(panel_year[~pd.isna(panel_year)]):
        year_mask = panel_year == year
        x = np.zeros(len(ids))
        count = np.zeros(len(ids))
        rows = np.where(year_mask & network_mask)[0]
        for r in rows:
            i = id_to_idx[panel_cds[r]]
            v = treat[r]
            if not np.isnan(v):
                x[i] += v
                count[i] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(count > 0, x / np.maximum(count, 1), 0.0)
        wx = np.asarray(W @ x).ravel()
        for r in rows:
            out[r] = wx[id_to_idx[panel_cds[r]]]
    # Non-network rows are 0 by initialization
    return pd.Series(out, index=panel.index, name=f"{treatment_col}_spatial_lag")


# --------------------------------------------------------------------------
# Two-way FE panel
# --------------------------------------------------------------------------

def _fit_two_way_fe(panel: pd.DataFrame, outcome: str,
                    extra_regressors: Optional[list[str]] = None) -> dict:
    """Fit y_it = alpha_i + lambda_t + beta * T_it + gamma'X_it + e_it.

    Returns a dict with the coefficient + clustered SE on the treatment and any
    extra regressors. Uses linearmodels.PanelOLS with entity and time effects.
    """
    from linearmodels.panel import PanelOLS

    extra_regressors = list(extra_regressors or [])
    cols_needed = [outcome, TREATMENT, *CONTROLS, *extra_regressors,
                   "cds", "year_num", "county_code"]
    sub = panel.dropna(subset=cols_needed).copy()
    sub = sub.set_index(["cds", "year_num"])
    y = sub[outcome]
    X = sub[[TREATMENT, *CONTROLS, *extra_regressors]]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True,
                   drop_absorbed=True, check_rank=False)
    # County-level cluster: use cluster_entity=False and pass the cluster series
    cluster = sub["county_code"].astype(str).astype("category")
    res = mod.fit(cov_type="clustered", clusters=cluster)
    params = res.params.to_dict()
    se = res.std_errors.to_dict()
    pvals = res.pvalues.to_dict()
    out: dict = {
        "outcome": outcome,
        "n_obs": int(res.nobs),
        "n_districts": int(sub.index.get_level_values(0).nunique()),
        "n_years": int(sub.index.get_level_values(1).nunique()),
        "rsq_within": float(res.rsquared_within)
            if hasattr(res, "rsquared_within") else float("nan"),
        "treatment_coef": float(params.get(TREATMENT, float("nan"))),
        "treatment_se": float(se.get(TREATMENT, float("nan"))),
        "treatment_p": float(pvals.get(TREATMENT, float("nan"))),
        "controls": {
            c: {
                "coef": float(params.get(c, float("nan"))),
                "se": float(se.get(c, float("nan"))),
                "p": float(pvals.get(c, float("nan"))),
            } for c in CONTROLS
        },
    }
    for reg in extra_regressors:
        out[f"{reg}_coef"] = float(params.get(reg, float("nan")))
        out[f"{reg}_se"] = float(se.get(reg, float("nan")))
        out[f"{reg}_p"] = float(pvals.get(reg, float("nan")))
    return out


# --------------------------------------------------------------------------
# Mediation analysis
# --------------------------------------------------------------------------

def _mediation_via_injury(panel: pd.DataFrame, outcome: str,
                          n_bootstrap: int = 5000) -> dict:
    """Bootstrapped mediation: TREATMENT -> MEDIATOR -> outcome.

    pingouin's mediation_analysis runs OLS for the X->M, M+X->Y, and X->Y
    paths and reports the indirect effect with bootstrap CI. We additionally
    report the proportion mediated.
    """
    import pingouin as pg

    cols = [outcome, TREATMENT, MEDIATOR]
    sub = panel.dropna(subset=cols).copy()
    if len(sub) < 100:
        return {"n_obs": int(len(sub)), "note": "insufficient observations"}
    try:
        med = pg.mediation_analysis(
            data=sub, x=TREATMENT, m=MEDIATOR, y=outcome,
            n_boot=int(n_bootstrap), seed=42,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("pingouin mediation failed (%s); skipping", e)
        return {"n_obs": int(len(sub)), "note": f"mediation failed: {e}"}

    # pingouin returns rows for Total, Direct, Indirect (path-name varies by version)
    out: dict = {"n_obs": int(len(sub)), "n_bootstrap": int(n_bootstrap)}
    # Be flexible with row labeling across versions
    paths = {str(p).lower(): row for p, row in
              med.set_index("path").iterrows() if isinstance(p, str)}
    direct_row = next((v for k, v in paths.items() if "direct" in k), None)
    indirect_row = next((v for k, v in paths.items() if "indirect" in k), None)
    total_row = next((v for k, v in paths.items() if "total" in k), None)

    def _coef(row, key="coef"):
        if row is None:
            return float("nan")
        for cand in (key, "Beta", "beta", "Estimate"):
            if cand in row.index:
                return float(row[cand])
        return float("nan")

    direct_eff = _coef(direct_row)
    indirect_eff = _coef(indirect_row)
    total_eff = _coef(total_row)
    if not np.isnan(total_eff) and total_eff != 0:
        prop_mediated = float(indirect_eff / total_eff)
    elif not (np.isnan(direct_eff) or np.isnan(indirect_eff)) and (direct_eff + indirect_eff) != 0:
        prop_mediated = float(indirect_eff / (direct_eff + indirect_eff))
    else:
        prop_mediated = float("nan")
    out.update({
        "direct_effect": direct_eff,
        "indirect_effect": indirect_eff,
        "total_effect": total_eff,
        "prop_mediated": prop_mediated,
    })
    # Try to also pull CI for the indirect effect, if present
    if indirect_row is not None:
        for lo_key, hi_key in (("CI[2.5%]", "CI[97.5%]"), ("ci_lo", "ci_hi"),
                                ("CI_low", "CI_high")):
            if lo_key in indirect_row.index and hi_key in indirect_row.index:
                out["indirect_ci_95"] = [
                    float(indirect_row[lo_key]), float(indirect_row[hi_key])
                ]
                break
    return out


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------

def run(fast: bool = False) -> dict:
    """Run the crime->absenteeism pipeline. `fast=True` uses fewer mediation
    bootstrap iterations to keep total wall-clock under ~2 minutes."""
    log.info("crime_absenteeism: loading master panel")
    panel = _load_panel()

    log.info("crime_absenteeism: loading spatial weights")
    edge_index, ids = _load_spatial_weights()
    panel["crime_spatial_lag"] = _spatial_lag_column(
        panel, treatment_col=TREATMENT, edge_index=edge_index, ids=ids
    )

    results: dict = {
        "treatment_source": (
            "kidsdata_gang_arrest_rate_mean (kidsdata juvenile-arrest rate "
            "per 1k, county-year, broadcast to all districts in the county) "
            "— substituted for the originally-spec'd CA DOJ OpenJustice "
            "violent-crime panel, which is not yet available on R2"),
        "mediator": MEDIATOR,
        "controls": list(CONTROLS),
        "spatial_weights_source": (
            "edu-spatial-rl/results/spatiotemporal/edge_index.npy "
            "(row-normalized sparse W over 92 districts)"),
    }

    # --- Primary outcome: chronic absenteeism ----------------------------
    log.info("crime_absenteeism: baseline FE (chronic absenteeism)")
    base = _fit_two_way_fe(panel, PRIMARY_OUTCOME)
    log.info("crime_absenteeism: spatial-lag FE (chronic absenteeism)")
    spatial = _fit_two_way_fe(panel, PRIMARY_OUTCOME,
                              extra_regressors=["crime_spatial_lag"])
    n_boot = 500 if fast else 5000
    log.info("crime_absenteeism: mediation (chronic absenteeism, %d boots)", n_boot)
    med = _mediation_via_injury(panel, PRIMARY_OUTCOME, n_bootstrap=n_boot)

    results["primary_outcome"] = PRIMARY_OUTCOME
    results["crime_coef"] = base["treatment_coef"]
    results["crime_se"] = base["treatment_se"]
    results["crime_p"] = base["treatment_p"]
    results["n_obs"] = base["n_obs"]
    results["n_districts"] = base["n_districts"]
    results["spatial_lag_coef"] = spatial.get("crime_spatial_lag_coef", float("nan"))
    results["spatial_lag_se"] = spatial.get("crime_spatial_lag_se", float("nan"))
    results["spatial_lag_p"] = spatial.get("crime_spatial_lag_p", float("nan"))
    results["prop_mediated_injury"] = med.get("prop_mediated", float("nan"))
    results["mediation_detail"] = med
    results["baseline_fe"] = base
    results["spatial_fe"] = spatial

    # --- Secondary outcome: suspension rate ------------------------------
    log.info("crime_absenteeism: baseline FE (suspension)")
    sec_base = _fit_two_way_fe(panel, SECONDARY_OUTCOME)
    sec_spatial = _fit_two_way_fe(panel, SECONDARY_OUTCOME,
                                  extra_regressors=["crime_spatial_lag"])
    results["secondary_outcome"] = {
        "name": SECONDARY_OUTCOME,
        "crime_coef": sec_base["treatment_coef"],
        "crime_se": sec_base["treatment_se"],
        "spatial_lag_coef": sec_spatial.get(
            "crime_spatial_lag_coef", float("nan")),
        "n_obs": sec_base["n_obs"],
    }

    # --- Persist ---------------------------------------------------------
    out_dir = REPO / "results" / "crime_absenteeism"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float),
                        encoding="utf-8")
    log.info("crime_absenteeism: wrote %s", out_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    run()
