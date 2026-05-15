# edu-violence-safety

**Status: Phase A scaffold.** Module stubs in place; implementation (Phase B) pending.

Violence + safety pathway models — crime->absenteeism, E-Rate fuzzy RD. Model 20 + components of 6/18.

## Research questions

- Does neighborhood violent crime causally raise chronic absenteeism via spatial spillovers?
- Does crossing the 75% FRPL E-Rate threshold cause measurable safety investment + downstream outcome changes?

## Models / modules

- `src/models/crime_absenteeism.py` -- Two-way FE panel + PySAL spatial spillover; FBI UCR fallback documented for years where CA DOJ is missing. (-> `results/crime/twoway_fe_spillover.json`)
- `src/models/safety_investment.py` -- E-Rate fuzzy RD using the 75% FRPL discount threshold. (-> `results/safety/fuzzy_rd.json`)
- `src/models/violence_paper_assembly.py` -- Headline assembler combining crime + E-Rate modules. (-> `results/violence_paper_summary.json`)

## Data sources

Master panel (canonical join, 10,100 rows x 537 cols) at R2 key `processed/joined/master_panel.parquet`. Repo-specific parquets are listed in `config/config.yaml` under `data:`.

## Target journal

Journal of Public Economics or Demography

## Quick start

```
pip install -r requirements.txt
cp .env.example .env   # fill in 5 R2 vars
python scripts/preflight.py        # GO/NO-GO on R2 + parquets
python scripts/run_violence.py --fast  # scaffold stub
```

## Layout

```
config/        # config.yaml + r2_client.py (canonical copy)
src/models/    # one stub .py per model in this paper
src/causal/    # (Phase B)
src/utils/     # config_loader.py
scripts/       # preflight.py + run_violence.py orchestrator
tests/         # smoke tests for stub parsing
notebooks/     # exploration.ipynb (empty)
figures/       # paper figures (Phase B)
results/       # JSON outputs (Phase B)
```

