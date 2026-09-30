"""Paths for the empirical pipeline. Override with environment variables.

JMP_VESTING  folder with the vesting-level files built from Ginnie Mae's public MBS loan-level
             disclosures (00c_gnma_dlq_analysis_data_00{1,2}_twelve_months, as CSV or parquet)
JMP_PANEL    folder with the monthly histories of seriously delinquent loans (gnma_dlq_subsample_00{1,2})
JMP_ISSUERS  issuer file with charter types (issuer_info_202011_with_type.csv)
JMP_ISSUER_MONTHLY  issuer-month portfolio summary (gnma_issuer_monthly_summary.csv): loan counts, forbearance counts, advances
JMP_WORK     folder for loan-level intermediates; these are never distributed
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW = Path(os.environ.get("JMP_DATA", Path.home() / "Downloads"))
VESTING = Path(os.environ.get("JMP_VESTING", RAW))
PANEL = Path(os.environ.get("JMP_PANEL", RAW))
ISSUERS = Path(os.environ.get("JMP_ISSUERS", RAW / "issuer_info_202011_with_type.csv"))
WORK = Path(os.environ.get("JMP_WORK", ROOT / "work"))
OUT = Path(os.environ.get("JMP_OUT", ROOT / "data" / "empirics"))
FIGS = Path(os.environ.get("JMP_FIGS", ROOT / "figures"))
TABLES = Path(os.environ.get("JMP_TABLES", ROOT / "tables"))
PMMS = Path(os.environ.get("JMP_PMMS", ROOT / "data" / "pmms_monthly.csv"))   # Freddie Mac PMMS 30-year rate, monthly mean (FRED MORTGAGE30US)
STATE_UR = Path(os.environ.get("JMP_STATE_UR", ROOT / "data" / "state_ur.csv"))  # BLS state unemployment rates, SA, via FRED (XXUR)
ISSUER_MONTHLY = Path(os.environ.get("JMP_ISSUER_MONTHLY", RAW / "gnma_issuer_monthly_summary.csv"))  # issuer-month portfolio summary built from the disclosures
for _d in (WORK, OUT, FIGS, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

PERIODS = [("pre", 201903, 202002), ("freeze", 202003, 202006), ("release", 202007, 202012),
           ("recovery1", 202101, 202106), ("recovery2", 202107, 202112), ("ratehike", 202201, 202209)]
PERIOD_LABEL = {"pre": "Mar 2019--Feb 2020", "freeze": "Mar--Jun 2020", "release": "Jul--Dec 2020",
                "recovery1": "Jan--Jun 2021", "recovery2": "Jul--Dec 2021", "ratehike": "Jan--Sep 2022"}
WINDOW = 3   # exercise within this many months of vesting
