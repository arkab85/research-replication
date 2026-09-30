"""Paths and constants for the whole pipeline.

Every script imports from here, so a replicator sets the three directories below once
and nothing else changes. Previously each script carried an absolute Windows path, which
made the package unusable on another machine. Override any of these with an environment
variable of the same name.

  JMP_ROOT   the project directory containing code/, out/, paper/, figs/
  JMP_DATA   where the raw inputs live (the Ginnie Mae extract and issuer files)
  JMP_CACHE  a scratch directory for downloaded SEC filings

On first use:
    python code/00_patch_honestdid.py     # one-off upstream fix, see that file
    python code/02_build.py               # builds out/panel.parquet from JMP_DATA
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("JMP_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("JMP_DATA", Path.home() / "Downloads"))
CACHE = Path(os.environ.get("JMP_CACHE", ROOT / "out" / "edgar_cache"))

OUT = ROOT / "out"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGS = ROOT / "figs"
for _d in (OUT, PAPER, TABLES, FIGS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# raw inputs
EXTRACT = DATA / "DK_analy_data_20210223.csv"
ISSUER_TYPES = DATA / "issuer_info_202011_with_type.csv"
ISSUER_NAMES = [DATA / n for n in ("issuer_info_202011.txt", "issrinfo_202112.txt",
                                   "issrinfo_202212.txt", "active_issuers_202212.txt")]

# a descriptive User-Agent is all data.sec.gov requires; set JMP_SEC_UA to your own
SEC_UA = os.environ.get("JMP_SEC_UA", "Academic research (replication) contact@example.edu")

# estimation window and the shock date used throughout
WINDOW = (201901, 202009)
SHOCK = 202003
APM_2007 = 202007          # Ginnie Mae APM 20-07 bites on buyouts from 1 July 2020

if __name__ == "__main__":
    print(f"ROOT   {ROOT}")
    print(f"DATA   {DATA}   extract present: {EXTRACT.exists()}")
    print(f"OUT    {OUT}")
    print(f"PAPER  {PAPER}")
    print(f"CACHE  {CACHE}")
