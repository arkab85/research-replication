"""Borrower-level PII audit of everything that leaves this machine.

Firm names are expected and allowed: the issuers are public counterparties. What must
not appear is any borrower name, street address, loan identifier or account number.
"""
import re, os, glob
import pymupdf
from config import PAPER as PAP

SELF = "PII_AUDIT_PATTERNS"          # skip this file's own regexes

PATTERNS = {
    "SSN":            r"\b\d{3}-\d{2}-\d{4}\b",
    "street address": r"\b\d{1,6}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd|Ct|Court|Way|Pl|Place)\b",
    "ZIP+4":          r"\b\d{5}-\d{4}\b",
    "loan number":    r"\b(?:loan|acct|account)\s*(?:no\.?|number|#)\s*[:=]?\s*\d{6,}",
    "long digit run": r"\b\d{9,}\b",
    "email":          r"\b[\w.+-]+@(?!columbia\.edu)[\w.-]+\.[A-Za-z]{2,}\b",
    "phone":          r"\b\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b",
}

targets = sorted(glob.glob(os.path.join(PAP, "*.pdf")) +
                 glob.glob(os.path.join(PAP, "*.tex")) +
                 glob.glob(os.path.join(PAP, "tables", "*.tex")))
bad = 0
for f in targets:
    if f.lower().endswith(".pdf"):
        d = pymupdf.open(f)
        t = "".join(d[i].get_text() for i in range(d.page_count))
    else:
        t = open(f, encoding="utf-8", errors="ignore").read()
    hits = {}
    for name, rx in PATTERNS.items():
        m = [x.group(0) for x in re.finditer(rx, t)][:4]
        if m:
            hits[name] = m
    tag = "CLEAN" if not hits else "REVIEW"
    print(f"  {tag:<6} {os.path.relpath(f, PAP):<28} {len(t):>8,} chars"
          + ("" if not hits else "  " + "; ".join(f"{k}={v}" for k, v in hits.items())))
    bad += bool(hits)
print(f"\n{len(targets)} files, {bad} needing review")
