"""Repair the two literal-backslash macro lines a shell quoting slip wrote, then apply
the two additions with the line breaks as they actually appear in the source."""
import os, json, re
from config import PAPER as PAP, OUT

# ---------------------------------------------------------------- repair
pth = os.path.join(PAP, "numbers_pool.tex")
s = open(pth, encoding="utf-8").read()
bad = re.findall(r"^\\\\newcommand.*$", s, re.M)
print(f"corrupted macro lines found: {len(bad)}")
for b in bad:
    print("   " + b)
s = re.sub(r"^\\\\newcommand\{\\\\(p[A-Za-z]+)\}\{([^}]*)\}\s*$",
           r"\\newcommand{\\\1}{\2}", s, flags=re.M)
open(pth, "w", encoding="utf-8").write(s)
print("repaired:")
for l in s.strip().split("\n")[-3:]:
    print("   " + l)

# ---------------------------------------------------------------- additions
p = os.path.join(PAP, "jmp.tex")
t = open(p, encoding="utf-8").read()
n = 0


def sub(old, new, label, count=1):
    global t, n
    c = t.count(old)
    assert c == count, f"[{label}] expected {count}, found {c}"
    t = t.replace(old, new)
    n += c
    print(f"  ok  {label}")


sub("""reversal. I have no way to rule that out, and the balance-sheet correlations in
Section~\\ref{sec:balance} are the only direct evidence on it.""",
    """reversal.

There is external evidence that the proxy measures the intended construct. Ginnie Mae
classifies every issuer into a peer group --- Mega, Large, Medium, Small or Very Small,
crossed with depository and non-depository --- and publishes the membership with issuer
identifiers. The classification is the agency's own, is built from the servicing portfolio
rather than from delinquency flow, and is maintained for supervisory purposes with no
reference to anything here. Across the \\pgmN{} issuers in the estimation sample that
appear in it, the rank correlation between the scale measure used in this paper and the
agency's own ordering is \\pgmRho. Within type, where the contrast is identified, it is
\\pgmRhoDep{} for depositories (\\pgmNDep{} issuers) and \\pgmRhoNb{} for nonbanks
(\\pgmNNb), and on the active sample the gradient is actually estimated on it is
\\pgmRhoActive{} (\\pgmNActive{} issuers). The mean of the scale measure is monotone in the
agency's ordering across every peer group it assigns. That does not eliminate measurement
error, but it does establish that the proxy and the regulator's own size classification
order these firms the same way.""",
    "10.3 external validation")

sub("""Seven issuers is not a panel, and they are the seven whose disclosure was specific
enough to use, which selects on being large and capital-markets-facing.""",
    """Seven issuers is not a panel, and they are the seven whose disclosure was specific
enough to use. The selection can be named rather than gestured at: the largest nonbank
issuer in the sample by vesting volume, and the third largest, are both absent, because
neither files periodic reports with the SEC. The filer set is therefore selected on being
publicly financed, which is plausibly correlated with the constraint this paper is about,
and in the direction that would understate it.""",
    "7.6 named selection limit")

open(p, "w", encoding="utf-8").write(t)
print(f"\n{n} edits applied")
