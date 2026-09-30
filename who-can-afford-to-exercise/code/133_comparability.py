"""The comparability catalogue: one standard, nine filers, five different line items.

ASC 860-50 requires an issuer that acquires the unilateral right to repurchase a
delinquent loan to re-recognise it, "regardless of intention to repurchase". Every filer
below is recognising the same economic quantity under the same paragraph of the same
standard. They report it under five distinct captions with three different scopes, and
only four of the nine report the quantity the standard's logic actually isolates -- the
balance NOT repurchased.

Scope codes:
  A    all loans whose option has vested, gross of repurchases
  B    the repurchased portion only
  C    the unexercised portion (what the recognition rule uniquely reveals)
  A*   a superset of A, mixed with other agency-guaranteed balances

Every caption and figure here was read from the filing and is quoted in the source notes
of the manuscript; the extraction scripts are 89-95 and 100-101.
"""
import os
import pandas as pd
from config import PAPER as PAP, OUT

ROWS = [
    # filer, type, caption as printed, scope, form, 2019, 2020, units
    ("PennyMac Financial", "nonbank", "Loans eligible for repurchase", "C",
     "10-K, 10-Q", 0.89, 17.18, "quarterly"),
    ("Mr.\\ Cooper Group", "nonbank", "Loans subject to repurchase from Ginnie Mae", "C",
     "10-K", 0.56, 6.16, "annual"),
    ("Caliber Home Loans", "nonbank", "Loans eligible for repurchase from GNMA", "C",
     "S-1/A", 0.195, 1.918, "annual"),
    ("AmeriHome", "nonbank", "Loans eligible for repurchase", "C",
     "S-1/A", 0.383, 2.426, "annual"),
    ("Flagstar Bank", "depository", "Loans with government guarantees", "A",
     "10-K", 0.736, 2.516, "annual"),
    ("Flagstar Bank", "depository", "\\quad \\ldots\\ repurchase options", "C",
     "10-K", 0.070, 1.851, "annual"),
    ("Fifth Third Bank", "depository", "Footnote decomposition of GNMA options", "A, B, C",
     "10-K", float("nan"), 0.039, "annual"),
    ("JPMorgan Chase", "depository", "Loans repurchased or option to repurchase", "B+C",
     "10-K", 2.941, 1.413, "annual"),
    ("M\\&T Bank", "depository", "Loans repurchased from Ginnie Mae pools", "B",
     "10-K", 0.807, 2.700, "annual"),
    ("BOK Financial", "depository", "Residential mortgage guaranteed by U.S.\\ agencies",
     "A*", "10-K", 0.198, 0.409, "annual"),
]
d = pd.DataFrame(ROWS, columns=["filer", "type", "caption", "scope", "form",
                                "y2019", "y2020", "freq"])

print(f"{len(d)} line items across {d.filer.nunique()} filers")
print(f"distinct captions : {d.caption.nunique()}")
print(f"distinct scopes   : {sorted(d.scope.unique())}")
# "B+C" is a combined line: it contains the quantity but does not isolate it, which is
# exactly the failure this paper is about. Only a standalone C, or an explicit
# decomposition, lets a reader recover the unexercised balance.
ISOLATES = {"C", "A, B, C"}
net = d[d.scope.isin(ISOLATES)]
combined = d[~d.scope.isin(ISOLATES)]
print(f"isolate the unexercised balance : {net.filer.nunique()} of {d.filer.nunique()} filers")
print(f"report it only inside a wider aggregate : "
      f"{sorted(set(combined.filer) - set(net.filer))}")
print(f"report it at quarterly frequency        : {(d.freq == 'quarterly').sum()}")
print()
print(d[["filer", "scope", "caption", "y2019", "y2020"]].to_string(index=False))


def f(x):
    return "---" if pd.isna(x) else f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"


L = [r"\begin{tabular}{llllrr}", r"\toprule",
     r"& & & & \multicolumn{2}{c}{Reported (\$bn)}\\",
     r"\cmidrule(lr){5-6}",
     r"Filer & Scope & Line item as printed & Source & 2019 & 2020\\", r"\midrule"]
last = None
for _, r in d.iterrows():
    name = "" if r.filer == last else r.filer
    last = r.filer
    L.append(f"{name} & {r.scope} & {r.caption} & {r.form} & "
             f"{f(r.y2019)} & {f(r.y2020)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
out = os.path.join(PAP, "tables", "A1.tex")
open(out, "w", encoding="utf-8").write("\n".join(L))
d.to_csv(os.path.join(OUT, "comparability.csv"), index=False)

mac = {
    "cpFilers": str(d.filer.nunique()),
    "cpItems": str(len(d)),
    "cpCaptions": str(d.caption.nunique()),
    "cpScopes": str(len(set(d.scope))),
    "cpNet": str(net.filer.nunique()),
    "cpQuarterly": str(int((d.freq == "quarterly").sum())),
}
# Rewrite in place. The append-if-absent pattern used elsewhere silently keeps a stale
# value, because \newcommand cannot redefine -- which is how an earlier count survived a
# correction and reached a compiled draft.
import re

pth = os.path.join(PAP, "numbers_pool.tex")
txt = open(pth, encoding="utf-8").read()
new = []
for k, v in sorted(mac.items()):
    pat = re.compile(r"(\\newcommand\{\\p" + k + r"\}\{)[^}]*(\})")
    if pat.search(txt):
        was = pat.search(txt).group(0).split("{")[-1].rstrip("}")
        if was != v:
            print(f"  rewrote \\p{k}: {was} -> {v}")
        txt = pat.sub(lambda m: m.group(1) + v + m.group(2), txt)
    else:
        new.append("\\newcommand{\\p" + k + "}{" + v + "}")
if new:
    txt = txt.rstrip() + "\n" + "\n".join(new) + "\n"
open(pth, "w", encoding="utf-8").write(txt)
print(f"\nwrote tables/A1.tex; {len(mac)} macros ({len(new)} new)")
