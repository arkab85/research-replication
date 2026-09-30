"""Find the MSR-holder table in the GMAR and show its real layout."""
import os, re
import pymupdf
from config import OUT

p = os.path.join(OUT, "gmar", "global_market_analysis_sep20.pdf")
doc = pymupdf.open(p)
print(f"{doc.page_count} pages")

hits = []
for i in range(doc.page_count):
    t = doc[i].get_text()
    if re.search(r"Servicing Rights|MSR", t):
        score = len(re.findall(r"\d{1,3},\d{3}", t))
        hits.append((i, score, t[:80].replace("\n", " ")))
for i, s, head in hits:
    print(f"  page {i+1:>3}  numeric-ish rows {s:>3}   {head}")

best = max(hits, key=lambda x: x[1])[0] if hits else None
if best is not None:
    print(f"\n--- full text of page {best+1} ---")
    print(doc[best].get_text()[:2600])
