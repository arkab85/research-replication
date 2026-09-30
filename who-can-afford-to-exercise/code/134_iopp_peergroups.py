"""Ginnie Mae's own size classification, as an external check on the paper's size proxy.

The size-gradient test uses log 2019 option volume, a proxy the reviewer reports keep
objecting to. Ginnie Mae itself assigns every issuer to a peer group -- Mega, Large,
Medium, Small, Very Small, crossed with depository and non-depository -- and publishes the
membership lists with Issuer IDs and no scores. If the paper's continuous proxy lines up
with the agency's own discrete classification, that is external validation of the measure
from the regulator that defines it.

This fetches the peer-group file and checks the overlap with the paper's classification.
"""
import urllib.request, io, os, re, ssl
import pandas as pd
from config import OUT

UA = {"User-Agent": "Academic research (replication)"}
URLS = [
    "https://www.ginniemae.gov/issuers/issuer_tools/IOPP/iopp_peer_group_202101.pdf",
    "https://ginniemae.gov/issuers/issuer_tools/IOPP/iopp_peer_groups_202108.pdf",
]
ctx = ssl.create_default_context()

raw = None
for u in URLS:
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA),
                                    timeout=90, context=ctx) as r:
            b = r.read()
        print(f"  {u.rsplit('/', 1)[-1]}: {len(b):,} bytes, pdf={b[:4] == b'%PDF'}")
        if b[:4] == b"%PDF":
            raw = (u, b)
            break
    except Exception as e:
        print(f"  {u.rsplit('/', 1)[-1]}: {type(e).__name__} {str(e)[:70]}")

if raw is None:
    print("\ncould not retrieve a peer-group file; nothing further to check")
    raise SystemExit

url, b = raw
open(os.path.join(OUT, "iopp_peer_groups.pdf"), "wb").write(b)
import pymupdf
doc = pymupdf.open(stream=b, filetype="pdf")
txt = "\n".join(doc[i].get_text() for i in range(doc.page_count))
print(f"\n  {doc.page_count} pages, {len(txt):,} chars")

# peer group headings, then "ID  Name" rows beneath each
# the PDF puts the heading on one line, then the issuer id and its name on successive
# lines, interleaved with annotation lines that must be skipped
groups, cur = {}, None
HEAD = re.compile(r"^IOPP\s+(Single Family|Multifamily)\s+(.+?)\s+Peer Group\s*(.*)$", re.I)
SKIP = re.compile(r"^(Issuer ID|Issuer Name|Data as of|Note:|Issuers highlighted|"
                  r"Moved to|New IOPP)", re.I)
lines = [l.strip() for l in txt.split("\n")]
i = 0
while i < len(lines):
    s = lines[i]
    h = HEAD.match(s)
    if h:
        cur = (h.group(2) + " " + h.group(3)).strip()
        groups.setdefault(cur, [])
        i += 1
        continue
    if cur and re.fullmatch(r"\d{4}", s):
        j2 = i + 1
        while j2 < len(lines) and (not lines[j2] or SKIP.match(lines[j2])
                                   or re.fullmatch(r"\d{4}", lines[j2])):
            if re.fullmatch(r"\d{4}", lines[j2]):
                break
            j2 += 1
        name = lines[j2] if j2 < len(lines) and not re.fullmatch(r"\d{4}", lines[j2]) else ""
        groups[cur].append((int(s), name))
    i += 1

print("\n  peer groups parsed:")
for g, mem in groups.items():
    if mem:
        print(f"    {g:<44} {len(mem):>3} issuers")

allm = [(g, i, n) for g, mem in groups.items() for i, n in mem]
if not allm:
    print("\n  layout not parsed by the row pattern; dumping a sample for inspection:")
    print("\n".join(txt.split("\n")[:40]))
    raise SystemExit

pg = pd.DataFrame(allm, columns=["peer_group", "issuer_id", "gm_name"])
pg.to_csv(os.path.join(OUT, "iopp_peer_groups.csv"), index=False)
print(f"\n  {len(pg)} issuer assignments saved")

il = pd.read_csv(os.path.join(OUT, "issuer_level.csv"))
j = il.merge(pg, on="issuer_id", how="inner")
print(f"  matched to the paper's issuer-level file: {len(j)} of {len(pg)}")
if len(j):
    j["big"] = j.peer_group.str.contains("Mega|Large", case=False)
    print("\n  paper's scale measure S (log 2019 option volume), by Ginnie Mae peer group:")
    print(j.groupby("peer_group").S.agg(["count", "mean", "min", "max"]).round(2).to_string())
    from scipy.stats import spearmanr
    rank = {"Mega": 5, "Large": 4, "Medium": 3, "Small": 2, "Very Small": 1}
    j["gm_rank"] = j.peer_group.apply(
        lambda g: next((v for k, v in rank.items() if k.lower() in g.lower()), np.nan)
        if False else next((v for k, v in rank.items() if k.lower() in g.lower()), None))
    k = j.dropna(subset=["gm_rank"])
    if len(k) > 5:
        rho, p = spearmanr(k.S, k.gm_rank)
        print(f"\n  Spearman(paper's S, Ginnie Mae peer-group rank) = {rho:+.3f}  p={p:.4f}"
              f"   n={len(k)}")
    j.to_csv(os.path.join(OUT, "size_validation.csv"), index=False)
