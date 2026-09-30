"""A cross-issuer panel of the immobilisation measure, from audited filings.

"Loans eligible for repurchase from Ginnie Mae" (ASC 860-50) is the unpaid balance of
loans on which the repurchase option has vested and which the issuer has NOT bought out.
It is the paper's dependent variable, in dollars, on a balance sheet. Most issuers tag it
with a company extension the XBRL frames API never returns, so read it out of the filing.

For each issuer in the size-gradient test whose parent files with the SEC, pull the annual
reports covering 2018-2021 and extract the line with its column dates.
"""
import urllib.request, json, os, re, time
import pandas as pd
from config import OUT, CACHE, SEC_UA

os.makedirs(CACHE, exist_ok=True)
H = {"User-Agent": SEC_UA}

# issuer in the loan-level gradient test -> (filing entity, CIK, type)
FIRMS = [
    ("Pennymac Loan Services, LLC",      "PennyMac Financial Services", 1745916, "nonbank"),
    ("Nationstar Mortgage, LLC",         "Mr. Cooper Group",             933136, "nonbank"),
    ("Phh Mortgage Corporation",         "Ocwen Financial",              873860, "nonbank"),
    ("Caliber Home Loans, Inc.",         "Caliber Home Loans",          1821440, "nonbank"),
    ("Amerihome Mortgage Company,LLC",   "AmeriHome",                   1820807, "nonbank"),
    ("Flagstar Bank, Fsb",               "Flagstar Bancorp",            1033012, "depository"),
    ("Bokf, Na",                         "BOK Financial",                875357, "depository"),
    ("M&T Bank",                         "M&T Bank Corp",                 36270, "depository"),
    ("Fifth Third Bank, National Ass Ociatio", "Fifth Third Bancorp",     35527, "depository"),
    ("Truist Bank",                      "Truist Financial",              92230, "depository"),
    ("U. S. Bank, Na",                   "U.S. Bancorp",                  36104, "depository"),
    ("Wells Fargo Bank, N.A.",           "Wells Fargo",                   72971, "depository"),
]

PAT = re.compile(
    r"(loans?\s+(?:eligible\s+for|subject\s+to)\s+repurchase[^|]{0,60}|"
    r"(?:ginnie\s*mae|gnma)[^|]{0,40}repurchase[^|]{0,40})", re.I)
NUM = re.compile(r"^\(?\$?\s*([\d][\d,]{2,})\)?$")


def get(u, name, tries=3):
    p = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", name))
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return open(p, "rb").read()
    for i in range(tries):
        try:
            b = urllib.request.urlopen(
                urllib.request.Request(u, headers=H), timeout=180).read()
            open(p, "wb").write(b)
            time.sleep(0.3)
            return b
        except Exception as e:
            if i == tries - 1:
                print(f"    fetch failed: {e}")
                return None
            time.sleep(2)


def annual_reports(cik, lo="2019-01-01", hi="2022-12-31"):
    b = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"sub_{cik}.json")
    if not b:
        return pd.DataFrame()
    r = json.loads(b)["filings"]["recent"]
    d = pd.DataFrame({k: r[k] for k in
                      ["accessionNumber", "filingDate", "form", "primaryDocument"]})
    d = d[d.form.isin(["10-K", "S-1", "S-1/A", "S-4"])]
    d = d[(d.filingDate >= lo) & (d.filingDate <= hi)]
    # one 10-K per year; the last S-1/A if there is no 10-K
    d["yr"] = d.filingDate.str[:4]
    keep = d[d.form == "10-K"].drop_duplicates("yr")
    if keep.empty:
        keep = d.drop_duplicates("yr")
    return keep.sort_values("filingDate")


def flatten(b):
    t = b.decode("utf8", "ignore")
    t = re.sub(r"(?s)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", "|", t)
    t = re.sub(r"&nbsp;?|&#160;|&#8203;|&#xA0;", " ", t)
    t = re.sub(r"&amp;", "&", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"(\|\s*)+", "|", t)


def header_dates(seg):
    """Column dates from the table header preceding the line."""
    yrs = re.findall(r"(?:December\s*31,?|September\s*30,?|June\s*30,?)\s*\|?\s*(20\d\d)", seg)
    yrs += re.findall(r"\|\s*(20[12]\d)\s*\|", seg)
    out = []
    for y in yrs:
        if y not in out:
            out.append(y)
    return out


rows = []
for issuer, entity, cik, typ in FIRMS:
    print(f"\n{entity} (CIK {cik})")
    f = annual_reports(cik)
    if f.empty:
        print("  no filings in window"); continue
    for _, r in f.iterrows():
        u = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
             f"{r.accessionNumber.replace('-','')}/{r.primaryDocument}")
        b = get(u, f"{cik}_{r.accessionNumber}_{r.primaryDocument}")
        if not b:
            continue
        t = flatten(b)
        hits = 0
        for m in PAT.finditer(t):
            seg = t[m.end():m.end() + 260]
            nums = [x for x in (NUM.match(c.strip()) for c in seg.split("|")) if x]
            if len(nums) < 2:
                continue
            pre = t[max(0, m.start() - 1400):m.start()]
            yrs = header_dates(pre)
            if len(yrs) < 2:
                continue
            scale = 1e3 if re.search(r"in\s+thousands", pre, re.I) else (
                1e6 if re.search(r"in\s+millions", pre, re.I) else None)
            if scale is None:
                continue
            for y, n in zip(yrs, nums[:len(yrs)]):
                rows.append({"issuer": issuer, "entity": entity, "cik": cik, "type": typ,
                             "form": r.form, "filed": r.filingDate, "year": int(y),
                             "value_bn": float(n.group(1).replace(",", "")) * scale / 1e9,
                             "label": m.group(1).strip()[:48]})
            hits += 1
            if hits >= 6:
                break
        print(f"  {r.form} {r.filingDate}: {hits} matched line(s), {len(b)/1e6:.1f}MB")

d = pd.DataFrame(rows)
if d.empty:
    print("\nnothing extracted"); raise SystemExit
d = d[(d.year >= 2017) & (d.year <= 2022)]
# within firm-year keep the modal value (the balance-sheet line, not a note subtotal)
agg = (d.groupby(["entity", "type", "year"]).value_bn
       .agg(lambda s: s.value_counts().idxmax()).reset_index())
piv = agg.pivot_table(index=["type", "entity"], columns="year", values="value_bn")
print("\n" + "=" * 92)
print("LOANS ELIGIBLE FOR REPURCHASE FROM GINNIE MAE, $bn, year end")
print("=" * 92)
print(piv.round(3).to_string())
d.to_csv(os.path.join(OUT, "eligible_panel_raw.csv"), index=False)
agg.to_csv(os.path.join(OUT, "eligible_panel.csv"), index=False)
print("\nwrote eligible_panel.csv")
