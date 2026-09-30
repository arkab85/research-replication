"""Broader XBRL sweep, then validate the scale proxy against the balance sheet."""
import urllib.request, urllib.error, json, os, time
import numpy as np, pandas as pd
from config import OUT, SEC_UA

H = {"User-Agent": SEC_UA, "Accept": "application/json"}

ISSUERS = {   # GNMA issuer_id -> (name, SEC registrant consolidating it at FY2019, CIK)
    4094: ("PennyMac Loan Services",   "PennyMac Financial Services",  "0001745916"),
    4052: ("Nationstar Mortgage",      "Mr. Cooper Group",             "0000933136"),
    2397: ("PHH Mortgage",             "Ocwen Financial",              "0000873860"),
    4206: ("NewRez",                   "New Residential Investment",   "0001556593"),
    4255: ("Home Point Financial",     "Home Point Capital",           "0001830197"),
    1555: ("Guild Mortgage",           "Guild Holdings",               "0001821160"),
    4180: ("loanDepot.com",            "loanDepot",                    "0001831631"),
    4042: ("Quicken Loans",            "Rocket Companies",             "0001805284"),
}

def companyfacts(cik):
    u = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        return json.loads(urllib.request.urlopen(
            urllib.request.Request(u, headers=H), timeout=90).read())
    except Exception as e:
        return {"__err__": str(e)[:60]}

def best(fx, must, avoid=(), end="2019-12-31", floor=1e6):
    """Largest FY2019 year-end USD value among us-gaap tags whose name matches `must`."""
    gaap = fx.get("facts", {}).get("us-gaap", {})
    out = []
    for tag, node in gaap.items():
        lt = tag.lower()
        if not all(m in lt for m in must): continue
        if any(a in lt for a in avoid): continue
        for unit, rows in node.get("units", {}).items():
            if unit != "USD": continue
            for r in rows:
                if r.get("end") == end and r.get("val") and abs(r["val"]) >= floor:
                    out.append((abs(float(r["val"])), tag, r.get("form"), r.get("fy")))
    if not out: return None, None
    out.sort()
    return out[-1][0], out[-1][1]

rows, cache = [], {}
for iid, (nm, parent, cik) in ISSUERS.items():
    if cik not in cache:
        cache[cik] = companyfacts(cik); time.sleep(0.3)
    fx = cache[cik]
    r = dict(issuer_id=iid, issuer=nm, parent=parent, cik=cik)
    if "__err__" in fx:
        print(f"  {parent:<30} ERR {fx['__err__']}"); rows.append(r); continue
    r["assets"], r["assets_tag"] = best(fx, ["assets"], ["intangible", "deferred",
                                        "heldforsale", "netofallowance", "acquired"], floor=1e8)
    r["cash"], r["cash_tag"] = best(fx, ["cash"], ["restricted", "flow", "paid",
                                    "received", "supplement", "noncash"], floor=1e6)
    r["msr"], r["msr_tag"] = best(fx, ["servicing"], ["liabilit", "expense", "revenue",
                                  "fee", "income"], floor=1e6)
    r["equity"], r["equity_tag"] = best(fx, ["equity"], ["method", "securities",
                                        "incentive", "compensation"], floor=1e6)
    rows.append(r)
    print(f"  {parent:<30} A={r['assets']!s:>14} C={r['cash']!s:>13} "
          f"MSR={r['msr']!s:>13} E={r['equity']!s:>13}")

sec = pd.DataFrame(rows)
sec.to_csv(os.path.join(OUT, "sec_liquidity.csv"), index=False)

# ---- merge to the issuer panel and validate the scale proxy --------------------
iss = pd.read_csv(os.path.join(OUT, "issuer_level.csv"), index_col=0)
m = sec.set_index("issuer_id").join(iss[["name", "ext", "n19", "n20", "ebo19",
                                          "ebo20", "d_ebo", "S", "upb19"]])
m = m[m.assets.notna() & m.S.notna()].copy()
m["cash_assets"] = m.cash / m.assets
m["cash_msr"] = m.cash / m.msr
m["equity_assets"] = m.equity / m.assets
m["log_assets"] = np.log(m.assets)
m["log_msr"] = np.log(m.msr)
m = m.sort_values("S", ascending=False)

print("\n" + "=" * 92)
print("FY2019 BALANCE SHEETS OF THE NONBANK ISSUERS WITH SEC-FILING PARENTS")
print("=" * 92)
show = m[["parent", "ext", "n19", "S", "assets", "cash", "msr", "cash_assets",
          "cash_msr", "equity_assets", "ebo19", "ebo20", "d_ebo"]].copy()
for c in ["assets", "cash", "msr"]:
    show[c] = (show[c] / 1e9).round(2)
for c in ["cash_assets", "cash_msr", "equity_assets", "ebo19", "ebo20", "d_ebo"]:
    show[c] = show[c].round(3)
print(show.to_string())

print("\n--- does the scale proxy measure the servicing book? (nonbanks only) ---")
nb = m[m.ext.isin(["nonbank", "techfirst"])]
for a, b in [("S", "log_assets"), ("S", "log_msr"), ("S", "cash_assets"),
             ("S", "cash_msr"), ("S", "equity_assets")]:
    d = nb[[a, b]].dropna()
    if len(d) >= 4:
        print(f"   corr({a}, {b:<14}) = {d[a].corr(d[b]):+.3f}   "
              f"rank = {d[a].corr(d[b], method='spearman'):+.3f}   n={len(d)}")

print("\n--- liquidity and the 2020 retrenchment, nonbanks with filings ---")
for v in ["cash_assets", "cash_msr", "equity_assets", "log_msr"]:
    d = nb[[v, "d_ebo"]].dropna()
    if len(d) >= 4:
        print(f"   corr({v:<14}, change in exercise) = {d[v].corr(d.d_ebo):+.3f}  n={len(d)}")

m.to_csv(os.path.join(OUT, "sec_merged.csv"))
json.dump({"n_matched": int(len(m)),
           "corr_S_logmsr": float(nb[["S", "log_msr"]].dropna().corr().iloc[0, 1])
           if nb[["S", "log_msr"]].dropna().shape[0] >= 4 else None},
          open(os.path.join(OUT, "results_sec.json"), "w"), indent=1)
print("\nwrote sec_liquidity.csv, sec_merged.csv")
