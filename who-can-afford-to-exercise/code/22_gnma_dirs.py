import urllib.request, urllib.error, json, urllib.parse
from config import SEC_UA
H = {"User-Agent": SEC_UA, "Accept": "application/json"}
A = "https://www.ginniemae.gov/disclosure-api/api"

def j(u):
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300]
    except Exception as e:
        return None, str(e)[:120].encode()

# 1. look for a directory-listing endpoint
for p in ["directory", "directorylist", "dirs", "folders", "getDirectories",
          "files/directories", "disclosure/directories", "categories", "menu",
          "fileTypes", "filetypes", "products"]:
    s, b = j(f"{A}/{p}")
    if s == 200 and b.strip() not in (b"[]", b""):
        print("HIT", p, b[:400].decode("utf8", "replace"))
    else:
        print(f"  {s} {p} {b[:80].decode('utf8','replace') if s==200 else ''}")

# 2. brute force directory names
NAMES = ["disclosurehistory", "DisclosureHistory", "history", "History",
         "monthly", "Monthly", "loanlevel", "LoanLevel", "llmon", "LLMON",
         "MBS", "mbs", "SFPS", "sfps", "bulk", "Bulk", "root", "/", "",
         "MonthlyLoanLevel", "Loan Level", "Disclosure History Files",
         "monthlyfiles", "DDDFiles", "ddd", "DDD"]
print("\n--- directory probe ---")
for n in NAMES:
    s, b = j(f"{A}/files?directory={urllib.parse.quote(n)}")
    if s == 200 and b.strip() not in (b"[]", b""):
        print("HIT", repr(n), len(b), b[:300].decode("utf8", "replace"))
