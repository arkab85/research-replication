"""Pull issuer names for the largest unclassified issuers so the classification can be hand-extended."""
import pandas as pd, re, os
from config import OUT, DATA

# ---- issuer_info_202011.txt : fixed width, id(4) + program(2) + name(100) + address... ----
names = {}
for path in [DATA / "issuer_info_202011.txt",
             DATA / "issrinfo_202112.txt",
             DATA / "issrinfo_202212.txt"]:
    with open(path, "r", errors="replace") as f:
        for line in f:
            if len(line) < 10: continue
            iid = line[:4].strip()
            nm  = line[6:106].strip()
            if iid.isdigit() and nm:
                names.setdefault(int(iid), nm)

# ---- active_issuers_202212.txt : id, program, shortname, full name ----
with open(DATA / "active_issuers_202212.txt", "r", errors="replace") as f:
    for line in f:
        m = re.match(r"^(\d{4})\s+(\S+)\s+(.{18})(.{60})", line)
        if m:
            iid = int(m.group(1)); nm = m.group(4).strip()
            if nm: names.setdefault(iid, nm)

print("names for", len(names), "issuer ids")
top = pd.read_csv(os.path.join(OUT, "unclassified_top40.csv"))
top.columns = ["issuer_id", "n"]
top["name"] = top.issuer_id.map(names)
top.to_csv(os.path.join(OUT, "unclassified_top40_named.csv"), index=False)
print(top.to_string(index=False))

allc = pd.read_csv(os.path.join(OUT, "issuer_counts.csv"))
allc.columns = ["issuer_id", "n"]
allc["name"] = allc.issuer_id.map(names)
print("\n=== LARGEST ISSUERS OVERALL ===")
print(allc.to_string(index=False))
pd.Series(names).to_csv(os.path.join(OUT, "issuer_id_names.csv"), header=["name"])
