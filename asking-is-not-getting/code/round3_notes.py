"""One pass, three purposes.
(i)  Directive search: any record 20 Mar-20 Apr 2020 mentioning forbearance AND policy/investor/conventional language, by day and type.
(ii) Stated need at the time of asking: flags on servicer notes dated [inq-1d, inq+2d] for each inquirer.
(iii) journal: first date of each hardship marker and first servicer note, per boarded loan.
Snippets are digit-masked and stay local."""
import csv, os, re, json, random
from collections import defaultdict, Counter
import pandas as pd
csv.field_size_limit(10**9); random.seed(11)
OUT = os.path.join(os.path.dirname(__file__), "out")
D = pd.read_parquet(os.path.join(OUT, "inq_timing.parquet")); D = D[(D.inq >= "2020-03-01") & (D.inq <= "2020-06-30")]
lo = {k: (v - pd.Timedelta(days=1)).strftime("%Y-%m-%d") for k, v in D.inq.items()}; hi = {k: (v + pd.Timedelta(days=10)).strftime("%Y-%m-%d") for k, v in D.inq.items()}
conv = set(D.index[D.Gov == 0])
cv = pd.read_csv(r"<DATA>/relief\COVID_Inquiry_FB_Apr1_2021.csv", usecols=["LoanID", "ServicingTransferDate"], dtype=str, na_values=["NULL"]).dropna()
board = cv.groupby("LoanID")["ServicingTransferDate"].min().str.slice(0, 10).to_dict()

FB = re.compile(r"forbearance|\bfb\b|covid|cares act", re.I)
POL = re.compile(r"conventional|non[- ]?(gov|agency|gse|federally)|not (a )?(federally|gov)|private(ly)?[- ]?(held|label|investor)|investor (approval|guideline|direct|instruct|decision|require|policy|will|has|does)|policy|guideline|effective (immediately|today|\d)|going forward|no longer|must (now )?(submit|complete|apply)|rma (is |will be )?(now )?required|blanket|auto(matic(ally)?)? ?(approv|grant|enrol)", re.I)
NEED = {
 "jobloss":   re.compile(r"unemploy|laid[- ]?off|lay[- ]?off|furlough|lost (his|her|their|my|the)? ?job|job loss|out of work|not working|no work|business (is |was |has )?(been )?clos|clos(ed|ing) (down )?(his|her|their|the|my)? ?business|reduc(ed|tion) (in |of )?(hours|income|pay|work)|hours (were |was |have been |got )?(cut|reduced)|loss of income|income (was |is |has been )?(reduced|cut|lost)|curtailment", re.I),
 "illness":   re.compile(r"\bsick\b|\bill\b|illness|hospital|tested positive|has covid|covid positive|quarantin|diagnos", re.I),
 "precaution":re.compile(r"just in case|precaution|in case (he|she|they|of|it)|may (be|lose|not be able)|might (be|lose|not be able)|worried|concern(ed)? (about|that)|not (yet )?(been )?(impacted|affected)|still (working|employed)|able to (make|pay)|can (still )?(make|pay)|will (still )?(make|pay)|made (the |a |his |her )?payment|wants? to know (if|about|what)|inquir(ed|ing) about|options", re.I),
 "cantpay":   re.compile(r"(can ?not|can'?t|unable to|not able to|won'?t be able to) (make|pay|afford)|no (income|money|funds)|behind on", re.I),
}
MK = {"employment": re.compile(r"unemploy|laid off|lay ?off|lost (his|her|their)? ?job|job loss|furlough|reduced hours|curtail", re.I),
      "health": re.compile(r"\bill(ness)?\b|hospital|medical|surgery|\bsick\b|disab|cancer", re.I),
      "family": re.compile(r"divorce|separat(ed|ion)|death|passed away|deceased|funeral|custody", re.I)}
pol_by = Counter(); pol_snip = []; need = defaultdict(Counter); need_n = Counter(); first = defaultdict(dict); n = 0
with open(r"<DATA>/panel\Messages_Dec20.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f); rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        n += 1
        v = (row.get("CreatedAt") or "")[:10]; pid = (row.get("ParentId") or "").strip(); typ = (row.get("Type") or "").strip(); t = row.get("Text") or ""
        if "2020-03-20" <= v <= "2020-04-20" and FB.search(t) and POL.search(t):
            key = "conv-inquirer note" if (typ == "Servicer Note" and pid in conv) else ("other servicer note" if typ == "Servicer Note" else "NON-SERVICER: " + (typ or (row.get("AppliesTo") or "?")))
            pol_by[(v, key)] += 1
            if False and (key.startswith("NON") and len(pol_snip) < 300) or (key.startswith("conv") and "2020-04-03" <= v <= "2020-04-10" and random.random() < .25 and len(pol_snip) < 600):
                m = POL.search(t); s = max(0, m.start() - 110); pol_snip.append((v, key, re.sub(r"\d", "#", t[s:m.end() + 140].replace("\n", " "))))
        if typ != "Servicer Note": continue
        if pid in lo and lo[pid] <= v <= hi[pid]:
            need_n[pid] += 1
            for k, rx in NEED.items():
                if rx.search(t): need[pid][k] += 1
        b = board.get(pid)
        if b and v >= b:
            fd = first[pid]
            if "note" not in fd or v < fd["note"]: fd["note"] = v
            for k, rx in MK.items():
                if (k not in fd or v < fd[k]) and rx.search(t): fd[k] = v
F = pd.DataFrame({k: pd.Series({p: c[k] for p, c in need.items()}) for k in NEED}).reindex(D.index).fillna(0); F["n_call_notes"] = pd.Series(need_n).reindex(D.index).fillna(0)
F.to_parquet(os.path.join(OUT, "inq_need_flags_wide.parquet"))
FD = pd.DataFrame.from_dict(first, orient="index"); FD["board"] = pd.Series(board); pass
pol_snip = []  # snippets are no longer stored
P = pd.Series(pol_by).rename("n").reset_index(); P.columns = ["day", "kind", "n"]; P.to_csv(os.path.join(OUT, "policy_mentions_by_day.csv"), index=False)
print("rows", n, "| inquirers with call-day notes:", int((F.n_call_notes > 0).sum()), "of", len(F))
print("\npolicy-language records by day (conv-inquirer notes vs non-servicer records):")
print(P[P.kind.str.startswith(("conv", "NON"))].pivot_table(index="day", columns="kind", values="n", aggfunc="sum").fillna(0).astype(int).to_string())
print("\nNON-SERVICER snippets (masked):")
for v, k, s in [x for x in pol_snip if x[1].startswith("NON")][:25]: print("  ", v, "|", k[14:40], "|", s[:300])
print("\nconventional-inquirer notes 3-10 April (masked sample):")
for v, k, s in [x for x in pol_snip if x[1].startswith("conv")][:30]: print("  ", v, "|", s[:300])
