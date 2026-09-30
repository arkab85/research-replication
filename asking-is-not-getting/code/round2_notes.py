"""What did the servicer's notes say in the 60 days after each hardship inquiry? Keyword flags per loan; no note text is exported
except short, digit-masked snippets for vocabulary discovery (kept local)."""
import csv, os, re, json, random
from collections import defaultdict, Counter
import pandas as pd
csv.field_size_limit(10**9); random.seed(7)
OUT = os.path.join(os.path.dirname(__file__), "out")
D = pd.read_parquet(os.path.join(OUT, "inq_timing.parquet")); D = D[(D.inq >= "2020-03-01") & (D.inq <= "2020-06-30")]
inq = {k: v.strftime("%Y-%m-%d") for k, v in D.inq.items()}; hi = {k: (v + pd.Timedelta(days=60)).strftime("%Y-%m-%d") for k, v in D.inq.items()}
lo = {k: (v - pd.Timedelta(days=2)).strftime("%Y-%m-%d") for k, v in D.inq.items()}
RX = {
 "docs":      re.compile(r"\brma\b|request for mortgage assistance|financial (package|docs|documents|information)|hardship (letter|affidavit|application)|proof of income|pay ?stubs?|bank statements?|tax returns?|4506|(docs?|documents?|documentation) (are |is |was |were )?(needed|required|requested|missing|not received)|(send|sent|mail|mailed|email|emailed|upload)\w* (in )?(the )?(docs|documents|package|application|paperwork)|loss mit(igation)? (package|application)|workout package|complete (the )?(package|application)", re.I),
 "incomplete":re.compile(r"incomplete|missing (docs?|documents?|items|information)|not (yet )?(received|returned)", re.I),
 "denied":    re.compile(r"\bden(y|ied|ial)\b|not eligible|ineligible|does ?n[o']t qualify|not qualify|declin(e|ed)|not approved|unable to (offer|approve|assist)", re.I),
 "investor":  re.compile(r"investor|not (a )?(federally|gov)|non[- ]?(gse|agency|government)|private(ly)? (held|owned|investor|label)|not (an? )?(fha|va|usda|fannie|freddie|gse)", re.I),
 "cares":     re.compile(r"cares act|federally backed|attest|verbal(ly)?|no (docs?|documents?|documentation|paperwork) (is |are )?(needed|required|necessary)", re.I),
 "approved":  re.compile(r"(forbearance|fb|plan)\b.{0,40}\b(approved|set ?up|granted|activated|booked|completed)|approved.{0,30}(forbearance|fb)", re.I),
 "repay":     re.compile(r"repayment plan|\brpp?\b|reinstate|promise to pay|\bptp\b|payment arrangement", re.I),
 "mod":       re.compile(r"modif", re.I),
}
flags = defaultdict(Counter); nnotes = Counter(); snippets = defaultdict(list); n = 0
with open(r"<DATA>/panel\Messages_Dec20.csv", "r", encoding="utf-8", errors="replace", newline="") as f:
    rd = csv.DictReader(f); rd.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rd.fieldnames]
    for row in rd:
        n += 1
        pid = (row.get("ParentId") or "").strip()
        if pid not in inq: continue
        v = (row.get("CreatedAt") or "")[:10]
        if v < lo[pid] or v > hi[pid]: continue
        if (row.get("Type") or "").strip() != "Servicer Note": continue
        t = row.get("Text") or ""; nnotes[pid] += 1
        for k, rx in RX.items():
            m = rx.search(t)
            if m:
                flags[pid][k] += 1
                if len(snippets[k]) < 400 and random.random() < .05:
                    s = max(0, m.start() - 70); snippets[k].append((pid, v, re.sub(r"\d", "#", t[s:m.end() + 70].replace("\n", " "))))
F = pd.DataFrame({k: pd.Series({p: c[k] for p, c in flags.items()}) for k in RX}).reindex(D.index).fillna(0)
F["notes60"] = pd.Series(nnotes).reindex(D.index).fillna(0)
F.to_parquet(os.path.join(OUT, "inq_note_flags.parquet"))
json.dump({k: v for k, v in snippets.items()}, open(os.path.join(OUT, "inq_snippets_LOCAL_ONLY.json"), "w"))
X = D.join(F); X["post"] = (X.inq >= "2020-04-06").astype(int)
print("rows streamed", n, "| inquirers", len(X), "| with notes", int((X.notes60 > 0).sum()))
print("\nshare of inquirers with >=1 note matching, by backing and period (first inquiry before / on-after 6 Apr 2020):")
print((X.assign(**{k: (X[k] > 0) for k in RX}).groupby(["Gov", "post"])[list(RX) ].mean() * 100).round(1).assign(n=X.groupby(["Gov", "post"]).size(), notes_med=X.groupby(["Gov", "post"]).notes60.median()).to_string())
print("\nconventional, post-6-Apr, by whether forbearance was completed:")
c = X[(X.Gov == 0) & (X.post == 1)]; print((c.assign(**{k: (c[k] > 0) for k in RX}).groupby("fb")[list(RX)].mean() * 100).round(1).assign(n=c.groupby("fb").size()).to_string())
print("\nsample masked snippets (vocabulary discovery):")
for k in ["docs", "investor", "denied", "cares"]:
    print("--", k)
    for pid, v, s in [x for x in snippets[k] if x[0] in set(c.index)][:6]: print("   ", v, "|", s[:230])
