"""Verify the loss-reserve validation papers, which carry the paper's positioning.

The methodological claim is that validating a recognised amount against an independent
benchmark is an established but thinly populated genre, whose home is the property-casualty
loss-reserve literature. Citing that literature wrongly would be worse than not citing it,
so check each against Crossref before it goes in.
"""
import urllib.request, json, time, urllib.parse

UA = {"User-Agent": "Academic reference check (mailto:researcher@example.edu)"}
WANT = [
    ("Optimistic reporting in the property casualty insurance industry",
     "Journal of Accounting and Economics"),
    ("The characteristics and valuation of loss reserves of property casualty insurers",
     "Review of Accounting Studies"),
    ("Discretionary behavior with respect to allowances for loan losses and the behavior of "
     "security prices", "Journal of Accounting and Economics"),
    ("The effect of reporting incentives on the reliability of managers' loss reserve estimates",
     "Journal of Accounting Research"),
    ("Discretionary reporting of loss reserves by property casualty insurers",
     "Journal of Accounting Research"),
]

for title, venue in WANT:
    q = ("https://api.crossref.org/works?query.bibliographic="
         + urllib.parse.quote(title) + "&rows=3")
    try:
        with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=60) as r:
            j = json.loads(r.read())
    except Exception as e:
        print(f"  QUERY FAILED  {title[:55]}  {type(e).__name__}")
        continue
    items = j.get("message", {}).get("items", [])
    print(f"\n  searched: {title[:66]}")
    for it in items[:2]:
        au = "; ".join(f"{a.get('family','')}, {a.get('given','')}"
                       for a in it.get("author", [])[:4])
        ct = (it.get("container-title") or [""])[0]
        yr = (it.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
        print(f"    -> {(it.get('title') or [''])[0][:70]}")
        print(f"       {au}")
        print(f"       {ct} ({yr}) vol {it.get('volume','?')} no {it.get('issue','?')} "
              f"pp {it.get('page','?')}  doi {it.get('DOI','?')}")
    time.sleep(1.5)
