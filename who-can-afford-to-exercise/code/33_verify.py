"""Verify the open items: submission venues and the employment disclosure."""
import pymupdf, glob, os, zipfile, re
from config import DATA as DL


print("=" * 78); print("SUBMISSION EVIDENCE"); print("=" * 78)
for f in ["journal-S-26-01380.pdf", "main_v_blind.pdf", "JREFE_anonymized_source.pdf"]:
    p = os.path.join(DL, f)
    if not os.path.exists(p): print(f"  {f}: not found"); continue
    d = pymupdf.open(p)
    t = d[0].get_text()[:900].replace("\n", " ")
    print(f"\n--- {f} ({d.page_count} pp, modified "
          f"{__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d}) ---")
    print("   ", t[:700])

for f in ["cover_letter_v.docx", "cover_letter_v.pdf"]:
    p = os.path.join(DL, f)
    if not os.path.exists(p): print(f"\n  {f}: not found"); continue
    print(f"\n--- {f} ---")
    if f.endswith(".pdf"):
        d = pymupdf.open(p); print("   ", d[0].get_text()[:900].replace("\n", " "))
    else:
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml").decode("utf8", "replace")
        txt = re.sub(r"<[^>]+>", " ", xml)
        print("   ", re.sub(r"\s+", " ", txt)[:900])

print("\n" + "=" * 78); print("EMPLOYER NAME IN THE USER'S OWN PAPERS"); print("=" * 78)
for f in ["RF_Who_Can_Afford_to_Exercise_WP_SSRN.pdf", "Delegated_Relief_WP_SSRN.pdf",
          "JFQA_What_Relief_Buys_WP_SSRN.pdf"]:
    p = os.path.join(DL, f)
    if not os.path.exists(p): continue
    d = pymupdf.open(p)
    t = "".join(d[i].get_text() for i in range(min(3, d.page_count)))
    for kw in ["Rocktop", "Carrington", "Principal Data Scientist", "employed",
               "employment", "Clarion"]:
        for m in re.finditer(kw, t, re.I):
            print(f"  {f}: ...{t[max(0,m.start()-160):m.start()+200]}...".replace("\n", " "))
            break
