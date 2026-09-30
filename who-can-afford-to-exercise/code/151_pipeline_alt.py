"""What is currently sitting at journal, and at the journal.

Recommending a venue without knowing how many of the author's papers are already there
would be careless: a third or fourth simultaneous submission to the same journal draws the
same reviewer and reads badly.
"""
import glob, os, re
import pymupdf

FILES = sorted(set(glob.glob(r"C:\Users\arkab\Downloads\journal*.pdf")
                   + glob.glob(r"C:\Users\arkab\Downloads\*v*.pdf")
                   + glob.glob(r"C:\Users\arkab\Downloads\*MS_and_JFQA*")))

for f in FILES:
    if os.path.isdir(f):
        continue
    try:
        d = pymupdf.open(f)
        t = d[0].get_text()
    except Exception as e:
        print(f"  {os.path.basename(f):<44} unreadable ({type(e).__name__})")
        continue
    t1 = re.sub(r"\s+", " ", t)
    num = re.search(r"Manuscript Number:\s*([A-Za-z0-9\-]+)", t1)
    title = re.search(r"Full Title:\s*(.{5,95}?)\s+(?:Article Type|Short Title|Keywords)", t1)
    typ = re.search(r"Article Type:\s*([A-Za-z ]+?)\s+(?:Keywords|Manuscript|Corresponding)", t1)
    draft = "Manuscript Draft" in t1
    print(f"\n  {os.path.basename(f)}  ({d.page_count}p)")
    print(f"     number : {num.group(1) if num else '(none shown)'}")
    print(f"     title  : {title.group(1) if title else t1[:70]}")
    print(f"     type   : {typ.group(1).strip() if typ else '?'}"
          + ("   [marked Manuscript Draft]" if draft else ""))
