import pymupdf, os
from config import PAPER as PAP
P = os.path.join(PAP, "jmp.pdf")
d = pymupdf.open(P)
t = "".join(d[i].get_text() for i in range(d.page_count))
i = t.find("When a financing constraint binds")
print("=== ABSTRACT ===")
print(t[i:i + 1400].replace("\n", " "))
j = t.find("The usual consequence of a binding")
print("\n=== CONCLUSION OPENING ===")
print(t[j:j + 900].replace("\n", " "))
