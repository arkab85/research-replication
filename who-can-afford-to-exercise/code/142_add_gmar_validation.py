"""Replace the peer-group validation with the stronger dollar-denominated one, and
resolve the differential-attenuation worry the paper had said it could not resolve.

Ginnie Mae's monthly market report publishes the servicing book in dollars for the thirty
largest issuers. Twenty-six of them are in the estimation sample. That turns the scale
proxy from something the paper argues measures the book into something measured against
the book.

It also answers a question Section 10.3 had raised and left open. Measurement error that
is differential across types would distort the contrast. It is differential: the proxy
tracks the book almost perfectly among nonbanks and only loosely among depositories. The
direction is the useful part. A noisier proxy attenuates that type's gradient more, so the
depository gradient is the more attenuated of the two; correcting for it would make the
depository gradient larger and positive, and therefore the contrast between the two larger
in magnitude, not smaller. The reported difference is conservative.
"""
import os, json, re
from config import OUT, PAPER as PAP

v = json.load(open(os.path.join(OUT, "results_gmar_validation.json")))
print("GMAR validation:", v)


def num(x, k=2):
    return ("$-$" if x < 0 else "") + f"{abs(x):.{k}f}"


mac = {
    "gmarN": str(v["n"]), "gmarRho": num(v["spearman"]), "gmarPearson": num(v["pearson"]),
    "gmarRhoNb": num(v["nonbank"][0]), "gmarNNb": str(v["nonbank"][2]),
    "gmarRhoDep": num(v["depository"][0]), "gmarNDep": str(v["depository"][2]),
    "gmarPDep": f"{v['depository'][1]:.2f}",
}
pth = os.path.join(PAP, "numbers_pool.tex")
txt = open(pth, encoding="utf-8").read()
new = []
for k, val in sorted(mac.items()):
    pat = re.compile(r"(\\newcommand\{\\p" + k + r"\}\{)[^}]*(\})")
    if pat.search(txt):
        txt = pat.sub(lambda m: m.group(1) + val + m.group(2), txt)
    else:
        new.append("\\newcommand{\\p" + k + "}{" + val + "}")
if new:
    txt = txt.rstrip() + "\n" + "\n".join(new) + "\n"
open(pth, "w", encoding="utf-8").write(txt)
print(f"  {len(mac)} macros ({len(new)} new)")

p = os.path.join(PAP, "jmp.tex")
t = open(p, encoding="utf-8").read()
old_start = "There is external evidence that the proxy measures the intended construct."
i = t.index(old_start)
j = t.index("order these firms the same way.", i) + len("order these firms the same way.")

NEW = r"""There is external evidence on what the proxy measures, and it is worth reporting in full
because it resolves the question the previous paragraph leaves open.

Ginnie Mae publishes, monthly and without restriction, the unpaid principal serviced by
each of the thirty largest holders of Ginnie Mae servicing rights. That is the servicing
book itself, in dollars, from the agency. \pgmarN{} of those issuers are in the estimation
sample. Across them the rank correlation between the scale measure used here and the log
of the servicing book is \pgmarRho{} (Pearson \pgmarPearson). The agency separately assigns
every issuer to a size peer group; across the \pgmN{} sample issuers it classifies, the
scale measure correlates \pgmRho{} with that ordering, and its mean is monotone in the
agency's ordering across every group.

The informative part is that the agreement is \emph{not} uniform across types. Among
nonbanks the proxy tracks the servicing book almost exactly, \pgmarRhoNb{} on
\pgmarNNb{} issuers. Among depositories it tracks it only loosely, \pgmarRhoDep{} on
\pgmarNDep{} issuers, which is not distinguishable from zero at conventional levels
($p = \pgmarPDep$). So the measurement error is differential, which is precisely the case I
could not rule out above.

Its direction, though, is the opposite of a threat. A noisier proxy attenuates that type's
gradient more, so of the two it is the \emph{depository} gradient that is the more
attenuated. Correcting for it would make the depository gradient larger and more positive,
and therefore the contrast $\pi_3$ between the two gradients larger in magnitude rather
than smaller. The reported difference is conservative with respect to the measurement
problem it has, and the sign of the contrast --- which is what the theory pins down --- is
not something classical error of either magnitude can manufacture."""
t = t[:i] + NEW + t[j:]
open(p, "w", encoding="utf-8").write(t)
print("  rewrote the Section 10.3 validation passage")
