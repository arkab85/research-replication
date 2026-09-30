"""Two additions: external validation of the scale proxy, and a named selection limit.

Both reviewer reports objected that issuer size is a proxy measured with error, and the
paper could only answer with an attenuation argument. Ginnie Mae publishes its own peer-
group classification of the same issuers, free and without a login, built from servicing
portfolio rather than from delinquency flow. Agreement between the two is external
evidence that the proxy measures the intended construct.

Separately, the balance-sheet evidence is selected in a way that can now be named rather
than gestured at: the largest and third-largest nonbank issuers by vesting volume in the
sample file nothing with the SEC.
"""
import os, json
from config import PAPER as PAP, OUT

v = json.load(open(os.path.join(OUT, "results_size_validation.json")))
print("size validation: " + ", ".join(f"{k} rho={x[0]:+.3f} n={x[2]}" for k, x in v.items()))

p = os.path.join(PAP, "jmp.tex")
t = open(p, encoding="utf-8").read()
n = 0


def sub(old, new, label, count=1):
    global t, n
    c = t.count(old)
    assert c == count, f"[{label}] expected {count}, found {c}"
    t = t.replace(old, new)
    n += c
    print(f"  ok  {label}")


sub(r"""What measurement error could do is make the two types' gradients differentially
attenuated --- if scale measures true exposure better for one type than the other
--- which would distort the magnitude of $\pi_3$ without generating the sign
reversal. I have no way to rule that out, and the balance-sheet correlations in
Section~\ref{sec:balance} are the only direct evidence on it.""",
    r"""What measurement error could do is make the two types' gradients differentially
attenuated --- if scale measures true exposure better for one type than the other ---
which would distort the magnitude of $\pi_3$ without generating the sign reversal.

There is, however, external evidence that the proxy measures the intended construct.
Ginnie Mae classifies every issuer into a peer group --- Mega, Large, Medium, Small or
Very Small, crossed with depository and non-depository --- and publishes the membership
with issuer identifiers. The classification is the agency's own, is built from servicing
portfolio rather than from delinquency flow, and is constructed for supervisory purposes
with no reference to anything in this paper. Across the \pgmN{} issuers in the estimation
sample that appear in it, the rank correlation between the scale measure used here and
the agency's own ordering is \pgmRho. Within type, where the test is identified, it is
\pgmRhoDep{} for depositories (\pgmNDep{} issuers) and \pgmRhoNb{} for nonbanks
(\pgmNNb{}), and on the active sample that the gradient is estimated on it is
\pgmRhoActive. The mean of the scale measure is monotone in the agency's ordering across
every peer group. That does not eliminate measurement error, but it does say the proxy
and the regulator's own size classification are ordering these firms the same way.""",
    "10.3 external validation of the scale proxy")

sub(r"""Seven issuers is not a panel, and they are the seven whose disclosure was specific
enough to use, which selects on being large and capital-markets-facing.""",
    r"""Seven issuers is not a panel, and they are the seven whose disclosure was specific
enough to use. The selection can be named rather than gestured at: the largest nonbank
issuer in the sample by vesting volume, and the third largest, are both absent, because
neither files periodic reports with the SEC --- one is privately held and funds itself in
the Rule 144A market, the other is a subsidiary whose parent does not report. Between them
they are a substantial share of the nonbank side of the market. The filer set is therefore
selected on being publicly financed, which is plausibly correlated with the constraint the
paper is about, and in the direction that would understate it.""",
    "7.6 named selection limit")

open(p, "w", encoding="utf-8").write(t)
print(f"\n{n} edits applied")
