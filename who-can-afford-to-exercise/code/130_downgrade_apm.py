"""Section 9.3 claimed more than the data support. Rewrite it down.

I built a design around the July 2020 boundary: APM 20-07 cut the value of exercise for
everyone, Fannie's LL-2020-02 relaxed the liquidity charge for non-depositories only, so
differencing across the boundary should isolate capacity. The design fails its own
placebo. The July 2020 increment in the type gap is +0.072 (0.047), p = 0.13, and the
same calendar boundary one year earlier on a matched window gives +0.067 (0.024),
p = 0.007 -- the same magnitude, in a year with no policy change. Scanning every boundary
from April to September 2020 the increment rises monotonically from -0.139 to +0.123,
with nothing distinctive at July.

So the July movement cannot be attributed to the July rules, and the earlier text
overstated it. What survives is narrower and still worth reporting: the estimated gap
does not widen after the repooling restriction takes effect, so the headline is not an
artefact of it.
"""
import os, json
from config import PAPER as PAP, OUT

r = json.load(open(os.path.join(OUT, "results_dual_placebo.json")))
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


sub(r"""Three readings, in order of how much weight they can bear. First, the estimate is not an
artefact of the repooling restriction: the gap is at its widest before APM 20-07 takes
effect and narrows afterwards, so dropping July onward would strengthen rather than
weaken the headline. Second, the type that responded to the value cut is the type that
was actually exercising, which is what a value shock must do --- it can only bind on
someone with an option in the money. Third, and most usefully, a common value shock
moved the two types in \emph{opposite} directions here too, which is the same pattern
as March 2020 and the same reason it is hard to explain with value alone.

What this cannot do is serve as a clean difference-in-differences: both types are
treated, the nonbank increase is confounded with funding conditions normalising over the
same months, and three months is a short window. It is a dated event whose direction is
informative, not an identified effect.""",
    r"""One reading survives scrutiny and one does not, and it is worth separating them.

What survives is a robustness point. The estimated gap is at its widest before APM 20-07
takes effect and does not widen afterwards, so the headline estimate is not an artefact
of the repooling restriction; dropping July onward would not weaken it. The post-July gap
is \pApmAfter{} (s.e.\ \pApmAfterSE) and remains significant on its own.

What does not survive is the temptation to read the July movement as a test. The
narrowing is \pDualIncr{} (s.e.\ \pDualIncrSE, $p = \pDualIncrP$), which is not
distinguishable from zero. More tellingly, the same calendar boundary one year earlier,
on a window matched in length and structure, gives \pDualPlac{} (s.e.\ \pDualPlacSE,
$p = \pDualPlacP$) --- the same magnitude in a year with no policy change. Placing the
boundary at every month from April to September 2020 produces an increment that rises
monotonically across the window, with nothing distinctive at July. There is a seasonal
pattern in the type gap around midyear, and it is large enough to account for what
happens in July 2020.

I had expected more from this episode. Fannie Mae's Lender Letter 2020-02, issued
22~April 2020 and effective with the quarter ending 30~June, counted only thirty percent
of a COVID-forbearance loan's balance toward the seriously delinquent base that triggers
the incremental liquidity charge, and it reaches non-depositories only. Two rules
therefore land at the same date pulling in opposite directions, and differencing across
the boundary should in principle isolate the capacity channel from the common value
shock. The placebo says the boundary cannot carry that weight. The episode is reported
here as a robustness check on the headline, and as a description of what the
market did, not as an identified effect.""",
    "9.3 downgraded to a robustness check")

sub(r"""the
APM~20-07 episode of Section~\ref{sec:apm}, which is a dated shock; and the audited
balance-sheet evidence of Section~\ref{sec:reversal}, which is not estimated from this
panel at all. Those three are the load-bearing evidence.""",
    r"""and the audited balance-sheet evidence of
Section~\ref{sec:reversal}, which is not estimated from this panel at all. Those two are
the load-bearing evidence. The APM~20-07 episode of Section~\ref{sec:apm} is a robustness
check on the headline rather than a third leg; its own placebo will not support more.""",
    "7.5 no longer counts APM as a leg")

sub(r"""\emph{Third, a dated cut to the value of exercise moved the types in opposite directions
too.} Ginnie Mae's APM 20-07, effective 1~July 2020, made loans bought out of forbearance
ineligible for existing pool types and admitted them only to a non-TBA pool after six
timely payments. That is a reduction in the redelivery price, common to every issuer.
Depository exercise fell from \pApmDepPre\% to \pApmDepPost\%; nonbank exercise
\emph{rose}, from \pApmNbPre\% to \pApmNbPost\%, and the estimated gap narrowed rather
than widened (Table~\ref{tab:p10}). The headline estimate is therefore not an artefact of
that policy, and the one clean value shock in the sample behaves as March 2020 did.""",
    r"""\emph{Third, the estimate is not an artefact of the one policy change inside the
window.} Ginnie Mae's APM 20-07, effective 1~July 2020, cut the value of exercise for
every issuer by making loans bought out of forbearance ineligible for existing pool
types. The estimated gap does not widen after it takes effect
(Table~\ref{tab:p10}), so dropping July onward would not strengthen the headline.
Section~\ref{sec:apm} reports what the July boundary can and cannot support; a seasonal
pattern in the type gap means it cannot support a test.""",
    "intro: APM claim downgraded")

open(p, "w", encoding="utf-8").write(t)
print(f"\n{n} edits applied")
