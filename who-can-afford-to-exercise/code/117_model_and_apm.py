"""Fix what the model actually derives, and add the APM 20-07 test.

The reviewer is right on three counts in Section 3: G was defined over V but the
solution ranks V/B; an unconstrained issuer's exercise RATE is invariant to the number
of vested options, so (P1) cannot claim a larger flow raises it; and proportional
scaling of capacity gives a ZERO depository gradient, not a positive one, so (P3)
signs the difference of gradients but not each level. The claim that no mechanism
operating through value can produce the reversal is also too strong, because the
paper's own carrying cost is owner-specific.
"""
import os
from config import PAPER as PAP

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


# ---------------------------------------------------------------- notation
sub(r"""Let $G_{jt}(\cdot)$ denote the distribution of $V$
across the options vesting for issuer $j$ in month $t$.""",
    r"""Because the budget constraint is in dollars, the object the issuer ranks on is
value per dollar committed. Write $r_i = V_i / B_i$ and let $G_{jt}(\cdot)$ denote the
distribution of $r$ across the options vesting for issuer $j$ in month $t$, and
$H_{jt}(\cdot)$ the corresponding balance-weighted distribution, $dH \propto B\,dG$.
Everything below is a cutoff in $r$.""",
    "notation: rank on V/B")

sub(r"""The problem is a continuous knapsack. The issuer ranks loans by $V_i/B_i$ and
repurchases from the top until the budget is exhausted, so the solution is a
cutoff $v^{*}_{jt}$ with""",
    r"""The problem is a continuous knapsack. The issuer repurchases in decreasing order
of $r_i$ until the budget is exhausted, so the solution is a cutoff $v^{*}_{jt}$ in
$r$, with""",
    "solution wording")

# ---------------------------------------------------------------- (P1)
sub(r"""\paragraph{(P1) Sign.} A fall in $P$ or in $\pi$ --- the trade becoming less
valuable --- is a market-wide price change. It lowers $V$ for every issuer and
therefore lowers exercise for both types. It cannot raise it for either. A
binding constraint on one type, by contrast, raises the other's exercise rate:
the same common shock delivers the unconstrained a larger stock of vested
options on their own books, at a moment when the shadow price of their own
balance-sheet space has not risen. Note that this is not a transfer: the option
belongs to the issuer of record (Section~\ref{sec:multi}), so a depository cannot
buy a nonbank's foregone option. Each type acts only on its own book. \emph{Opposite movement by type is diagnostic of
capacity; common movement is consistent with either.}""",
    r"""\paragraph{(P1) Sign.} Two things follow, and it is worth separating what the
model delivers from what the data will have to supply.

What it delivers: a fall in the common components of value --- the redelivery price
$P$ or the cure probability $\pi$ --- lowers $r$ for every issuer and therefore lowers
exercise for \emph{both} types. It cannot raise it for either. And for an
unconstrained issuer, $v^{*}=0$ and $\Pr(\text{exercise}) = 1 - G_{jt}(0)$, which does
not depend on the number of options vesting: a bigger wave changes the number
exercised, not the rate.

What it does not deliver: a mechanical reason for the unconstrained type's rate to
\emph{rise}. Nothing here makes one issuer's constraint loosen another's, and the
option cannot be transferred (Section~\ref{sec:multi}), so a depository cannot take up
what a nonbank declines. Each type acts only on its own book. A rise in the
unconstrained rate requires something further --- a fall in their own carrying cost
$\kappa$, or a change in the composition of what vests to them --- and
Section~\ref{sec:activity} reports that depository exercise did rise.

The diagnostic content is therefore one-sided but sharp. \emph{Opposite movement by
type cannot be produced by a common shock to $P$ or $\pi$; common movement is
consistent with either story.}""",
    "(P1) derivation")

# ---------------------------------------------------------------- (P3)
sub(r"""For a depository, $L$ scales with the institution, so $K/N$ is roughly
invariant to size and a larger institution --- with more slack and more room to
absorb what others decline --- weakly increases exercise. For a nonbank, $L$ is
set by facility limits that do not scale with the delinquent book and tighten
under stress, so $K/N$ is \emph{decreasing} in size. Hence
\[
\frac{\partial\,\Delta\Pr(\text{exercise})}{\partial\,\text{size}} \;>\; 0
\ \text{ for depositories},
\qquad
\frac{\partial\,\Delta\Pr(\text{exercise})}{\partial\,\text{size}} \;<\; 0
\ \text{ for nonbanks}.
\]
This is the paper's sharpest prediction, because no mechanism operating through
$V$ produces it. $V$ is a property of a loan, not of its owner: it does not know
the charter of the institution holding it, so it cannot make the sign of a size
gradient depend on that charter.""",
    r"""For a depository, $L$ scales with the institution, so $K/N$ is roughly invariant
to $S_j$ and the gradient is approximately zero; it is strictly positive only under the
further assumption that liquid resources scale \emph{more} than proportionately with
size, or that carrying cost $\kappa$ falls in size. For a nonbank, $L$ is set by
facility limits that do not scale with the delinquent book and tighten under stress, so
$K/N$ is decreasing in $S_j$ and the gradient is strictly negative. Hence
\[
\frac{\partial\,\Delta\Pr(\text{exercise})}{\partial S_j}\bigg|_{\text{dep}}
\;\geq\; 0
\;>\;
\frac{\partial\,\Delta\Pr(\text{exercise})}{\partial S_j}\bigg|_{\text{nonbank}},
\]
so what the model signs is the \emph{difference} of the two gradients, not the level of
either. That is also what Section~\ref{sec:size} estimates, and the empirical object
there is the difference.

This is the paper's sharpest prediction, and the reason is worth stating precisely
rather than sweepingly. The common components of value, $P$ and $\pi$, are properties
of a loan: they do not know the charter of the institution holding it, so they cannot
make the sign of a size gradient depend on that charter. The carrying cost $\kappa_i$
\emph{is} owner-specific, so a mechanism running through $\kappa$ could in principle
generate a charter-dependent gradient --- but a mechanism in which the cost of holding
a non-earning asset rises with a nonbank's book and not with a depository's is a
balance-sheet-capacity mechanism under another name. The prediction rules out value
stories that operate on the loan; it does not, and cannot, rule out every story that
operates on the owner.""",
    "(P3) derivation and the scope of the claim")

# ---------------------------------------------------------------- APM 20-07
anchor = r"""\subsection{Immobilised, not reallocated}"""
APM = r"""\subsection{A dated shock to the value of exercise}
\label{sec:apm}

The sample contains one clean, dated cut to the value of the buyout, and it provides a
test that does not depend on the model at all.

On 29~June 2020 Ginnie Mae issued APM 20-07. Any loan that entered COVID-19 forbearance
and is bought out on or after 1~July 2020 became ineligible collateral for existing pool
types; such a loan can be re-securitised only into a new ``RG'' pool, only after six
consecutive timely payments, and only once 210 days have passed since it was last
delinquent. RG pools are not TBA eligible and price materially below
them.\footnote{Ginnie Mae, All Participants Memorandum 20-07, ``Temporary Pooling
Restrictions on Re-performing Loans,'' 29~June 2020. Ginnie Mae's own September 2020
market commentary notes that the pricing of RG pools ``is expected to be significantly
less than TBA counterparts.'' Loans modified under FHA's COVID-19 loss-mitigation suite
were exempt.} That is a direct reduction in $P$, the redelivery price --- the
\emph{value} of exercise --- and it applies to every issuer equally, whatever its
funding structure.

The comparative statics differ sharply. A value account predicts that both types cut
exercise after 1~July. The capacity account predicts that the type gap is unaffected by
it, because $K_{jt}$ is untouched.

Table~\ref{tab:p10} splits the post period. Depository exercise \emph{fell} from
\pApmDepPre\% in March--June to \pApmDepPost\% in July--September; nonbank exercise
\emph{rose}, from \pApmNbPre\% to \pApmNbPost\%. The estimated gap narrows rather than
widens: with issuer and month effects the nonbank interaction is \pApmShock{} (s.e.\
\pApmShockSE) for March--June and \pApmAfter{} (s.e.\ \pApmAfterSE) for
July--September, and within pool and month \pApmShockPool{} (s.e.\ \pApmShockPoolSE)
and \pApmAfterPool{} (s.e.\ \pApmAfterPoolSE).

Three readings, in order of how much weight they can bear. First, the estimate is not an
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
informative, not an identified effect.

\subsection{Immobilised, not reallocated}"""
sub(anchor, APM, "APM 20-07 subsection")

# table P10, placed with the others
i = t.index(r"\input{tables/P9}")
j = t.index(r"\end{table}", i) + len(r"\end{table}")
TAB = r"""

\begin{table}[htbp]\centering\small
\caption{The type gap before and after the repooling restriction}
\label{tab:p10}
\input{tables/P10}
\vspace{0.4em}
\tabnote{Linear probability models of buyout on Nonbank interacted with each
sub-period, January 2019 to September 2020, omitted category January 2019 to February
2020. APM 20-07, issued 29~June 2020, made loans bought out on or after 1~July 2020 and
previously in COVID-19 forbearance ineligible for existing pool types. It cut the value
of exercise for every issuer regardless of funding structure. Controls are note rate,
credit score, current LTV and loan age. The second column restricts to pool-by-month
cells containing both issuer types. Standard errors clustered by issuer.
$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}
\end{table}"""
t = t[:j] + TAB + t[j:]
n += 1
print("  ok  table P10")

print(f"\n{n} edits applied")
open(p, "w", encoding="utf-8").write(t)
