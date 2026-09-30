"""Replace the reversal subsection with the four-issuer version and add table P8."""
import os
from config import PAPER as PAP

p = os.path.join(PAP, "jmp.tex")
t = open(p, encoding="utf-8").read()

start = t.index(r"\subsection{Outside the sample: the reversal}")
end = t.index("%=====================================================================\n"
              r"\section{Where the Loans Went}")

NEW = r"""\subsection{Outside the sample: the balance sheets}
\label{sec:reversal}

The loan-level extract stops in September 2020, which leaves two questions
unanswerable from inside it. A shock and a trend look alike while the shock is
happening; they look different afterwards. And a measure built by the author from
a disclosure file is worth more if an auditor independently signed off on the same
quantity. This subsection answers both from a source that requires no loan-level
data at all.

\subsubsection*{The same object, on an audited balance sheet}

Under ASC 860-50 a Ginnie Mae issuer that acquires the unilateral right to
repurchase a delinquent loan is deemed to have regained effective control of it,
and must bring the loan back onto its balance sheet as an asset with an offsetting
liability --- as Mr.\ Cooper's disclosure puts it, ``regardless of the Company's
intention to repurchase the loan.''\footnote{Mr.\ Cooper Group, Form 10-K for
fiscal 2020, Note 9. The line is variously captioned ``loans eligible for
repurchase,'' ``loans subject to repurchase from Ginnie Mae,'' and ``loans
eligible for repurchase from GNMA.''} The balance is therefore the unpaid
principal of options that have \emph{vested and not been exercised}. That is the
paper's immobilisation measure, in dollars, audited, and disclosed by every issuer
that files with the SEC.

Four of the nonbank issuers in the size-gradient test disclose it over the
episode, and together they are \pWedgeShare{} of the nonbank options vesting in
the sample. PennyMac and Mr.\ Cooper report it in their 10-Ks. Caliber and
AmeriHome were private in 2019 and never completed their offerings, but both filed
registration statements in late 2020 whose audited balance sheets carry the line
for December 2018, December 2019 and 30 September 2020 --- the last month of the
loan-level sample. Table~\ref{tab:p8} collects them.

\subsubsection*{The balance grew far faster than the flow}

The obvious objection is that this is a stock, so it rises when more loans go
delinquent whether or not exercise behaviour changed. That objection is testable,
because the loan-level data measures the flow directly: it counts every option
that vested at each issuer. If exercise had not changed, the stock of unexercised
options would have grown roughly in proportion to that flow.

It did not. Across the four issuers the flow of vesting options grew by a factor
of \pFlowMultMin{} to \pFlowMultMax{}. The reported balance grew by a factor of
\pRepMultMin{} to \pRepMultMax{}. Every firm's balance outran its own flow, by a
factor of \pWedgeMin{} to \pWedgeMax{}. And the size of the wedge lines up with
the size of the retrenchment: across the four firms it correlates \pWedgeCorr{}
with the fall in the issuer's own exercise rate. With four firms that correlation
is descriptive, not a test. The direction is not: it is the same sign as the
paper's estimate, in dollars, on statements someone else audited.

\subsubsection*{The measure validates}

The overlap is closer than corroboration usually is. Across the eight firm-years
in Table~\ref{tab:p8}, the unexercised balance computed from the loan-level
disclosure and the balance the issuer reports correlate \pValCorr{}
(\pValCorrLog{} in logs; \pValCorrTwenty{} across the four firms in 2020 alone).
The levels differ by a roughly constant factor of \pValRatioLo{} to
\pValRatioHi{}, which is what they should do: the loan-level measure accumulates
over the year while the filing reports a point-in-time balance from which cures
and later repurchases have already dropped out. The paper's dependent variable and
an audited public balance sheet are tracking the same thing.

\subsubsection*{The reversal}

Table~\ref{tab:p7} follows the largest of the four past the end of the sample, at
quarterly frequency. PennyMac's balance was flat before the shock: an average of
\$\pRecPreDol{}bn over 2018 to February 2020 with a standard deviation of
\$\pRecPreSd{}bn and no trend. It rose to \$\pRecPeakDol{}bn at its 2020 peak,
\pRecPeakMult{} times the pre-shock level. It then came back down, averaging
\$\pRecPostDol{}bn from 2022 onward. Scaled by total assets, to net out the growth
of the firm's balance sheet, the same pattern holds: a pre-shock mean of
\pRecPreA{} with a standard deviation of \pRecPreASd{} and a mildly \emph{negative}
trend, a peak of \pRecPeakA, and \pRecPostA{} from 2022. Mr.\ Cooper's annual
series does the same thing --- \$\pCoopPre{}bn at the end of 2019, \$\pCoopPeak{}bn
at the end of 2020, \$\pCoopPost{}bn at the end of 2021 --- and its 2020 balance
was \pCoopCaresPct\% CARES Act forbearance loans by the firm's own accounting.
Caliber's 2019 balance was \emph{below} its 2018 balance of \$\pCalTwoK{}bn before
it rose tenfold.

This is what the within-sample evidence could not supply. A differential trend
that happened to steepen in March 2020 does not reverse in 2021 and settle. A
liquidity shock does. The reversion is also incomplete --- the 2022-onward level
is \pRecPostMult{} times the pre-shock level in dollars and \pRecPostAMult{} times
scaled by assets --- which is consistent with a permanently larger delinquent book
rather than a return to the old regime, and the paper does not claim otherwise.

\subsubsection*{Limits}

Four issuers is not a panel, and they are the four that had a reason to file, so
they are larger and more capital-markets-facing than the typical nonbank issuer.
The annual frequency for three of them cannot separate March from December. The
wedge test assumes the duration of an unexercised option did not change, and
forbearance almost certainly lengthened it, which would inflate the stock for
reasons other than the decision --- though a longer duration is itself
immobilisation. What the exercise establishes is narrower than the main design and
independent of it: the quantity this paper says moved, moved by the amount it says,
on balance sheets the author did not construct.

"""

open(p, "w", encoding="utf-8").write(t[:start] + NEW + t[end:])

# table P8, right after P7 so the reorder step can place both by citation order
TAB = r"""
\begin{table}[htbp]\centering\small
\caption{The immobilisation measure on four issuers' audited balance sheets}
\label{tab:p8}
\input{tables/P8}
\vspace{0.4em}
\tabnote{Reported balance is the unpaid principal of Ginnie Mae loans whose
repurchase option had vested and had not been exercised, recognised under ASC
860-50. Sources: PennyMac Financial Services and Mr.\ Cooper Group, Forms 10-K;
Caliber Home Loans and AmeriHome, Forms S-1/A filed in late 2020 and January 2021.
Caliber and AmeriHome report as of 30 September 2020, the last month of the
loan-level sample, and their 2019 columns are the audited 31 December 2019
balances; options vesting and exercise rates for those two firms are computed over
January--September of each year to match. Options vesting and exercise rates are
from the loan-level disclosure. The ratio is balance growth divided by vesting
growth: it is one if exercise behaviour did not change.}
\end{table}
"""
t = open(p, encoding="utf-8").read()
anchor = "\\input{tables/P7}"
i = t.index(anchor)
j = t.index(r"\end{table}", i) + len(r"\end{table}")
open(p, "w", encoding="utf-8").write(t[:j] + "\n" + TAB + t[j:])
print("rewrote the reversal subsection and added table P8")
