"""Reframe the accounting manuscript for Review of Accounting Studies.

The draft had no citations at all, which is a decision at any accounting journal
regardless of what the evidence shows. This replaces the front half: an introduction that
opens on the accounting question, a related-literature and hypothesis-development section
positioning the paper inside the loss-reserve validation genre that RAS itself publishes,
and explicit hypotheses. The results sections are unchanged in substance.
"""
import os
from config import PAPER as PAP

p = os.path.join(PAP, "accounting.tex")
t = open(p, encoding="utf-8").read()

# ------------------------------------------------------------------ abstract
a = t.index(r"\begin{center}\textbf{Abstract}\end{center}")
b = t.index(r"\end{minipage}\end{center}", a)
NEWABS = r"""\begin{center}\textbf{Abstract}\end{center}
\small
Whether a recognised amount measures the construct a standard has in mind is rarely
testable, because the construct is usually unobservable --- which is generally why it
required a standard. The exception is a small literature that validates recognised
amounts against independently realised outcomes, concentrated in property-casualty loss
reserves. This paper adds an instance with an unusually direct benchmark. ASC 860-50
requires a Ginnie Mae issuer that acquires the unilateral right to repurchase a delinquent
loan to re-recognise it ``regardless of the Company's intention to repurchase'' it, so the
recognised balance is the unpaid principal of purchase options that have vested and have
\emph{not} been exercised --- an operating decision management never states. I benchmark it
against \nNDecisions{} loan-level exercise decisions from the programme's administrative
disclosure files, which measure the same quantity directly. The recognised amount tracks
the benchmark in ranking and magnitude, and the ratio between the two moves as lower
exercise implies. It also carries information the statements do not otherwise contain:
across filers the balance grew \pWmRepMin{} to \pWmRepMax{} times in 2020 while newly
vesting options grew only \pWmFlowMin{} to \pWmFlowMax{} times, and the ratio orders firms
by how far each withdrew from exercising. Comparability, however, fails. Nine filers
recognise the same quantity under the same paragraph in \pcpCaptions{} captions spanning
\pcpScopes{} scopes; only \pcpNet{} isolate the unexercised balance and one reports it
quarterly. Conditioning recognition on a contractual event rather than on intent produces
a measure with construct validity and a line item without comparability.
"""
t = t[:a] + NEWABS + t[b:]

# ------------------------------------------------------------------ keywords
t = t.replace(
    r"""\textit{Keywords}: recognition; derecognition; ASC 860; comparability; mortgage
servicing; disclosure informativeness.\quad
\textit{JEL}: M41, G21, G28""",
    r"""\textit{Keywords}: recognition; construct validity; comparability; transfers of
financial assets; mortgage servicing.\quad
\textit{JEL}: M41, M48, G21""")

# ------------------------------------------------------------------ introduction
i = t.index(r"\section{Introduction}")
j = t.index(r"\section{The recognition rule}")
NEWINTRO = r"""\section{Introduction}
%=====================================================================

A recognition rule specifies when an item belongs on the balance sheet. Whether the
resulting number measures the construct the rule has in mind is usually not testable,
because the construct is unobservable --- which is generally the reason it needed a rule.
Research on recognition therefore tends to ask whether recognised and disclosed amounts are
priced differently \citep{ahmedkiliclobo2006, mullerriedlsellhorn2015, michels2017}, taking
the market as the arbiter, rather than asking directly whether the recognised number is the
quantity the standard names.

A small literature does ask directly, by comparing a recognised estimate with an
independently realised outcome. Its home is property-casualty loss reserves, where
subsequently developed claims provide a benchmark against which the reported reserve can be
assessed \citep{petroni1992, beavermcnichols1998, petroni2000}. The benchmark there is
realised later and is itself an estimate for some time. This paper contributes an instance
with a sharper benchmark: the recognised amount is compared with contemporaneous,
transaction-level administrative records of the very events that trigger recognition.

\paragraph{Setting.} Under ASC 860-50, a Ginnie Mae issuer that acquires the unilateral
right to repurchase a delinquent loan out of a mortgage pool is deemed to have regained
effective control and must bring the loan back onto its balance sheet as an asset with an
offsetting liability. The recognition trigger is a contractual event --- three missed
payments --- and explicitly not management's intent. Mr.\ Cooper's disclosure states that it
recognises the asset and liability ``regardless of the Company's intention to repurchase the
loan''; Flagstar's says the same, ``regardless of whether the repurchase option has been
exercised.''\footnote{Mr.\ Cooper Group, Form 10-K for fiscal 2020, Note 9; Flagstar
Bancorp, Form 10-K for fiscal 2020, Note 5.}

Because recognition attaches when the option \emph{vests} rather than when it is
\emph{exercised}, the reported balance is the unpaid principal of options the issuer holds
and has not used. That is an operating decision --- whether to buy a delinquent loan back
out of a pool --- which management does not disclose, which the income statement does not
isolate, and which is otherwise unobservable to a financial statement user. Indifference to
intent is what makes the number informative: a preparer cannot shade it without either
repurchasing the loan or letting it cure.

\paragraph{Benchmark.} The same programme publishes administrative loan-level files
recording, for every Ginnie Mae loan reaching three months delinquent, the issuer of record
and the eventual disposition. I use \nNDecisions{} such decisions from December 2013 to
September 2020 to construct the unexercised balance directly, without reference to any
financial statement. This is the quantity ASC 860-50 recognises, measured independently of
the preparer.

\paragraph{Findings.} Three, corresponding to the hypotheses in
Section~\ref{sec:hyp}.

On construct validity, the recognised amount and the benchmark agree in ranking and in
order of magnitude. I deliberately do not lean on the level correlation, which is close to
mechanical when firm size spans two orders of magnitude. The informative comparison is the
ratio of the two, which is not constant and should not be: the benchmark cumulates over a
window while the balance sheet reports a stock at a date, so the ratio falls exactly when
fewer options are exercised and more of the vested balance remains outstanding at the
reporting date. It falls from about \pRatioPmOne{} to \pRatioPmTwo{} between 2019 and 2020.

On information content, the test requires no benchmark at all and could be run by a
statement user. If exercise behaviour were unchanged, the stock of unexercised options
would grow roughly in proportion to the flow of newly vesting ones. Across filers reporting
the balance net of repurchases, the flow grew by \pWmFlowMin{} to \pWmFlowMax{} times in
2020 and the balance by \pWmRepMin{} to \pWmRepMax{} times. The ratio between them orders
the firms by how far each withdrew from exercising, correlating \pWmCorr{} in logs with the
firm's own change in exercise rate, and falling below one for the single firm that exercised
\emph{more}.

On comparability, the standard fails. Nine filers recognise the same quantity under the same
paragraph, in \pcpCaptions{} distinct captions spanning \pcpScopes{} distinct scopes. Only
\pcpNet{} isolate the unexercised balance; three fold it into a wider aggregate from which
it cannot be recovered, and only \pcpQuarterly{} reports it quarterly. Because the standard
specifies recognition and is silent on presentation, every one of these treatments is
defensible, and a user comparing two filers on the face of the statements may be comparing
different quantities without being told.

\paragraph{Contribution.} The paper makes three. First, it extends external-benchmark
validation beyond loss reserves to a recognition rule, using administrative data rather than
subsequently realised outcomes, which removes the estimation horizon that complicates the
insurance setting. Second, it identifies a mechanism by which recognition produces
information that neither disclosure nor the income statement supplies: conditioning on a
contractual event makes a managerial decision observable as a by-product. This is a
different channel from the recognition-versus-disclosure reliability argument in
\citet{schipper2007}, where the distinction turns on how carefully an amount is prepared;
here the two would contain the same amount, and what recognition adds is that the amount
exists at all. Third, it quantifies a comparability cost of separating recognition
requirements from presentation requirements, in a setting where the correct application is
computable, so the cost can be stated rather than inferred
\citep{defranco2011, barth2012}.

The setting also connects to work on accounting for transfers of financial assets
\citep{landsman2008, dechow2010, dou2021}, which has largely asked whether transfers should
be treated as sales or borrowings. The question here is the converse: what a re-recognition
requirement reveals once the sale treatment lapses.

%=====================================================================
\section{Related literature and hypotheses}
\label{sec:hyp}
%=====================================================================

\subsection{Recognition, disclosure, and what a recognition trigger can reveal}

The recognition-versus-disclosure literature generally holds the underlying amount fixed and
asks whether placement on the face of the statements changes how it is used
\citep{ahmedkiliclobo2006, mullerriedlsellhorn2015, michels2017}. \citet{schipper2007} frames
the normative question in terms of reliability: recognised amounts are prepared with more
care and are subject to more assurance.

This setting differs. The amount would not exist under a disclosure regime conditioned on
intent, because the issuer does not report its exercise decisions and has no obligation to.
What produces the number is that the standard conditions on a contractual event the issuer
cannot control once the borrower stops paying. Recognition is doing something other than
improving reliability; it is generating an observation.

\paragraph{H1 (construct validity).} The amount recognised under ASC 860-50 corresponds to
the unpaid principal of vested and unexercised repurchase options, as measured independently
from administrative loan-level records.

\subsection{Information content without a market test}

Because the benchmark exists, information content can be assessed without inferring it from
prices. The relevant comparison is between the recognised stock and the flow of events that
generate it: if the underlying decision were unchanged, the two would grow together.

\paragraph{H2 (incremental information).} The growth of the recognised balance relative to
the growth of newly vesting options is informative about the issuer's exercise behaviour,
which is not otherwise disclosed.

\subsection{Comparability under a standard silent on presentation}

Comparability is usually assessed through the similarity of accounting mappings across firms
facing similar events \citep{defranco2011, barth2012}, because the correct mapping is
unknown. Here it is known: the standard defines the recognised quantity precisely. What it
does not define is where the quantity appears, at what level of aggregation, or whether the
exercised and unexercised portions are separated.

\paragraph{H3 (comparability).} Filers applying the same recognition requirement to the same
economic event report line items whose scope differs in ways that prevent a user from
recovering the recognised quantity for a subset of filers.

"""
t = t[:i] + NEWINTRO + t[j:]

# the old section 2 heading now duplicates the marker line above it
t = t.replace(r"""%=====================================================================
\section{The recognition rule}""",
              r"""%=====================================================================
\section{The recognition rule and the setting}""", 1)

open(p, "w", encoding="utf-8").write(t)
print("reframed: new abstract, keywords, introduction, hypotheses section")
print(f"citations now in the file: {t.count(chr(92) + 'cite')}")
