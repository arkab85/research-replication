"""Write the spatial results into the full-length journal paper: new section, revised abstract, intro, conclusion."""
import os, re, subprocess, shutil
H = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(H, "tex", "juefull"); TT = os.path.join(H, "tex", "tectonic.exe")
p = os.path.join(T, "main.tex"); s = open(p, encoding="utf-8").read(); open(p + ".bak2", "w", encoding="utf-8").write(s)
def rep(a, b, n=1):
    global s; assert s.count(a) >= 1, a[:90]; s = s.replace(a, b, n)

rep("\\input{numbers_full.tex}", "\\input{numbers_full.tex}\n\\input{numbers_geo.tex}\n\\input{numbers_geo2.tex}")

# ---------- abstract ----------
rep("Requests on government-backed loans, a statutory right, were unaffected. The break is the same across foreclosure regimes, regions and neighborhood types, and borrowers asking on either side of the date are alike. Those who asked just too late kept paying through the months forbearance would have paused, were less delinquent and less often modified six months later, and were not detectably worse off after a year, in rich and poor, dense and sparse, and majority-white and majority-minority neighborhoods alike. The paperwork cut take-up twelvefold without selecting needier recipients.",
    "Requests on government-backed loans, a statutory right, were unaffected, and borrowers asking on either side of the date are alike. Those who asked just too late kept paying through the months forbearance would have paused and were less delinquent and less often modified six months later. The screen fell equally on every kind of place: the discontinuity is the same across foreclosure regimes, regions, and rich and poor neighborhoods. Its consequences were not equal. A year later, refusal left borrowers in the most advantaged quartile of neighborhoods \\effHi{} points more likely to be performing and left those in the least advantaged quartile no better off (\\effLo{}). Paperwork rationed relief uniformly and distributed its burden unequally.")

# ---------- introduction ----------
rep("It is also the same everywhere: in judicial-foreclosure states and elsewhere, in each census region with enough loans to estimate it, with state fixed effects, and in zip codes above and below the median in income, population density, non-white share and blue-collar employment. It is a rule of the servicer and not a feature of any local housing market.",
    "It is also the same everywhere: in judicial-foreclosure states and elsewhere, in each census region with enough loans to estimate it, with state fixed effects, and across the distribution of neighborhood income, density, race and occupational structure. It is a rule of the servicer and not a feature of any local housing market.")
rep("Fourth, the paperwork did not select on need, in either of the senses the data allow. About a fifth of those turned away had told the agent that they had lost work or income, and the few who completed the application had reported a job loss no more often than those who had been approved on request (\\tgtJob{} against \\tgtJobEasy{} percent). And the consequences of refusal do not differ detectably between richer and poorer, denser and sparser, or whiter and less white neighborhoods.",
    "Fourth, the paperwork did not select on need. About a fifth of those turned away had told the agent that they had lost work or income, and the few who completed the application had reported a job loss no more often than those who had been approved on request (\\tgtJob{} against \\tgtJobEasy{} percent).\n\nThe fifth finding is the one a housing economist should care about most, and it qualifies the third. The screen itself was blind to place: the discontinuity in completion does not vary with any neighborhood characteristic I observe, and its interaction with an index of neighborhood advantage is \\ixfbI{} points (s.e.\\ \\ixfbSE{}). Its consequences were not blind to place. A year after the cutoff, the effect of refusal on performing status rises steeply with neighborhood advantage: \\effHi{} points (s.e.\\ \\effHiSE{}) in the top quartile of neighborhoods and \\effLo{} (s.e.\\ \\effLoSE{}) in the bottom, an interaction of \\ixperfor{} points per standard deviation (s.e.\\ \\ixperforSE{}) that survives randomization inference ($p=\\ixperforRI{}$), a wild cluster bootstrap ($p=\\ixWild{}$), loan controls interacted with the cutoff, state fixed effects and \\plN{} placebo cutoffs. The pattern is a property of the neighborhood and not of measured household income, which does not predict it. Where households had resources, forbearance was a convenience and refusing it cost them nothing; where they did not, forbearance was a bridge to a modification, and refusing it moved them toward the foreclosure pipeline rather than toward payment. A uniform administrative rule thus had a distributionally uneven effect, and the unevenness appears only at a horizon long enough for arrears to resolve.")

# ---------- replace the old neighborhood paragraph with a pointer ----------
a = s.index("\\paragraph{Neighborhoods} Forbearance policy"); b = s.index("\\begin{table}[t]\\centering\\caption{Consequences of asking after the cutoff, by neighborhood}")
c = s.index("\\end{table}", b) + len("\\end{table}")
s = s[:a] + ("\\paragraph{Neighborhoods} Whether the burden of the requirement fell evenly across places is the subject of Section~\\ref{sec:geo}. In brief: the screen did, and its consequences did not.\n") + s[c:]
rep("Together these say that the requirement reduced completion twelvefold, left the average screened-out borrower no worse off, did not visibly improve targeting on observed need, did not fall differently across neighborhoods, and may have harmed a minority, perhaps among owner-occupiers who had lost work, whom a sample of this size cannot isolate.",
    "Together these say that the requirement reduced completion twelvefold, left the average screened-out borrower no worse off, did not visibly improve targeting on observed need, and may have harmed a minority, perhaps among owner-occupiers who had lost work, whom a sample of this size cannot isolate.")

# ---------- the new section ----------
SEC = r"""
\section{The spatial incidence of the screen}\label{sec:geo}

A documentation requirement is set in a servicing centre and lands in neighborhoods. Two questions follow. Did the screen fall more heavily on some kinds of places than others? And did refusal have the same consequences everywhere? The answers differ, and the difference is the paper's main distributional result.

\subsection{A scattered portfolio}

The investor's book is spatially atomized. The \fsN{} conventional requests within fourteen days of the cutoff come from \geoZips{} distinct zip codes in \geoCtys{} counties: \geoPerZip{} loans per zip code, with \geoOneZip{} percent of zip codes contributing a single loan. The largest single county holds \geoMaxCty{} percent of the window and the zip-code Herfindahl index is \geoHHIz{} percent. Figure~\ref{fig:map} plots the requests. Borrowers who asked before and after the cutoff are interleaved throughout: of the \geoMixCty{} counties with at least three requests in the window, \geoMixBoth{} percent contain borrowers from both regimes.

Three things follow. First, the design is national, and the estimates cannot be confounded by any single local shock. Second, spatial correlation in the errors is unlikely to matter, which Section~\ref{sec:geoinf} confirms. Third, and substantively, a rule of this kind had little \emph{neighborhood-level} bite in this portfolio: a zip code with one affected loan does not experience a foreclosure cluster. The externality literature that motivates foreclosure-prevention policy (Campbell, Giglio, and Pathak 2011; Anenberg and Kung 2014; Gerardi et al.\ 2015) requires concentration, and a scattered specialist portfolio does not supply it. The same rule at a servicer with a geographically concentrated book would. What I can measure is the incidence across households by the kind of place they live in, which is what follows.

\begin{figure}[t]\centering\includegraphics[width=.92\textwidth]{fG_map.pdf}
\caption{Where the requests came from. Each point is a conventional loan whose borrower first asked for pandemic forbearance within fourteen days of 7 April 2020, plotted at the property's coordinates. Alaska and Hawaii are omitted from the figure and retained in the estimates.}\label{fig:map}\end{figure}

\subsection{The screen fell equally on every kind of place}

I measure neighborhoods at the property's zip code, matched for \geoMatchFull{} percent of the window. Beyond the usual characteristics I construct the share of local employment in occupations that cannot be performed from home, the neighborhoods where the pandemic's labor shock actually landed; it averages \flMean{} percent and runs from \flP10{} to \flP90{} percent between the tenth and ninetieth percentiles.

Table~\ref{tab:cont} interacts the discontinuity with each characteristic, standardized, one at a time. For the first stage, and for both six-month outcomes, nothing varies: the joint test that all eight interactions are zero has $p=\jointPmin{}$ or above for completion, delinquency and modification. A borrower's chance of obtaining forbearance fell by about three-quarters whether the property stood in a rich zip code or a poor one, a white one or a minority one, a dense one or a sparse one, one full of office workers or one full of frontline workers. Whatever else the servicer was doing on 7 April, it was not sorting on place.

\begin{table}[t]\centering\caption{The discontinuity interacted with neighborhood characteristics}\label{tab:cont}
\T{tG_cont.tex}
\tnote{Each row is a separate regression of the form $y = \alpha + \beta\,\text{post} + \gamma\,(\text{post}\times z) + \delta z + $ local linear terms, where $z$ is the standardized characteristic of the property's zip code; the table reports $\gamma$ in percentage points. Conventional loans within 14 days of 7 April 2020, triangular kernel. Occupations that cannot be done from home are those outside management, professional and administrative categories. Standard errors clustered by day of inquiry. With \nIntTests{} tests reported, the smallest Holm-adjusted $p$-value is \minIntHolm{}. * $p<.10$, ** $p<.05$, *** $p<.01$.}\end{table}

\subsection{Its consequences did not}\label{sec:geocons}

The twelve-month outcome behaves differently, and the characteristics that predict it are collinear with one another, so I summarize them in a single index rather than report eight correlated tests. The first principal component of seven zip-code characteristics---household income and college share entering positively, and non-white share, prior unemployment, frontline employment, renter share and vacancy entering negatively---explains \pcVar{} percent of their joint variation. I standardize it and call it \emph{neighborhood advantage}. Its bottom quintile averages \$\advPTenhinc{},000 in household income, \advPTennw{} percent non-white and \advPTencol{} percent college-educated; its top quintile averages \$\advPNinetyhinc{},000, \advPNinetynw{} percent and \advPNinetycol{} percent. The index is continuous at the cutoff (\advBal{} standard deviations, s.e.\ \advBalSE{}), so what follows is heterogeneity in the effect of refusal and not a change in who was asking.

Table~\ref{tab:index} interacts the discontinuity with the index, one test per outcome. Completion, September delinquency, modification and the foreclosure pipeline show nothing: the interactions are \ixfbI{}, \ixdqsepTwo{}, $0.4$ and \ixfcpath{} points per standard deviation, none distinguishable from zero under randomization inference. Performing status in April 2021 shows a great deal: \ixperfor{} points per standard deviation (s.e.\ \ixperforSE{}), with a randomization-inference $p$-value of \ixperforRI{} and a wild cluster bootstrap $p$-value of \ixWild{}. Evaluated across the distribution, the effect of asking a day too late on the probability of performing a year later is \effLo{} points (s.e.\ \effLoSE{}) at the mean of the bottom quartile of neighborhoods and \effHi{} (s.e.\ \effHiSE{}) at the mean of the top. Figure~\ref{fig:index} shows both halves of the result together: a flat screen and a steep gradient in what the screen did.

\begin{figure}[t]\centering\includegraphics[width=.95\textwidth]{fG_index.pdf}
\caption{The screen was uniform; its consequences were not. Differences across 7 April 2020 in the probability of being granted forbearance (left) and of performing in April 2021 (right), estimated separately within quartiles of the neighborhood advantage index and plotted at the quartile means. Bars are 95 percent confidence intervals, standard errors clustered by day of inquiry.}\label{fig:index}\end{figure}

\begin{table}[t]\centering\caption{The discontinuity interacted with neighborhood advantage}\label{tab:index}
\T{tG_index.tex}
\tnote{Conventional loans within 14 days of 7 April 2020. Neighborhood advantage is the standardized first principal component of seven zip-code characteristics (Section~\ref{sec:geocons}). Each row is one regression; the first two columns report the jump at the cutoff at the mean of the index and the last four its interaction with the index, in percentage points. RI $p$: randomization inference reassigning the post-cutoff indicator across whole days, 4,999 draws. Controls are February-2020 delinquency, non-performing pool, loan-to-value ratio, log balance, tenure, prior contact rates and the prior hardship flag. Standard errors clustered by day.}\end{table}

\subsection{What happened instead}

Where the divergence comes from is visible in the composition of outcomes. Table~\ref{tab:gdispo} reports April-2021 status across the cutoff, separately for neighborhoods below and above the median of the index. In more advantaged neighborhoods, refusal raised performing status by \hiperf{} points and cut modifications by \himodi{}: those borrowers simply kept paying and never needed their arrears capitalized. In less advantaged neighborhoods, refusal raised performing status by only \loperf{} points and cut modifications by \lomodi{}; the difference went instead into a notice of intent (\lonoti{} points) and the foreclosure pipeline (\lofore{}), neither individually distinguishable from zero. The contrast in performing status and in modifications is significant ($p<0.01$ for both); the contrast in the foreclosure categories is not, and I do not claim it.

The reading I favor is the ordeal logic of Section~\ref{sec:frame} with a liquidity interpretation. In neighborhoods where households had resources, a request for forbearance revealed a preference for liquidity; refusing it cost those households nothing, and spared them a modification. In neighborhoods where households did not, the request revealed an inability to pay; refusing it did not make them pay, and moved them a step closer to losing the home. The screen could not tell the two apart, because it screened on the capacity to complete an application rather than on need.

\begin{table}[t]\centering\caption{April 2021 status across the cutoff, by neighborhood advantage}\label{tab:gdispo}
\T{tG_dispo.tex}
\tnote{Conventional loans within 14 days of 7 April 2020, split at the median of the neighborhood advantage index. Before and After are shares in each status among borrowers asking in the fourteen days before and after 7 April; Diff.\ is the difference in means. The final column is the $p$-value on the interaction of the post-cutoff indicator with an indicator for the less advantaged half, in a pooled local linear regression. Percentages and percentage points; standard errors clustered by day.}\end{table}

\subsection{A neighborhood pattern, or household resources measured twice?}

Zip-code averages are a coarse proxy for the household, and the commercial consumer file supplies a modeled household income for \hhMatch{} percent of the window. The two are only weakly related here (correlation \corrHhAdv{}), which is what one expects in a portfolio of distressed loans scattered across many markets. Entered alone, household income interacted with the cutoff gives \hhAlone{} points (s.e.\ \hhAloneSE{}), indistinguishable from zero. Entered together with the neighborhood index, household income falls to \hhBoth{} (s.e.\ \hhBothSE{}) and the neighborhood index retains \advBoth{} (s.e.\ \advBothSE{}). The gradient is a property of the place, not of the modeled income of the borrower, though modeled income is measured with enough error that this comparison favors the better-measured variable and should not be pressed further.

\subsection{Robustness}\label{sec:georob}

Appendix Table~\ref{tab:grobust} collects the checks. The interaction is \ixperfor{} points at the fourteen-day bandwidth and ranges from $14.1$ at seven days to $9.1$ at twenty-eight; it is \ixDonut{} excluding 6 and 7 April, \ixCtrlX{} (s.e.\ \ixCtrlXSE{}) with loan characteristics interacted with the cutoff, and \ixStateFE{} (s.e.\ \ixStateFESE{}) with state fixed effects. Of \plN{} placebo cutoffs far enough from 7 April that their windows exclude it, none produces a significant interaction and the largest in absolute value is \plMax{} points. Shifts of seven days are not reported because a fourteen-day window around them still contains the true cutoff.

Three caveats bound the claim. The outcome is the one whose \emph{average} effect is imprecise, and the paper does not treat the average twelve-month effect as a finding; the interaction is estimated far more precisely than the average, which is common when an average masks offsetting effects but deserves to be stated plainly. The components of the index are correlated, so the paper identifies a single gradient and not the contribution of race, income or occupation separately. And the horizon ends in April 2021, under foreclosure moratoria, so what the gradient predicts about completed foreclosures is unknown.
"""
rep("\\section{The asking margin}\\label{sec:asking}", SEC + "\n\\section{The asking margin}\\label{sec:asking}")

# ---------- inference subsection: Conley etc. ----------
rep("Estimates are stable across bandwidths from five to twenty-eight days (Appendix Figure~\\ref{fig:bw}) and under reweighting (Appendix Table~\\ref{tab:ipw}).",
    "Estimates are stable across bandwidths from five to twenty-eight days (Appendix Figure~\\ref{fig:bw}) and under reweighting (Appendix Table~\\ref{tab:ipw}).\n\n\\subsubsection*{Spatial correlation}\\label{sec:geoinf}\nBecause the portfolio is scattered, spatial dependence is a small concern, and Appendix Table~\\ref{tab:ginfer} confirms it. Clustering by county, two-way clustering by day and county, and Conley spatial standard errors with Bartlett kernels at 100 and 250 kilometres all leave the standard errors close to their day-clustered values; the first stage, for example, has a standard error of \\fsMainSE{} clustered by day and \\cnlfb{} under Conley at 100 kilometres. County fixed effects leave the first stage at \\ctyfb{} (s.e.\\ \\ctySEfb{}) but cannot identify the outcome effects, since \\geoCtys{} counties hold \\fsN{} loans and only \\geoMixCty{} counties contain borrowers from both regimes.")

# ---------- conclusion ----------
rep("The households that asked a day too late were, six months on, less delinquent and less often modified than those who asked in time, and a year on were no worse off that the data can detect; those who cleared the paperwork were no needier than those who had been waved through. Whether the same holds for borrowers less seasoned in distress than these, and over horizons long enough to see who kept the home, is the question that the next data should answer.",
    "The households that asked a day too late were, six months on, less delinquent and less often modified than those who asked in time; those who cleared the paperwork were no needier than those who had been waved through. The screen was blind to place, and what it did was not. A year later, refusing relief had left borrowers in the most advantaged quartile of neighborhoods \\effHi{} points more likely to be performing and borrowers in the least advantaged quartile no better off, because in the first kind of place forbearance was a convenience and in the second it was a bridge. That is the sense in which an ordeal is a blunt instrument: it rations uniformly and it does not know what it is rationing. Whether the same holds for borrowers less seasoned in distress than these, and over horizons long enough to see who kept the home, is the question that the next data should answer.")

# ---------- appendix additions ----------
rep("\\begin{table}[h]\\centering\\caption{Bias-corrected robust inference}\\label{tab:cct}",
    """\\begin{table}[h]\\centering\\caption{Robustness of the neighborhood gradient}\\label{tab:grobust}
\\T{tG_robust.tex}
\\tnote{Dependent variable: performing in April 2021. Conventional loans; local linear with a triangular kernel; the interaction is with the standardized neighborhood advantage index. Controls are February-2020 delinquency, non-performing pool, loan-to-value ratio, log balance, tenure, prior contact rates and the prior hardship flag; the last row also interacts delinquency, loan-to-value and log balance with the cutoff. Percentage points; standard errors clustered by day.}\\end{table}

\\begin{table}[h]\\centering\\caption{Household income and neighborhood advantage}\\label{tab:ghh}
\\T{tG_hh.tex}
\\tnote{Dependent variable: performing in April 2021. Household income is a modeled category from a commercial consumer file, standardized; neighborhood advantage is the zip-code index. Conventional loans within 14 days of 7 April 2020 with both measures. Percentage points; standard errors clustered by day.}\\end{table}

\\begin{table}[h]\\centering\\caption{Standard errors under alternative assumptions about dependence}\\label{tab:ginfer}
\\T{tG_infer.tex}
\\tnote{Conventional loans within 14 days of 7 April 2020, local linear with a triangular kernel. The Estimate column is the baseline; the next five columns report standard errors for that estimate under clustering by day of inquiry, by county, two-way by day and county, and Conley spatial HAC with a Bartlett kernel at 100 and 250 kilometres combined with full within-day dependence. The last two columns add county fixed effects, which the design supports for the first stage only.}\\end{table}

\\begin{table}[h]\\centering\\caption{Neighborhood characteristics at the cutoff}\\label{tab:gbal}
\\T{tG_balance.tex}
\\tnote{Local linear, 14-day bandwidth, triangular kernel, standard errors clustered by day; units are the characteristic's own units.}\\end{table}

\\begin{table}[h]\\centering\\caption{Bias-corrected robust inference}\\label{tab:cct}""")
open(p, "w", encoding="utf-8").write(s)
r = subprocess.run([TT, "main.tex"], cwd=T, capture_output=True, text=True)
print("compile:", "ok" if r.returncode == 0 else r.stderr[-1200:])
