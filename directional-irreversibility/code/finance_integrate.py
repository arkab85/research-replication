from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent.parent;S=P/'source';F=P/'finance'
r=json.loads((F/'financial_results.json').read_text())
assert r['protocol_sha256']==hashlib.sha256((F/'analysis_protocol.md').read_bytes()).hexdigest()
assert len(r['primary_tests'])==12 and r['evaluation_n']==120
nrej=sum(x['p_intersection']<=.05 for x in r['primary_tests']);nwill=sum(x['p_wild']<=.05 for x in r['primary_tests'])
assert nrej==0 and nwill==7 # Guard fixed numerical interpretation when data/results change.
labels={'FX':'Yen/USD','Treasury':'Treasury yield','Credit':'Credit spread','VIX':'VIX'}
s=(S/'main.tex').read_text()
rows=[]
for x in r['primary_tests']:
 rows.append(f"{labels[x['asset']]} & {x['h']} & {1000*x['dii']:.3f} & {x['p_wild']:.3f} & {x['p_paired']:.3f} & {x['p_intersection']:.3f} & {x['p_holm']:.3f} "+r'\\')
pr=[]
for x in r['predictive_comparisons']:
 pr.append(f"{labels[x['asset']]} & {x['h']} & {x['quadratic_gain_percent']:.3f} & [{x['gain_ci_low']:.3f}, {x['gain_ci_high']:.3f}] "+r'\\')
sec=r'''\section{A reproducible financial application}\label{sec:application}
\subsection{Question, sources, and fixed design}
Does an observed monetary-policy surprise retain directional residual asymmetry in a later financial change, and is that asymmetry accompanied by useful nonlinear response prediction? We investigate this question in four markets using the monthly monetary-policy component published by Jaroci\'nski and Karadi, combined with public FRED series. The shock is the authors' median-rotation MP proxy. The outcomes are the yen per US dollar exchange rate, the ten-year Treasury yield, the Baa--Aaa corporate yield spread, and the VIX. Data links are \url{https://github.com/marekjarocinski/jkshocks_update_fed} and the FRED series pages for \texttt{DEXJPUS}, \texttt{GS10}, \texttt{BAA}, \texttt{AAA}, and \texttt{VIXCLS}. The replication package records source URLs, retrieval dates, and file hashes. The shock construction uses high-frequency interest-rate and equity surprises; the present study does not re-estimate that upstream decomposition.

The protocol was written before this application's estimation, after the methodological development and earlier simulations. It is an internally fixed retrospective design, not an externally preregistered study. Training origins run from January 1991 through December 2007, a gap covers calendar 2008, and the 120 evaluation origins run from January 2009 through December 2018. Outcomes through June 2019 supply the longest horizon. No observation after 2019 enters the application. The data vintage is current at retrieval and the upstream shock construction is not recursive, so this is a historical held-out exercise, not a real-time forecast.

For exchange rates and VIX, the financial change is 100 times the log change in the last available daily observation of each month. For the Treasury yield and credit spread, it is the change in the monthly series, in percentage points. The target at $h=1,3,6$ is the single-month financial change at $t+h$, excluding the announcement month. Monthly-average yield data and month-end exchange-rate/VIX data measure different intervals; these definitions are held fixed and no pooled cross-market DII is constructed. Zero-shock months remain in the calendar. All chosen training and evaluation rows are complete.

Both representations condition on the previous month's shock and the previous month's financial change. Total-degree-two polynomial OLS fits the two conditional means, with transformations learned on training rows and frozen. Bandwidth-one Gaussian kernels use these standardized variables and residuals. The primary bootstrap uses 12-month blocks, 999 draws, and a common path across each market's horizons. Fixed sensitivities use six-month blocks and, separately, cubic conditional-mean fits. All 12 primary tests are reported, together with Holm adjustment. Prespecified averages, hump contrasts, and persistence tests are secondary outputs in the replication files.

\subsection{Directional evidence and calibration sensitivity}
Table~\ref{tab:finance} reports the complete primary family. No intersection test rejects at five percent, even before adjustment. The smallest intersection p-value is 0.064 for VIX at six months, followed by 0.078 for the Treasury yield at three months. The wild component alone has seven p-values below 0.05. The difference illustrates calibration sensitivity in actual financial data; without knowledge of the population restrictions, it does not establish that these seven are false positives.

\begin{table}[htbp]\centering\small
\caption{Later financial changes: complete primary DII results}\label{tab:finance}
\begin{tabular}{lrrrrrr}\toprule
Market & $h$ & $1000\widehat D_h$ & Wild $p$ & Paired $p$ & Intersection $p$ & Holm $p$\\\midrule
% DIIR
\bottomrule\end{tabular}
\par\vspace{4pt}\raggedright\noindent All cells use 204 training origins and 120 held-out origins. DII is in fixed standardized kernel units, not return units. Holm adjustment covers all 12 primary intersection tests. Bootstrap interpretation is conditional on the working assumptions discussed in the text.
\end{table}

Changing the block length to six months yields no intersection rejection at five percent. Cubic fits give three unadjusted p-values below five percent, but none survives Holm adjustment within that 12-test sensitivity family. This dependence on nuisance specification is evidence against selecting a favorable polynomial degree to support a directional headline. It also prevents reading the primary null results as proof that all population contrasts vanish.

\subsection{An independent prediction comparison}
To assess an economic use beyond the diagnostic's own statistic, fit three frozen conditional-mean models: history alone, history plus the current shock, and history plus the shock and its square. They share the two lagged controls. The reported gain compares the quadratic-shock model with the linear-shock model on held-out mean squared error:
\[
G=100\left(1-\frac{\operatorname{MSE}_{\rm quadratic}}{\operatorname{MSE}_{\rm linear}}\right).
\]
Positive values favor the additional nonlinear term. The interval resamples paired prediction errors in 12-month circular blocks using 1,999 draws; it is conditional on the frozen training fits and does not include uncertainty from upstream shock construction. The predictions and DII target different features of the distribution, so success on one is not a mathematical requirement for success on the other.

\begin{table}[htbp]\centering
\caption{Held-out prediction: adding a quadratic shock term}\label{tab:financeprediction}
\begin{tabular}{lrrl}\toprule
Market & $h$ & MSE gain (\%) & Approx. 95\% block interval\\\midrule
% PREDR
\bottomrule\end{tabular}
\par\vspace{4pt}\raggedright\noindent Gains are percentages of linear-model MSE. These are retrospective prediction comparisons, not investment returns. Intervals are pointwise approximations, not simultaneous confidence statements.
\end{table}

No positive gain has an interval excluding zero. The largest estimated improvement is about 0.344 percent for yen/USD at six months, with an interval that includes a loss. Several gains are negative. This simple nonlinear extension therefore supplies no demonstrated positive predictive benefit in the chosen holdout. It is not a comparison against every nonlinear forecasting model, and it cannot rule out nonlinear distributional effects beyond the conditional mean.

\subsection{What this application adds and leaves open}
The application establishes a transparent, source-traceable implementation and an independent predictive benchmark. It does not establish a new persistent shock-transmission finding. The value of the completed exercise is that a tempting directional story must survive joint calibration, a fixed primary family, nuisance sensitivity, and a separate test of its proposed economic use.

The theoretical conditions also remain consequential. Exact finite dependence is not established for these financial series. The evaluation lag-one autocorrelations of credit-spread and VIX changes are about 0.51 and $-0.27$, respectively. Such diagnostics neither prove nor disprove a particular finite-memory model. Nor do the 204 training observations establish the sufficient asymptotic first-stage rate. As a result, the block p-values are empirical diagnostics under maintained working approximations, rather than a claim that the theorem has been fully verified for the application. External shock identification does not remove these sampling issues or automatically confer an independent-noise representation on the observed proxy.

These qualifications mean that the exercise cannot be used to declare the journal companion's empirical conclusions disproved: its series, calendar, estimator, and tests differ. It also cannot be used to strengthen those conclusions by importing isolated favorable sensitivity results. A substantive finance contribution would require stable incremental evidence and a sampling argument appropriate to the actual data process.

'''.replace('% DIIR','\n'.join(rows)).replace('% PREDR','\n'.join(pr))
s=s.replace('\\section{Financial transmission and the use of DII}',sec+'\\section{Financial transmission and the use of DII}')
s=s.replace('Section 5 develops the financial research use and interpretation of horizon profiles. Section 6 concludes.','Section 5 reports the new retrospective financial application. Section 6 develops the financial research use and interpretation of horizon profiles. Section 7 concludes.')
s=s.replace('The empirical evidence in this paper consists of controlled simulations and analytical models. No institutional dataset or financial-return result is claimed. A reproducible financial application remains an important opportunity to demonstrate economic relevance beyond those models.','The empirical application provides a reproducible test of the proposed financial use, with null primary results and no demonstrated positive predictive gain. These results delimit what this particular implementation supports; they do not establish an absence of economic shock transmission.')
s=s.replace('The simulations make the calibration failures and conservatism explicit.','The simulations make the calibration failures and conservatism explicit. The financial application demonstrates calibration and nuisance sensitivity on observed data, while providing no primary rejection or clearly positive held-out predictive gain.')
# Add the new empirical evidence explicitly without rewriting a null as success.
start=s.index('\\begin{abstract}');end=s.index('\\end{abstract}')
abstract=s[start+len('\\begin{abstract}'):end].strip()
abstract+=' A fixed four-market financial application yields no primary DII rejection and no clearly positive held-out gain from a quadratic shock term, exposing the gap between suggestive directional estimates and substantiated financial use.'
s=s[:start]+'\\begin{abstract}\n'+abstract+'\n'+s[end:]
(S/'main.tex').write_text(s);(P/'abstract.txt').write_text(abstract)
print('Financial section integrated; abstract words',len(abstract.split()))
