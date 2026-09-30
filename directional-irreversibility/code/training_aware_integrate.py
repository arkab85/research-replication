from pathlib import Path
import re,json
S=Path(__file__).resolve().parent;P=S.parent;s=(S/'main.tex').read_text()
# Move the detailed earlier diagnostics to the companion, preserving all evidence.
a=s.index(r'\subsection{A persistent-process diagnostic at the financial sample allocation}')
b=s.index(r'\section{A reproducible financial application}',a)
supp=s[a:b];s=s[:a]+s[b:]
marker=r'\section{An optional decision-use illustration}'
s=s.replace(marker,(S/'training_aware_theory.tex').read_text()+'\n'+r'\section{Supplementary diagnostic designs}\label{app:diagnostics}'+'\n'+supp+'\n'+marker)
a=s.index(r'\subsection{Reducing sensitivity to fitted conditional means}')
b=s.index(r'\section{Simulation evidence:',a)
main=r'''
\subsection{Keeping training uncertainty in the original DII}\label{sec:training}
A fitted financial exposure is uncertain because its conditional-mean coefficients were learned from a finite training period. Freezing those coefficients when resampling the evaluation period omits that uncertainty. The practical route developed here retains the original DII point estimator and propagates uncertainty from both periods.

For correctly specified fixed-dimensional mean models, write $\mathcal A$ for the derivative of the covariance operator with respect to regression coefficients, $G$ for the joint evaluation-operator fluctuation, and $Z$ for the joint training-coefficient fluctuation. With $n/m\to\lambda\in(0,\infty)$, Appendix~\ref{app:training} proves
\[
\sqrt n(\widehat C-C)\Rightarrow G-\sqrt\lambda\mathcal A Z.
\]
An independent training-block draw and evaluation-block draw approximate these two terms; common draws across directions and horizons preserve their covariance within each period. The bootstrap calibrates squared operator fluctuations at the all-zero null and the linear contrast at a regular law. Their intersection retains the stated pointwise guarantee on the covered null strata. The training and evaluation periods must be independent or sufficiently separated for the specified coupling argument.

This route allows comparable training and evaluation sizes without changing the observed DII or learning a conditional derivative function. It requires both conditional means to belong to the prespecified finite-dimensional bases. It does not make an arbitrary polynomial correct, account automatically for sample-derived scaling, or cover an increasing basis dimension. The contribution is the joint implementation for fitted DII across its two null regimes, not the general principle of propagating parameter uncertainty.

An alternative orthogonal estimator preserves the same population target and has a second-order nuisance-rate condition, but it changes the sample estimator and influence process. The completed attribution audit finds substantial power loss even with population nuisance information. Its theory and full experiments remain in Appendices~\ref{app:orthogonal} and \ref{app:diagnostics}; it is not the preferred small-sample implementation on the present evidence.

'''
s=s[:a]+main+s[b:]
r=json.loads((P/'training_aware_study_results.json').read_text());assert r['datasets']==800 and r['seed']==20260917
expected={(0.,'weak'):.29,(.6,'weak'):.33,(0.,'strong'):.78,(.6,'strong'):.75,(0.,'equal_positive'):.07,(.6,'equal_positive'):.07}
assert all(x['rate']==expected[(x['rho'],x['design'])] for x in r['summary'] if x['method']=='training_aware' and (x['rho'],x['design']) in expected)
methods=['original_nbb','evaluation_only_linearized','training_aware','orthogonal'];rows=[]
for rho in (0.,.6):
 for design,label in [('double','Double zero'),('equal_positive','Equal positive'),('weak','Weak exposure'),('strong','Strong exposure')]:
  d={x['method']:x for x in r['summary'] if x['rho']==rho and x['design']==design}
  rows.append(f'{rho:.1f} & {label} & '+' & '.join(f"{100*d[k]['rate']:.0f}" for k in methods)+r' \\')
new=r'''
\subsection{Training-aware calibration: new-seed validation}
A new 800-dataset validation study uses 204 training observations, 120 evaluation observations, a 39-row gap, fixed population scales, and 399 bootstrap draws. Four prespecified designs are run at persistence parameters zero and 0.6, with 100 replications in each cell. The double-zero and positive-exposure designs use the polynomial family in Appendix~\ref{app:diagnostics}. For the equal-positive null, let $A_t,B_t$ be independent stationary Gaussian AR(1) sequences with the same persistence parameter and unit variance, and $c_t$ an independent iid sign. Set $X_t=(1+0.65c_t)A_t$ and $Y_t=(1+0.65c_t)B_t$, and include $c_t$ in both regressor vectors. Both conditional means are zero. Exchangeability gives equal HSIC components, and scale dependence on $c_t$ makes both positive. The fitted basis is $(1,z,z^2,c,zc)$, with no redundant columns.

All comparisons share the same data and mean fits. The original nonoverlapping procedure uses the centered recomputed paired statistic; the evaluation-only control uses a linearized regular statistic but omits training uncertainty; the training-aware version includes that uncertainty. The orthogonal estimator is retained as a comparator. Thus improvements relative to the original procedure reflect the combined calibration change; including training uncertainty alone is not claimed to increase power.

\begin{table}[htbp]\centering
\caption{New-seed validation: intersection rejection percentages}
\begin{tabular}{rlrrrr}\toprule
$\rho$ & Design & Original & Eval. only & Training-aware & Orthogonal\\\midrule
% ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent All methods use nonoverlapping blocks of four rows. ``Eval. only'' omits training uncertainty. Each cell has 100 replications. Equal-positive training-aware rejection of 7\% has a Wilson 95\% interval of approximately [3.4\%,13.8\%]; the experiment does not establish exact five-percent size.
\end{table}

Weak-exposure detection rises from 18\% to 29\% without persistence and from 22\% to 33\% with persistence. Paired Monte Carlo intervals for these 11-percentage-point differences are approximately [3.7,18.3] and [2.7,19.3] percentage points. They describe simulation uncertainty within the prespecified designs, not a general superiority result. Strong-exposure differences are three and minus four percentage points, with paired intervals containing zero. The validation therefore supports a limited practical advantage at weak exposures while retaining strong-exposure performance broadly comparable to the original benchmark.

Omitting training uncertainty produces considerably higher rejection under the positive exposures. That is not a free gain: it removes a term required by the comparable-sample asymptotic expansion. Under the equal-positive null its rejection frequencies are 8\% and 7\%, compared with 7\% and 7\% for the training-aware procedure. These small cells cannot establish a broad ranking of size distortion. The case for including training uncertainty comes from the sampling expansion, not from interpreting every rejection of an unknown empirical null as a false positive.

\subsection{Why the correction is retained as an extension}
A separate attribution audit replays the preceding 600 orthogonal-diagnostic datasets using known conditional means and numerically integrated population derivatives. At the strong exposure, uncorrected known-mean rejection is 90\% and 87\%, compared with 34\% and 22\% after correction. Estimating the derivative introduces further costs, but cannot explain the entire loss. The conditional-derivative integrations use 512 quadrature nodes and agree with 768-node calculations within $2\times10^{-14}$ over the audited signal range. These are synthetic population-information benchmarks, not empirically available estimators. Full designs, results and numerical checks are preserved in the companion and reproducibility package.

'''.replace('% ROWS','\n'.join(rows))
s=s.replace(r'\section{A reproducible financial application}',new+r'\section{A reproducible financial application}')
f=json.loads((P/'finance/training_aware_results.json').read_text());assert not any(x['p_intersection']<=.05 for x in f['comparisons'].values())
extra=r'''
\paragraph{Secondary training-aware audit.} Applying the training-aware procedure to the same 12 fixed cells yields no unadjusted intersection rejection. Original DII point estimates are retained. The analysis is explicitly post-results and does not replace the primary protocol. The correctly specified finite-basis condition and joint dependence are unverified in these data; moreover training-derived scaling is not included in the additional coefficient-only expansion. The output is a working diagnostic, not a theorem-certified financial finding.

'''
s=s.replace(r'\section{Financial transmission and the use of DII}',extra+r'\section{Financial transmission and the use of DII}')
abstract="""A shock can move an outcome, change its risk, or generate an asymmetric relationship between opposing dynamic models. The Directional Irreversibility Index (DII) separates this directional structure from mean responses and general dependence. Financial benchmarks show why positive DII can coexist with a zero linear response and why a persistent footprint need not imply delayed information absorption. Inference must account for fitted residuals, dependent horizon comparisons, and null configurations with different convergence rates. We develop a training-aware block bootstrap that preserves the original DII estimator and propagates regression uncertainty when training and evaluation samples are comparable, under correctly specified finite-dimensional mean models and explicit dependence conditions. Simultaneous component bounds distinguish directional asymmetry from approximate model validity. An orthogonal alternative relaxes nuisance-rate requirements but has substantial small-sample power costs. A new 800-dataset validation shows improved weak-exposure detection relative to the original nonoverlapping procedure in the studied designs, without establishing general superiority or exact size control. A fixed four-market application has no primary directional rejection. The contribution is a scoped framework for measuring and inferring directional economic dynamics while separating statistical footprints, model adequacy, and financial mechanisms."""
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',s,flags=re.S)
s=s.replace('including an orthogonal correction that relaxes the regression-error requirement','including a training-aware bootstrap that retains the original estimator at comparable sample sizes')
s=s.replace('Both the bounds and the implemented orthogonal correction have substantial small-sample costs, which are measured rather than assumed away.','The training-aware implementation improves weak-exposure detection in a new-seed validation; simultaneous bounds and orthogonal correction have substantial small-sample costs. These are distinct results with distinct assumptions.')
s=s.replace('including a correction that reduces its sensitivity to regression estimation error','including explicit propagation of training uncertainty into the original estimator')
s=s.replace('Orthogonal estimation further replaces first-order sensitivity to fitted means with a second-order product-rate condition, under the stated moment and learning assumptions.','For correctly specified fixed-dimensional means, training-aware calibration retains the original estimator and includes coefficient uncertainty at comparable sample sizes. Its new-seed validation improves weak-exposure detection in the studied designs. Orthogonal estimation remains a separate extension with a relaxed nuisance-rate condition and substantial small-sample costs.')
(S/'main.tex').write_text(s);(P/'abstract.txt').write_text(abstract)
print('Training-aware result and attribution evidence integrated; abstract words',len(abstract.split()))
