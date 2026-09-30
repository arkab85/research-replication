from pathlib import Path
import json,re
S=Path(__file__).resolve().parent;P=S.parent;s=(S/'main.tex').read_text();r=json.loads((P/'decision_use_study_results.json').read_text());assert r['datasets']==2400 and r['seed']==20260919
expected={(120,0.,'Laplace'):(17,107),(120,.6,'Laplace'):(3,72),(480,0.,'Laplace'):(170,187),(480,.6,'Laplace'):(87,186),(120,0.,'Moment_matched'):(2,0),(120,.6,'Moment_matched'):(1,3),(480,0.,'Moment_matched'):(15,0),(480,.6,'Moment_matched'):(1,0)}
assert all((x['dii']['rejections'],x['reverse']['rejections'])==expected[(x['n'],x['rho'],x['law'])] for x in r['summary'] if x['law']!='Gaussian')
s=s.replace(r'\section{An optional decision-use illustration}',(S/'decision_use_theory.tex').read_text()+'\n'+r'\section{An optional decision-use illustration}')
# An explicit economic use and a distinct population comparison, without claiming novelty for LiNGAM.
a=s.index(r'\subsection{Persistent DII need not mean delayed incorporation}')
new=r'''
\subsection{The same mean and variance can conceal different directional structure}\label{sec:momentmatch}
Consider the financial modeling decision of whether shocks can be represented using disturbances whose distribution is invariant to the conditioning variables. Accurate mean and risk forecasts do not by themselves settle that question. Nor does positive DII establish forward independence: both dependence components and a substantively justified forward-model tolerance must be considered.

Let $X=A$ and $Y=A+B$, where $A,B$ are iid symmetric innovations with mean zero and variance one. In every case $\E[Y\mid X]=X$, $\operatorname{Var}(Y\mid X)=1$, and $\E[X\mid Y]=Y/2$. Gaussian innovations give DII zero. Unit-variance Laplace innovations give positive DII because the reverse residual $(A-B)/2$ depends on $Y$. Thus the forward mean and variance functions can remain the same while directional structure differs. This uses the established linear non-Gaussian insight of Shimizu et al. (2006); it is an interpretation of DII, not a new identification theorem.

A more demanding illustration mixes unit-variance Laplace innovations with probability $2/7$ and unit-variance uniform innovations otherwise. Its fourth moment equals the Gaussian value of three, but its sixth moment is $1395/49$. In this model,
\[
\operatorname{Cov}\!\left(\{(A-B)/2\}^2,Y^2\right)=0,
\quad
\operatorname{Cov}\!\left(\{(A-B)/2\}^2,Y^4\right)=330/49>0.
\]
DII is positive although that particular reverse squared-moment diagnostic is zero. The full proof and a persistent-process counterpart are in Appendix~\ref{app:moments}. The examples do not share identical predictor marginals or joint distributions. They also do not defeat every variance diagnostic: a richer reverse variance model can detect the dependence. Section~\ref{sec:momentvalidation} evaluates the finite-sample cost of using the broader kernel comparison.

'''
s=s[:a]+new+s[a:]
rows=[]
for x in r['summary']:
 label={'Gaussian':'Gaussian','Laplace':'Laplace','Moment_matched':'Matched fourth moment'}[x['law']]
 rows.append(f"{x['n']} & {x['rho']:.1f} & {label} & {100*x['dii']['rate']:.1f} & {100*x['forward']['rate']:.1f} & {100*x['reverse']['rate']:.1f}"+r' \\')
validation=r'''
\subsection{What does DII add beyond a simpler moment diagnostic?}\label{sec:momentvalidation}
A new matched experiment compares the Gaussian-kernel DII with forward and reverse diagnostics of $\operatorname{Cov}(u^2,z_1^2)$, using the three innovation laws from Section~\ref{sec:momentmatch}. Both directional conditional means are affine and correctly specified. All methods use the same trained normalization and residuals and propagate coefficient, scale and centering uncertainty. Across evaluation sizes 120 and 480, persistence zero and 0.6, and three laws, the experiment contains 2,400 datasets, with 200 replications per cell and 399 bootstrap draws. Appendix~\ref{app:moments} gives the full design and moment-calibration assumptions.

\begin{table}[htbp]\centering
\caption{Matched diagnostic rejection percentages}
\begin{tabular}{rrlrrr}\toprule
$n$ & $\rho$ & Innovation law & DII & Forward moment & Reverse moment\\\midrule
% ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent DII is zero for Gaussian innovations and positive for the other two laws. The forward moment is zero throughout; the reverse moment is positive only for Laplace innovations. These procedures test different population quantities. Full Wilson intervals are archived; 200 replications do not establish exact size or universal power rankings.
\end{table}

The focused reverse-moment diagnostic is substantially more powerful for Laplace innovations. At $n=120$ its rejection rates are 53.5\% and 36.0\%, versus 8.5\% and 1.5\% for DII. At $n=480$ the corresponding rates are 93.5\% and 93.0\%, versus 85.0\% and 43.5\%. A known moment restriction should therefore not be replaced by DII on a claim of demonstrated power superiority.

The matched-fourth-moment model establishes a broader population distinction, but its finite-sample evidence is weak. DII rejection is 1.0\% and 0.5\% at $n=120$, rising to 7.5\% and 0.5\% at $n=480$. The Wilson interval for the 7.5\% cell is approximately [4.6\%,12.0\%]. The experiment does not demonstrate a practically reliable way to detect that distinction at these sample sizes. DII's omnibus target and its realized diagnostic value are separate claims. In particular, no rejection in a small financial sample cannot establish the absence of directional structure.

'''.replace('% ROWS','\n'.join(rows))
s=s.replace(r'\section{A reproducible financial application}',validation+r'\section{A reproducible financial application}')
f=json.loads((P/'finance/decision_use_results.json').read_text());assert len(f['comparisons'])==12;assert all(x['p_holm']>.05 for x in f['comparisons']);assert sum(x['trained_shock_square_coefficient']>1e-12 for x in f['comparisons'])==3
financial=r'''
\subsection{A direct decision about forecasting error magnitude}\label{sec:riskdecision}
A separate secondary audit asks whether squared shock exposure improves forecasts of the affine mean model's squared errors. This uses the same 12 cells, data, outcome windows, and training/evaluation origins. Nonnegative least squares forecasts squared training residuals from an intercept and squared lagged shock and asset change; the augmented model adds the squared current shock. The nonnegative specification permits an upward shock-square adjustment, not a negative coefficient. Both forecasts are floored at one percent of training mean squared error. They are evaluated with $L(v,e)=\log v+e^2/v$, whose conditional expected value is minimized at $v=\E[e^2\mid\mathcal I]$. This is a forecast-risk decision about a fixed fitted mean model, not necessarily structural conditional variance.

The squared-shock coefficient is zero in nine of the 12 training fits. The remaining cells are yen/USD at one month and credit spreads at one and six months. Their held-out baseline-minus-augmented score differences are approximately 0.0013, $-0.1454$, and $-0.0151$, with paired block 95\% intervals [$-0.0138$,0.0152], [$-0.3920$,$-0.0014$], and [$-0.0337$,0.0011]. No cell has a supported positive gain; all Holm-adjusted p-values for improvement equal one. Complete outputs and the prespecified secondary protocol are archived. The intervals condition on the trained forecasts under working stability and evaluation-dependence assumptions, not the DII theorem.

The observed data therefore support neither a new directional finding nor a demonstrated benefit from adding this shock-square risk predictor. That does not make the two questions equivalent. The analytical benchmarks separate them; the low-power diagnostics explain why the financial non-rejections are inconclusive about underlying directionality.

'''
s=s.replace(r'\section{Financial transmission and the use of DII}',financial+r'\section{Financial transmission and the use of DII}')
# Put the decision question early, and remove any suggestion that an omnibus object is automatically useful.
needle='An economic shock can move an outcome, change its risk, or leave an asymmetric relation between two dynamic representations.'
s=s.replace(needle,'A financial model may forecast the response to a shock accurately while leaving unresolved which directional representation has disturbances independent of its conditioning variables. That distinction matters when a researcher uses fitted residuals to simulate counterfactual shock scenarios. An economic shock can move an outcome, change its risk, or leave an asymmetric relation between two dynamic representations.')
s=s.replace('The paper has three connected contributions:', 'The paper has three connected contributions:')
s=s.replace('These are distinct results with distinct assumptions.','A further matched benchmark shows that a simpler residual moment can be much more powerful than DII, while a direct financial forecast-risk audit has no supported positive gain. The population distinctions should not be read as demonstrated empirical superiority.')
abstract="""Mean and risk forecasts do not determine which of two dynamic representations has disturbances independent of its conditioning variables. The Directional Irreversibility Index (DII) measures an asymmetry between these residual-dependence restrictions. Financial benchmarks distinguish it from mean exposure, risk dependence, and delayed information absorption. An additional benchmark holds the forward conditional-mean and variance functions fixed while directional structure changes, including a case missed by a particular reverse residual moment. We develop joint block inference for fitted regressions and preprocessing across dependent horizons, with distinct null calibrations and a mean-approximation sensitivity bound. Simulations evaluate its scope and practical limits: in a new matched comparison, a focused moment diagnostic is substantially more powerful in one design, while DII has low power for the broader distinction. A fixed four-market application has no primary directional rejection or supported gain from a prespecified shock-square forecast-risk predictor. The contribution is an explicit framework separating a directional estimand, its sampling and specification requirements, and the additional evidence needed for a financial modeling decision."""
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',s,flags=re.S)
s=s.replace(r'\end{thebibliography}',r'\bibitem{Shimizu} Shimizu, S., P. O. Hoyer, A. Hyv\"arinen, and A. Kerminen (2006). A Linear Non-Gaussian Acyclic Model for Causal Discovery. \emph{Journal of Machine Learning Research}, 7, 2003--2030. \url{https://www.jmlr.org/papers/v7/shimizu06a.html}.'+'\n'+r'\end{thebibliography}')
s=s.replace('The financial application supplies no primary rejection or demonstrated positive predictive gain.','The matched benchmark favors a simpler diagnostic in a prespecified moment setting and reveals low DII power in the harder setting. The financial application supplies no primary rejection or supported gain in mean or forecast-risk prediction.')
(S/'main.tex').write_text(s);(P/'abstract.txt').write_text(abstract+'\n');print('Decision-use comparison integrated; abstract words',len(abstract.split()))
