"""Additional analytical interpretation of the unchanged simulation study."""
from pathlib import Path
import csv,json,math,statistics
S=Path(__file__).resolve().parent;P=S.parent
rows=list(csv.DictReader((P/'full_study_raw.csv').open()))
rows=[r for r in rows if r['version']=='feasible' and r['allocation']=='quadratic']
out=[]
for n in (125,216):
 for alt in ('weak','strong'):
  groups=[]
  for design in ('regular',alt):
   rr=[r for r in rows if int(r['n'])==n and r['design']==design]
   assert len(rr)==300
   groups.append([int(float(r['p_wild'])<=.05)-int(float(r['p_intersection'])<=.05) for r in rr])
  null,pos=groups
  delta=.5*(statistics.mean(pos)-statistics.mean(null))
  se=math.sqrt(.25*(statistics.variance(null)/len(null)+statistics.variance(pos)/len(pos)))
  out.append(dict(n=n,alternative=alt,intersection_minus_wild_loss=delta,mc_se=se,mc_interval_low=delta-1.96*se,mc_interval_high=delta+1.96*se))
(P/'second_pass_results.json').write_text(json.dumps({'interpretation':'Pointwise approximate 95% Monte Carlo intervals for constructed equal-weight equal-cost loss differences; paired rule decisions within each replication; independent simulation cells. No real-world decision uncertainty is covered.','rows':out},indent=2))
s=(S/'main.tex').read_text()
insert=r'''\subsection{A fixed model inventory and an error budget}
A team usually reviews several candidates. Let $K$ be a fixed, prespecified number of model--horizon comparisons, and let $A_k$ indicate advancing candidate $k$ using its intersection screen. The candidate list, targets, kernels, and error allocations are fixed before evaluation. The relevant count is $V=\sum_{k\in\mathcal H_0} A_k$, where $\mathcal H_0$ contains candidates with nonpositive population contrasts. Then
\begin{equation}\label{eq:inventory}
\mathbb E[V]=\sum_{k\in\mathcal H_0}\Pr(A_k=1),\qquad
\Pr(V\geq1)\leq\sum_{k\in\mathcal H_0}\Pr(A_k=1).
\end{equation}
Neither relation requires independent candidates. Suppose each candidate satisfies the marginal pointwise size conclusion established below at its assigned level $\alpha_k$. For fixed $K$ and a fixed collection of covered data-generating processes, choosing $\alpha_k=\alpha w_k$ with prespecified $w_k>0$ and $\sum_k w_k=1$ gives
\[
\limsup_{n\to\infty}\Pr(V\geq1)\leq\alpha,
\qquad \limsup_{n\to\infty}\mathbb E[V]\leq\alpha.
\]
The proof is the finite union bound and the sum of the marginal size bounds. This is a standard error-budget consequence of the theorem, not a new multiple-testing method. The result inherits the theorem's nuisance rates and bootstrap approximation conditions. It does not cover a growing inventory, adaptive search over models, repeated monitoring, or local sequences excluded by the marginal theorem.

This changes the meaning of a nominal five-percent screen. Testing every model at five percent does not give a five-percent inventory-wide error bound. For an illustrative inventory of 100 null candidates each having the larger-sample regular-null rejection rates in Section 5, expected false advances would be 44.3 under the wild rule and 5.0 under the intersection. These are linear extrapolations of simulation frequencies, not observed institutional counts; they require the stipulated marginal rates. Dependence among candidates affects the distribution of the count but not its expectation. An inventory-wide budget instead requires recalibration at the allocated levels. The existing five-percent experiments cannot establish its power, and 399 bootstrap draws would be inadequate for very small allocations. A binding limit on total review capacity would require a separate selection policy.

'''
s=s.replace('\\section{Generated residuals under cross-lag dependence}',insert+'\\section{Generated residuals under cross-lag dependence}')
table_rows=[]
for r in out:
 table_rows.append(f"{r['n']} & {r['alternative'].capitalize()} & {r['intersection_minus_wild_loss']:.3f} & {r['mc_se']:.3f} & [{r['mc_interval_low']:.3f}, {r['mc_interval_high']:.3f}] "+r'\\')
mc=r'''\subsection{Simulation uncertainty in the decision comparison}
The two rules are evaluated on the same datasets, so their errors should be compared as paired observations. Write $Z_{0r}=A_{wr}-A_{\cap r}$ for null replication $r$ and $Z_{1r}=A_{wr}-A_{\cap r}$ for an alternative replication. At equal scenario weights and unit costs, the estimated intersection-minus-wild loss is $\widehat\Delta=(\bar Z_1-\bar Z_0)/2$. Independent simulation cells give estimated Monte Carlo standard error
\[
\widehat{\mathrm{se}}(\widehat\Delta)=\frac12\sqrt{s_0^2/R_0+s_1^2/R_1},
\]
where $s_j^2$ is the sample variance of paired differences and $R_j=300$. This accounts for within-dataset correlation between the rules, including their shared resampling draws.

\begin{table}[htbp]\centering
\caption{Monte Carlo uncertainty in constructed loss differences}
\begin{tabular}{rlrrl}\toprule
$n$ & Alternative & Loss difference & MC SE & Approx. 95\% interval\\\midrule
% ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent Negative differences favor the intersection. Intervals are pointwise normal approximations for simulation error at the stated cost and scenario weights. They are not simultaneous intervals, institutional treatment-effect intervals, or uncertainty bounds for unknown cost ratios. Both alternatives at a given sample size share the same simulated null cell.
\end{table}

The sign reversal for the weaker alternative persists after accounting for Monte Carlo variation in this grid. This strengthens the evidence that the decision ranking depends on the evaluation size in these designs. It does not establish a population-wide ranking across financial settings. Institutional scenario frequencies, error costs, and the validity of the maintained model remain separate uncertainties.

'''.replace('% ROWS','\n'.join(table_rows))
s=s.replace('\\section{Implications for financial model validation}',mc+'\\section{Implications for financial model validation}')
s=s.replace('Third, it connects the screening errors to a transparent loss comparison that identifies when reducing false advances is worth the accompanying loss of power.','Third, it connects screening errors to a transparent loss comparison, reports paired Monte Carlo uncertainty, and shows how marginal guarantees translate to an error budget for a fixed model inventory. These decision consequences identify both when conservatism is worthwhile and where further policy design is required.')
s=s.replace('The available macroeconomic shock tables from earlier versions have therefore not been used as a verified institutional benchmark in this manuscript. Their underlying executable project was unavailable, and their calendar and bandwidth choices do not automatically match the new theorem. A strong field application remains a material opportunity for further validation of the proposed decision use.','No institutional dataset is analyzed here. A field evaluation would need to fix the candidate inventory and screening targets before evaluation, justify the dependence and training assumptions, and measure subsequent validation outcomes independently of the screening statistic. The present results establish neither institutional savings nor a deployable validation policy.')
(S/'main.tex').write_text(s)
print(json.dumps(out,indent=2))
