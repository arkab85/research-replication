from pathlib import Path
import re,json
S=Path(__file__).resolve().parent;P=S.parent
meta=json.loads((P/'full_study_results.json').read_text())
assert meta['replications_per_cell']==300 and meta['bootstrap_draws']==399 and meta['seed']==20260911
s=(S/'base_manuscript.tex').read_text()
s=s.replace('\\documentclass[12pt]','\\documentclass[11pt]')
s=s.replace('Dynamic Model Discrimination\\\\with Estimated Innovations','Screening Dynamic Financial Models\\\\under Uncertain Test Calibration')
s=re.sub(r'\\author\{[^}]*\}',r'\\author{}',s)
s=re.sub(r'\\date\{.*?\}\n\\begin\{document\}',lambda m:'\\date{}\n\\begin{document}',s,flags=re.S)
s=s.replace('\\clearpage\n\\section{Introduction}','\\section{Introduction}')
a='''Financial model-validation teams must decide when evidence of dynamic asymmetry warrants further investigation. A screening rule can advance unsupported models if it calibrates equality of residual-dependence measures as though both measures vanish. This paper develops inference for signed contrasts of estimated residual-independence measures under stationary finite-memory dynamics. Covariance-operator bounds handle generated residuals, and joint block approximations accommodate different leading orders within the null. An intersection rule provides pointwise control over the covered configurations. In a reproducible 4,800-dataset study, the wild component rejects 41--44 percent of equal-positive nulls at a nominal five-percent level; the intersection rejects five percent but is conservative at double independence. An illustrative loss analysis translates false advances and missed positive contrasts into explicit cost thresholds, without assuming that the more conservative rule is always preferable. The method supports an evidence-screening decision, not automatic deployment or causal certification. Its value depends on the modeling restrictions, training allocation, and costs of the errors it trades off.'''
x=s.index('\\begin{abstract}');y=s.index('\\end{abstract}')
s=s[:x]+'\\begin{abstract}\n'+a+'\n'+s[y:]
s=s.replace('Keywords: dynamic models; residual independence; generated residuals; kernel tests; time series.','Keywords: financial model validation; statistical decision making; residual independence; time series; bootstrap inference.')
x=s.index('\\section{Introduction}');y=s.index('\\subsection{Relationship to existing work}',x)
intro=r'''\section{Introduction}
A financial model-validation team must decide which apparent dynamic relationships deserve further investigation. A candidate shock may predict a risk outcome, a model may leave unexplained distributional dependence, or two representations may fit the same conditional means while imposing different innovation restrictions. Advancing a model on misleading evidence consumes validation capacity and can lend unwarranted credibility to a proposed mechanism. Rejecting a useful signal too readily also has a cost. The decision is therefore about the reliability of evidence and the consequences of screening errors, before any model is authorized for deployment.

This paper studies one input to that decision: a signed difference between residual-independence measures for opposing, history-conditioned dynamic representations. The statistic compares departures from two distributional restrictions. Under a maintained nonreversible additive-noise model it can distinguish an admissible ordering. Outside that class, a positive difference indicates unequal specification departures. The distinction is essential for model governance: a screening flag should trigger additional validation, not be treated as evidence that a causal model has been certified.

The calibration problem is easy to overlook. Equality of the two dependence measures may mean that both are zero, or that both are positive. The first configuration produces a degenerate quadratic fluctuation; regular instances of the second produce a linear fluctuation. A resampling rule suitable for one can produce seriously misleading evidence in the other. Generated residuals and temporal dependence make the problem operationally relevant because the researcher must estimate the underlying regressions before comparing them.

The paper makes three contributions. First, it establishes generated-residual perturbation bounds and joint block approximation for a fixed family of signed dependence contrasts under finite-memory dynamics. The joint argument allows cross-lag innovation dependence and shared shocks. Second, it evaluates the complete procedure across exact nulls, nonlinear alternatives, and excluded or insufficient-rate stress cases. Third, it connects the screening errors to a transparent loss comparison that identifies when reducing false advances is worth the accompanying loss of power.

The full-procedure experiment contains 4,800 datasets, with oracle and feasible residuals evaluated on each. At a nominal five-percent level, the wild component rejects approximately 41--44 percent of equal-positive nulls. The intersection of wild and paired decisions rejects five percent. At double independence, however, the intersection makes no rejections in the reported cells. This protects against one calibration failure at the cost of conservatism elsewhere. It is a trade-off, not a uniformly superior screening rule.

The loss analysis makes that trade-off explicit. For example, combining the equal-positive null and weaker alternative with equal scenario weights and equal unit error costs gives estimated loss 0.240 for the wild rule and 0.170 for the intersection at the larger evaluation size. At the smaller size the ranking reverses: 0.308 versus 0.383. These are constructed decision scenarios based on the simulations, not measured institutional costs or financial returns. The paired-only and intersection rules make identical screening decisions in all 4,800 feasible evaluations, so the study does not establish an empirical advantage of the intersection over paired-only screening.

The intended contribution is a disciplined connection between a distributional model comparison, its feasible calibration, and the decision consequences of its errors. Existing kernel and block-bootstrap tools supply the ingredients. The incremental result lies in their joint treatment for estimated dynamic contrasts and the precise delineation of when the resulting evidence is reliable. Fixed weighted horizon comparisons and balanced panels fall within the finite-dimensional extension; general persistent processes and irregular changing calendars do not follow automatically.

Section 2 defines the target and the screening decision. Section 3 treats generated residuals. Section 4 establishes joint calibration. Section 5 reports size, power, and decision-loss comparisons. Section 6 explains implications and limitations for financial model validation. Section 7 concludes. The joint proof is included in Appendix A.

'''
s=s[:x]+intro+s[y:]
x=s.index('\\section{Generated residuals under cross-lag dependence}')
decision=r'''\subsection{The model-validation decision}
Let an action $a=1$ mean advancing a model for further validation on the basis of positive residual asymmetry; $a=0$ means that this screening criterion supplies no such advance. This is a limited decision. It does not authorize trading, establish a causal transmission channel, or rule out advancement on separate economic evidence. Its statistical target is $D_h>0$, with the structural interpretation kept conditional on the assumptions above.

For a screening rule $s$, let $f_s$ be its false-advance probability in a specified null scenario and $t_s$ its advance probability in a specified positive-contrast scenario. If the scenario mixture places weight $\pi$ on the latter, and error costs are $c_F$ for false advancement and $c_M$ for missing a positive contrast, the normalized expected loss is
\begin{equation}\label{eq:decisionloss}
L_s=(1-\pi)c_F f_s+\pi c_M(1-t_s).
\end{equation}
The costs describe errors in this screening decision, not realized portfolio losses. Values of $\pi,c_F,c_M$ must be supplied by the decision context. They are not estimated from a test p-value.

Compare the wild rule $w$ with the intersection rule $\cap$. When $f_w>f_\cap$ and $t_w\ge t_\cap$, the intersection has lower expected loss exactly when
\begin{equation}\label{eq:threshold}
\frac{c_F}{c_M}>\frac{\pi}{1-\pi}
\frac{t_w-t_\cap}{f_w-f_\cap}.
\end{equation}
This follows by subtracting \eqref{eq:decisionloss} for the two rules. It is a decision-accounting identity, not a new identification theorem or a claim that either rule is Bayes optimal. Estimated frequencies produce an illustrative threshold with simulation uncertainty. Section 5 reports those thresholds and both rankings rather than selecting a single favorable cost assumption.

A practical use is to require that a proposed screening procedure document its null configuration, evaluation error budget, and action consequences. A large directional estimate is not enough: both component dependence measures, the common horizon target, the joint resampling construction, and the permissible interpretation must be visible to the validation team. Finite-memory restrictions should be justified for the model under review; they cannot be assumed merely because a rolling data window is finite.

'''
s=s[:x]+decision+s[x:]
x=s.index('\\section{What the available shock evidence establishes}')
y=s.index('\\section{Conclusion}',x)
imp=r'''\section{Implications for financial model validation}
The application context is an evidence-screening step in model validation. The experiments demonstrate a calibration failure that can be reproduced under a known population null and quantify the associated trade-off. They do not establish measured improvements in a bank's model inventory, portfolio performance, regulatory capital, or analyst productivity. Those outcomes would require an independently observed decision setting and a suitable evaluation design.

Three implications follow from the completed analysis. First, the validation record should report both dependence components, since equality can arise in configurations requiring different calibration. Second, the data used to fit a representation and the data used to evaluate its residual restrictions cannot be budgeted independently of the required nuisance rate. Third, choosing a conservative screen involves a cost trade-off; neither nominal size nor power alone determines its decision value.

The paired-only benchmark is particularly important. It matches the intersection's observed decisions in the current simulation grid. The intersection's theoretical appeal is that a valid component controls the rejection event on its own covered null domain; the current experiments do not demonstrate an incremental practical benefit over paired-only screening. A broader claim would require additional designs that establish such a benefit without hiding the associated power costs.

For a financial application, the proposed method must be connected to the observed data process. Finite distributed-lag or finite moving-average representations can satisfy the theorem's dependence restriction. A persistent autoregression or a series whose dependence merely decays does not satisfy exact finite dependence automatically. Likewise, evaluation-selected bandwidths, unequal calendar coverage, and misspecified first stages need separate treatment. A real-data example can illustrate calculations but cannot retrospectively supply those assumptions.

The available macroeconomic shock tables from earlier versions have therefore not been used as a verified institutional benchmark in this manuscript. Their underlying executable project was unavailable, and their calendar and bandwidth choices do not automatically match the new theorem. A strong field application remains a material opportunity for further validation of the proposed decision use.

'''
s=s[:x]+imp+s[y:]
x=s.index('\\section{Conclusion}');y=s.index('\\appendix',x)
s=s[:x]+r'''\section{Conclusion}
A model-validation screen should connect the evidence it reports to the decision it supports. Residual-independence contrasts compare distributional restrictions; they certify a structural ordering only under additional maintained assumptions. Their equality null can change its leading order, making calibration a consequential part of the screening design.

For finite-memory dynamics, the paper establishes joint block approximations and generated-residual transfer for fixed signed collections of dependence measures. The experiments document severe overrejection from an inappropriate single calibration, alongside conservatism of the intersection rule. The loss comparison identifies which combinations of scenario weights and error costs favor that protection and which do not.

The contribution is a framework for reliable evidence screening with estimated dynamic representations, with its statistical and decision limits stated together. Its practical relevance should be assessed through the documented error trade-off and the extent to which the maintained model class describes the financial setting where the screen will be used.

'''+s[y:]
# Add decision results immediately before managerial implications.
d=json.loads((P/'decision_results.json').read_text())
rows=[]
for r in d['rows']:
 rows.append(f"{r['n']} & {r['alternative'].capitalize()} & {r['critical_cost_ratio_at_equal_prior']:.3f} & {r['wild_equal_prior_equal_cost_loss']:.3f} & {r['intersection_equal_prior_equal_cost_loss']:.3f} "+r'\\')
sec=r'''\subsection{Illustrative decision-loss comparison}
This analysis was added after the simulation study to interpret its error rates. It is not a preregistered managerial experiment or an empirically calibrated welfare calculation. The null scenario is the equal-positive baseline; the positive scenarios are the weaker and stronger nonlinear alternatives. The table reports the threshold factor in \eqref{eq:threshold} and normalized losses at $\pi=1/2$, $c_F=c_M=1$.

\begin{table}[htbp]\centering
\caption{When is reducing false advancement worth the lost power?}
\begin{tabular}{rlrrr}\toprule
$n$ & Alternative & Threshold factor & Wild loss & Intersection loss\\\midrule
% LOSS ROWS
\bottomrule\end{tabular}
\par\vspace{5pt}\raggedright\noindent For other scenario weights, multiply the threshold factor by $\pi/(1-\pi)$ to obtain the break-even ratio $c_F/c_M$. Losses are normalized constructed quantities, not monetary estimates. The paired-only rule has the same loss as the intersection in this grid because their feasible decisions coincide.
\end{table}

For the weaker alternative at $n=125$, equal costs favor the wild screen in this constructed mixture despite its excessive null rejection, because the intersection loses substantial power. At $n=216$, equal costs favor the intersection. This reversal is a reason to report costs and power jointly. The analysis does not support choosing a method by its most favorable scenario or treating the point estimates as known population frequencies.

'''.replace('% LOSS ROWS','\n'.join(rows))
s=s.replace('\\section{Implications for financial model validation}',sec+'\\section{Implications for financial model validation}')
(S/'main.tex').write_text(s)
(P/'abstract.txt').write_text(a)
print('Abstract words:',len(a.split()))
