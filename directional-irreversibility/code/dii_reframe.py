"""Restore DII as the contribution; retain verified inference and simulations."""
from pathlib import Path
S=Path(__file__).resolve().parent;P=S.parent
s=(S/'main.tex').read_text()
def span(a,b):return s[s.index(a):s.index(b)]
prefix=s[:s.index('\\section{Introduction}')]
prefix=prefix.replace('Screening Dynamic Financial Models\\\\under Uncertain Test Calibration','Directional Irreversibility in Economic Dynamics\\\\Inference and Shock Transmission')
a='''How can researchers distinguish a dynamic relationship's directional structure from its predictive strength and its effects on risk? This paper develops the Directional Irreversibility Index (DII), a horizon-specific contrast between residual-dependence measures for opposing dynamic representations. Financial exposure models show why a nonzero linear response, distributional dependence, and positive DII are distinct: nonlinear exposure can yield directional asymmetry with zero linear response, while a pure variance channel can remain directionally symmetric. An immediate-incorporation benchmark separates persistent DII from delayed absorption of news. DII has an ordering interpretation under explicit additive-noise and nonreversibility restrictions; otherwise it measures unequal departures from residual independence. The main inferential contribution treats generated residuals and joint horizon contrasts when the null contains both quadratic and linear limits. Under stationary finite-memory dynamics, joint block approximations yield pointwise-valid inference over the covered null configurations. A reproducible 4,800-dataset study documents the consequences of incorrect calibration and the conservatism of an intersection procedure. The framework distinguishes horizon-specific direction, average asymmetry, and persistence, providing a common basis for research on shock transmission in finance and other dynamic business settings.'''
i=prefix.index('\\begin{abstract}');j=prefix.index('\\end{abstract}')
prefix=prefix[:i]+'\\begin{abstract}\n'+a+'\n'+prefix[j:]
prefix=prefix.replace('financial model validation; statistical decision making; residual independence; time series; bootstrap inference.','directional irreversibility; financial shocks; residual independence; dynamic models; bootstrap inference.')
intro=r'''\section{Introduction}
An economic shock can move an outcome, change its risk, or leave an asymmetric relation between two dynamic representations. These are different empirical objects. A risk manager assessing a nonlinear exposure, a researcher studying monetary news in asset prices, and an operations analyst tracing a disruption through a production system need to know which object their evidence identifies. A mean-response coefficient describes one aspect of transmission. Distributional dependence describes a broader statistical relationship. Neither, by itself, describes whether the forward and reverse representations satisfy the same residual-independence restrictions.

This paper develops the Directional Irreversibility Index, $\D(h)$, to measure that last object at a prespecified horizon. DII compares the dependence of the forward residual on its full regressor vector with the corresponding dependence in the reverse representation. When one ordering admits independent additive noise and the other does not, the contrast has an ordering interpretation. More generally, it measures the asymmetry of two specification departures. The reverse regression is a comparison of statistical representations, not a claim that a future asset price causes yesterday's shock.

The economic distinction can be seen in three simple financial exposures. A linear Gaussian exposure has a nonzero linear shock response but zero DII. A quadratic exposure has positive DII even though its population linear shock coefficient is zero. A pure scale exposure has shock-dependent risk and zero conditional mean, yet DII is zero because the residual-dependence measures are equal. These results show both the information DII adds and what it deliberately leaves unresolved. Positive DII is not an omnibus measure of dependence; zero DII does not imply that a shock is economically irrelevant. Section 2 establishes these comparisons analytically, including a finite-lag dynamic construction. It also gives an immediate-incorporation benchmark in which impact-inclusive changes retain positive DII at every fixed horizon while strictly post-impact changes are independent of the shock. Thus persistence of DII and slow absorption of information are distinct claims.

The main methodological contribution is inference for this contrast after fitting the two models. No asymmetry can mean that both residual-dependence measures vanish, or that they are equally positive. The statistic can consequently have a quadratic limit at rate $n$ or a linear limit at rate $\sqrt n$. Estimated residuals and common time-series shocks enter the same inferential problem. The paper establishes operator perturbation bounds and a joint block approximation for a fixed family of contrasts under finite-memory dynamics. The scope includes prespecified horizon averages, horizon-shape comparisons, and balanced panels whose joint rows satisfy the dependence condition.

A complete-procedure simulation study shows why this distinction matters. At a nominal five-percent level, the wild component rejects approximately 41--44 percent of equal-positive nulls in the reported baseline cells; the intersection of the wild and paired decisions rejects five percent. The intersection is conservative at double independence and can sacrifice substantial power. Its decisions coincide with paired-only testing throughout the 4,800 feasible evaluations, so the contribution cannot rest on an empirical claim that the intersection improves on that benchmark. The contribution instead combines a precisely defined dynamic object, the generated-residual and joint-calibration argument, and a reproducible account of the cases in which the evidence can be interpreted.

DII is designed to complement the established tools used to study economic dynamics. A financial application can compare its horizon profile with mean and risk responses to externally identified news. The comparison can distinguish a nonlinear exposure from a simple linear response, but it does not itself establish delayed information absorption, market inefficiency, or an intervention effect. In marketing or operations, the same contrast can be used for a prespecified intervention--outcome representation when its modeling restrictions are defensible. The common research problem is how to interpret and compare dynamic distributional restrictions, rather than a particular institutional screening workflow.

The paper has three connected contributions: a DII interpretation that separates directional structure from mean and risk effects; feasible joint inference for fitted dynamic contrasts with different null limits; and a reusable framework for distinguishing one-horizon evidence, average asymmetry, and persistence. The analytical examples make the first contribution concrete, the proofs support the second, and the software and finance discussion operationalize the third. The additive-noise identification principle, kernel independence measure, and intersection-union logic have established antecedents; the paper's claim is their scoped development for this dynamic inference problem.

Section 2 defines DII and develops the economic examples. Section 3 explains implementation and the inferential results. Section 4 reports the completed simulation evidence. Section 5 develops the financial research use and interpretation of horizon profiles. Section 6 concludes. Appendices give the full residual and bootstrap arguments, followed by an optional decision-loss application.

'''
related=span('\\subsection{Relationship to existing work}','\\section{The economic comparison}')
related=related.replace('Peters, Mooij, Janzing, and Sch\"olkopf (2014)','Peters, Mooij, Janzing, and Sch\"olkopf (2014)')
related+=r'''Peters, Janzing, and Sch\"olkopf (2013) study time-series structural models with independent noise. Their process-level identification restrictions should not be confused with independence of a single innovation from a contemporaneous regressor. DII's sampling argument below permits dependence among different horizon rows and treats their joint law explicitly. The inference theorem here is narrower than a general causal-discovery result for unrestricted time series.

'''
econ=span('\\section{The economic comparison}','\\subsection{The model-validation decision}')
econ=econ.replace('\\section{The economic comparison}','\\section{DII: the object and its economic meaning}\\label{sec:object}')
econ=econ.replace('D_h=H_b-H_f',r'\D(h)\equiv D_h=H_b-H_f')
econ=econ.replace('The superscripts name opposing regressions, not physical backward causation.','The Directional Irreversibility Index (DII) refers to the asymmetry of these representations; it is not a test that the full stochastic process differs from its time-reversed law. The superscripts name opposing regressions, not physical backward causation.')
examples=r'''\subsection{Mean responses, risk effects, and directional structure}
Consider a standardized shock $S\sim N(0,1)$ and independent $\varepsilon\sim N(0,1)$, with an empty history for this illustration. Write $b=\operatorname{Cov}(S,Y)$ for the population linear slope and $J=\HS(S,Y)$ for unconditional dependence. Use fixed characteristic kernels for each variable, with the same variable-specific kernel in the forward and reverse comparison. The three quantities $b$, $J$, and $\D$ answer different questions.

\begin{proposition}[Three distinct features of a financial exposure]\label{prop:separation}
For nonzero $a,\gamma$ and positive $\lambda$:
\begin{enumerate}
\item If $Y=aS+\varepsilon$, then $b=a\ne0$, $J>0$, and $\D=0$.
\item If $Y=\gamma(S^2-1)+\varepsilon$, then $b=0$, $J>0$, and $\D>0$.
\item If $Y=\sqrt{1+\lambda S^2}\,\varepsilon$, then both conditional means $\E[Y\mid S]$ and $\E[S\mid Y]$ vanish, $\operatorname{Var}(Y\mid S)=1+\lambda S^2$, $J>0$, and $\D=0$ with equal-positive dependence components.
\end{enumerate}
\end{proposition}
\begin{proof}
In the first case, joint Gaussianity makes each conditional-mean residual independent of its regressor; $b=a$ implies dependence of $S$ and $Y$. In the second case, symmetry of the conditional law of $Y$ in $S$ gives $\E[S\mid Y]=0$, while the forward residual is $\varepsilon$. Also $\E[SY]=0$ and $\operatorname{Cov}(S^2,Y)=2\gamma\ne0$. Thus the reverse residual $S$ is dependent on $Y$ and the forward residual is independent of $S$. Characteristicness gives a strictly positive contrast. In the third case, symmetry again gives $\E[S\mid Y]=0$, and independence and zero mean of $\varepsilon$ give $\E[Y\mid S]=0$. The residuals are therefore $Y$ and $S$. With matched kernels, symmetry of HSIC gives $H_f=\HS(Y,S)=\HS(S,Y)=H_b$. Both are positive because $\operatorname{Cov}(S^2,Y^2)=2\lambda>0$.
\end{proof}

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{../dii_population_examples.pdf}
\caption{Three analytical financial exposures. The curve is the conditional mean and the band is one conditional standard deviation; neither is an estimated confidence interval. Parameters are $a=\gamma=\lambda=0.6$. DII statements are population results under matched characteristic kernels.}
\end{figure}

The quadratic case is a stylized convex exposure, relevant to the distinction between a linear factor loading and nonlinear payoff sensitivity. Its zero linear coefficient does \emph{not} mean that the conditional mean response vanishes: $\E[Y\mid S=s]=\gamma(s^2-1)$. A sufficiently flexible response analysis can detect that mean. The scale case represents shock-dependent risk without a conditional-mean effect. It shows why DII should be reported beside both dependence components and appropriate risk measures, rather than advertised as detecting every feature missed by mean regressions. These are explanatory analytical models, not claims about an estimated asset-pricing equilibrium or new universal identification results.

The distinction also has a dynamic construction. For fixed $H$, let shocks $S_t$ and disturbances $\varepsilon_t$ be independent iid standard Gaussian sequences and set
\begin{equation}\label{eq:exposure}
R_t=\sum_{j=1}^{H}\{a_jS_{t-j}+\gamma_j(S_{t-j}^2-1)\}+\varepsilon_t.
\end{equation}
For the comparison $(S_t,R_{t+h})$, the population linear response is $a_h$ and the conditional mean is $a_hS_t+\gamma_h(S_t^2-1)$. The remaining terms form mean-zero noise independent of $S_t$. If $a_h=0$ and $\gamma_h\ne0$, the symmetry and covariance argument above gives $\D(h)>0$ despite zero linear response. If every $\gamma_j=0$, each shock--return pair is Gaussian and its DII vanishes even when $a_h\ne0$. The joint vector of any fixed set of these comparisons is a finite-memory process, so it fits the dependence framework below when the remaining inferential conditions hold. Hence a horizon with zero linear response and positive DII can arise from nonlinear exposure alone; it need not represent slow incorporation of news.

'''
# A financial interpretation result that does not require an equilibrium claim.
examples+=r"""\subsection{Persistent DII need not mean delayed incorporation}
A horizon profile depends on how the outcome interval is defined. The following benchmark separates an impact-inclusive price change from a post-impact increment.

\begin{proposition}[An immediate-incorporation benchmark]\label{prop:incorporation}
Let $S_t$ and $e_t$ be independent iid standard Gaussian sequences, let $\gamma\ne0$, and let price increments be $\Delta P_t=\gamma(S_t^2-1)+e_t$. With respect to the filtration generated by shocks through $t$, $P_t$ is a martingale. For every fixed $h\ge1$, the impact-inclusive change $Y_{t,h}=P_{t+h}-P_{t-1}$ has strictly positive DII relative to $S_t$ with empty history and characteristic kernels. The strictly post-impact change $Q_{t,h}=P_{t+h}-P_t$ is independent of $S_t$ and has DII zero.
\end{proposition}
\begin{proof}
Each increment is mean zero and independent of all earlier information, so $\E[P_{t+h}\mid\mathcal F_t]=P_t$. Also $Y_{t,h}=\gamma(S_t^2-1)+U_{t,h}$ with $U_{t,h}=e_t+\sum_{j=1}^h\Delta P_{t+j}$ independent of $S_t$. Symmetry gives $\E[S_t\mid Y_{t,h}]=0$, and $\operatorname{Cov}(S_t^2,Y_{t,h})=2\gamma\ne0$. The forward residual is independent and the reverse residual is dependent, so DII is positive. The post-impact change contains only future independent increments, proving the final assertion.
\end{proof}

This is a statistical benchmark of immediate incorporation, not a complete equilibrium asset-pricing model. It shows that persistent asymmetry in an impact-inclusive outcome can reflect the retained impact itself. There is no predictable subsequent price increment in the benchmark. An empirical claim about gradual information absorption therefore needs an outcome interval and an economic model that distinguish retained impact, nonlinear exposure, and genuinely later responses. The proposition does not say that all persistent DII is a measurement artifact; it identifies a specific alternative explanation that can be ruled out by design.

"""

summary=r'''\section{Estimating DII and comparing horizon profiles}\label{sec:implementation}
\subsection{From fitted models to a directional contrast}
Fit the two conditional means on a training period, freeze their specification and transformations, and compute evaluation residuals on a separate period. Use a gap that separates training and evaluation under the maintained finite-memory model. The evaluation data contain both residuals, both full regressor vectors, and a common calendar for every prespecified horizon or unit. Report the two component dependence estimates as well as their difference. A positive difference between two positive departures has a different interpretation from evidence consistent with a valid additive representation in one ordering.

For a fixed finite collection, write the target as $D=\sum_d a_d\|C_d\|^2$, where $C_d$ is a residual--regressor covariance operator in feature space. A single DII has coefficients $+1$ and $-1$. This formulation also accommodates a fixed weighted horizon average, an average across aligned assets, or a prespecified contrast such as
\[
D_{3}-\tfrac12(D_{1}+D_{6}).
\]
The last quantity tests a particular horizon shape. It is neither a test of positivity at all horizons nor evidence that a causal effect peaks at three months. Kernel scales and transformations must be fixed across the quantities being compared; a numerical DII magnitude is not invariant to arbitrary rescaling or comparable to a return in percent.

\subsection{Why one equality null has different limits}
There are two covered boundary configurations. When all covariance operators vanish, the leading fluctuation of $D$ is a quadratic form in their joint Gaussian limit. At a regular equality with nonzero operators, it is a linear Gaussian fluctuation with positive variance. Appendix~\ref{app:calibration} establishes the joint block approximation and its transfer to fitted residuals. The key sufficient evaluation error condition is $\sqrt n\,r_{n,d}\to_p0$ for each residual component. Training separation alone does not establish that rate.

A Gaussian block multiplier captures the specified quadratic limit; a centered paired block construction captures the regular linear limit. All components use the same weights or resampled calendar. Independent resampling of the two directions or averaging separately computed p-values would discard the relevant joint covariance. For the same upper-tail target, use $p_\cap=\max(p_w,p_b)$. On each fixed law covered by the theorem, its rejection event is contained in that of a valid component, yielding pointwise control of the nonpositive null. The theorem does not establish uniform validity near degeneracy, cover exact cancellation automatically, or justify arbitrary persistent financial processes. The pure-scale example in Proposition~\ref{prop:separation} is a useful exact-cancellation boundary, not a regular equality example to which the theorem can be silently applied.

Appendix~\ref{app:residual} supplies the deterministic residual bounds. Appendix~\ref{app:calibration} states the full sampling assumptions and tests, and Appendix~\ref{app:joint} proves the joint block result. The results apply to fixed finite-dimensional collections; growing numbers of horizons or assets require additional theory.

\subsection{Three questions about a horizon profile}
The target must be chosen before inspecting the profile. A fixed horizon asks whether $D_h>0$. A fixed weighted average asks whether $\sum_h\omega_hD_h>0$ and requires aggregating the same joint bootstrap draw before calibration. Persistence asks whether every $D_h$ in the prespecified set is positive. If $p_{\cap,h}$ is valid on each component's covered nonpositive null, then
\[
p_{\mathrm{persist}}=\max_{h\in\mathcal H}p_{\cap,h}
\]
controls the conjunction test without a Bonferroni factor: under its union null, at least one component null is true. This statement does not justify searching for any positive horizon, any attractive subset, or an ex post peak. Those questions require separate multiplicity control or a prespecified signed contrast.

For a balanced panel, calendar-aligned resampling retains common shocks rather than treating assets as independent replications. An unbalanced historical currency panel needs an explicit treatment of its observation masks. Likewise, a rolling window of finite length does not make a persistent underlying process finitely dependent.

The accompanying reference implementation returns the forward and reverse dependence estimates, DII, both bootstrap p-values, fixed signed contrasts, and a conjunction p-value. It takes fitted residuals and aligned regressors as inputs, leaving the training design and economic identification visible. The included synthetic examples use known residuals to illustrate the interface; the larger simulation below evaluates the complete fitted procedure.

'''
resid=span('\\section{Generated residuals under cross-lag dependence}','\\section{Calibration, bootstrap transfer, and horizons}')
resid=resid.replace('\\section{Generated residuals under cross-lag dependence}','\\section{Generated residuals under cross-lag dependence}\\label{app:residual}')
cal=span('\\section{Calibration, bootstrap transfer, and horizons}','\\section{Size, power, and the cost of composite-null protection}')
cal=cal.replace('\\section{Calibration, bootstrap transfer, and horizons}','\\section{Calibration, bootstrap transfer, and horizons}\\label{app:calibration}')
cal=cal.replace('of Section 4.1','defined above').replace('of Section 4','defined above')
sim=span('\\section{Size, power, and the cost of composite-null protection}','\\subsection{Illustrative decision-loss comparison}')
sim=sim.replace('\\section{Size, power, and the cost of composite-null protection}','\\section{Simulation evidence: size, power, and null geometry}\\label{sec:sim}')
finance=r'''\section{Financial transmission and the use of DII}\label{sec:finance}
\subsection{What a shock--asset application can establish}
The financial question is whether an observed shock and a future asset outcome retain asymmetric residual-dependence restrictions after conditioning on a specified history. This is a useful comparison alongside a mean-response path and a risk-response path. It is most interpretable when the shock design, outcome transformation, time aggregation, and information set have a clear economic justification. Identified monetary-policy and oil-supply news provide motivating settings, while their identification strategies remain separate from the restrictions required for DII on the observed series.

A disciplined application distinguishes three layers. External identification supports a statement about the economic shock under the design's own assumptions. DII inference supports a statement about residual asymmetry at the observed horizons under its sampling and modeling conditions. An economic explanation, such as nonlinear exposure, changing risk premia, heterogeneous response timing, or gradual information incorporation, requires evidence that distinguishes it from alternatives. The first two layers do not automatically identify the third.

For finance researchers, the relevant choices concern the specification of shock exposures, the horizon at which a linear model ceases to describe them, and whether an observed pattern requires a richer mechanism. The dynamic exposure in \eqref{eq:exposure} shows that a vanishing linear response can coexist with positive DII without invoking delayed learning. Proposition~\ref{prop:incorporation} additionally shows why an impact-inclusive outcome can retain the shock indefinitely even when all subsequent increments are unpredictable. The outcome interval is therefore part of the financial hypothesis. The scale example shows that meaningful risk dependence can coexist with zero DII. These comparisons make DII an additional model restriction that can help discriminate explanations; they do not turn its magnitude into an estimate of economic importance, a hedge ratio, or an excess-return opportunity.

\subsection{A reporting structure for a financial horizon profile}
A financial study should place the DII profile alongside estimates targeted at the proposed mechanism. For a claim that linear exposure disappears, report the linear response and its uncertainty. For a claim about nonlinear conditional means or risk, fit those objects directly with a justified design. Failure to reject a linear coefficient does not establish that the full conditional mean is zero, and failure to reject a squared-return coefficient does not establish that all higher-order features are absent.

Report $H_f$, $H_b$, and $D_h$ jointly, with both bootstrap p-values and the specified formal test. A positive estimate in many assets is a sign pattern; it is not a rejection of a pooled or persistent null. A three-month peak requires a directly specified comparison against other horizons. Persistence over one, three, and six months requires the conjunction claim, whereas positivity at one horizon or on average does not establish it. Use a common calendar and common resampling path for any panel statistic to which the balanced-panel theorem is applied.

These distinctions support a substantive study of the duration and shape of shock-related asymmetry in asset prices. The methodological paper supplies the estimand, inference, and interpretation boundaries; a financial application supplies the independently substantiated economic pattern and mechanism. It need not repeat the proof to contribute, but it must check that its implementation satisfies the conditions it cites.

\subsection{Scope across business research}
The same question arises when a promotion and future sales, or a disruption and downstream output, admit competing dynamic representations. In those settings, DII may be used to compare distributional restrictions after conditioning on a defensible history. The generality lies in the common target and inference problem, not in assuming that additive-noise or causal-sufficiency restrictions hold in every domain. A managerial action based on the comparison additionally needs a loss or payoff model. Appendix~\ref{app:decision} illustrates that additional step for model validation, including cases where conservatism has higher loss; it is one use of DII rather than the definition of its contribution.

The empirical evidence in this paper consists of controlled simulations and analytical models. No institutional dataset or financial-return result is claimed. A reproducible financial application remains an important opportunity to demonstrate economic relevance beyond those models.

\section{Conclusion}
DII measures directional irreversibility in competing dynamic residual representations. It is distinct from a linear response coefficient and from general distributional dependence. The financial exposure examples show why all three may matter: a nonlinear exposure can be directionally asymmetric with zero linear slope, while a pure risk channel can remain symmetric under DII.

The paper's inferential contribution is the joint treatment of estimated residuals, temporal dependence, and different null limits for fixed dynamic contrasts. Within the stated finite-memory class, the resulting framework supports one-horizon comparisons, prespecified signed aggregates, and conjunction statements about persistence. The simulations make the calibration failures and conservatism explicit.

For finance and other business fields, the reusable contribution is a precise way to ask which dynamic restrictions the data support and over which horizons. Economic identification, statistical asymmetry, and an explanation of the observed pattern remain separate parts of a persuasive application. Keeping those parts explicit makes the methodological foundation useful to subsequent substantive research.

'''
proof=span('\\section{Proof of the joint block approximation}','\\begin{thebibliography}')
refs=s[s.index('\\begin{thebibliography}'):]
refs=refs.replace('\\end{thebibliography}',r'''\bibitem{peters2013} Peters, J., D. Janzing, and B. Sch\"olkopf (2013). Causal inference on time series using restricted structural equation models. \emph{Advances in Neural Information Processing Systems} 26. \url{https://proceedings.neurips.cc/paper/2013/hash/47d1e990583c9c67424d369f3414728e-Abstract.html}.
\end{thebibliography}''')
dec=span('\\subsection{The model-validation decision}','\\section{Generated residuals under cross-lag dependence}')
loss=span('\\subsection{Illustrative decision-loss comparison}','\\section{Implications for financial model validation}')
dec=(dec+loss).replace('Section 5 reports','This appendix reports').replace('in Section 5',r'in Section~\ref{sec:sim}')
final=prefix+intro+related+econ+examples+summary+sim+finance+'\\appendix\n'+resid+cal+proof+'\\section{An optional decision-use illustration}\\label{app:decision}\n'+dec+refs
# Remove old numeric navigation after sections moved to appendices.
final=final.replace('Section 3','Appendix~\\ref{app:residual}') if False else final
final=final.replace('Sch'+chr(92)+chr(92)+chr(34)+'olkopf','Sch'+chr(92)+chr(34)+'olkopf')
final=final.replace('The joint finite-memory theorem covers a fixed balanced panel on one calendar. It does not automatically cover the unequal, changing calendar masks in the source applications. Those calculations remain source results rather than verified applications of the new theorem.','The joint finite-memory theorem covers a fixed balanced panel on one calendar. Unequal, changing calendar masks require an additional sampling argument and are outside the result stated here.')
(S/'main.tex').write_text(final)
(P/'abstract.txt').write_text(a)
print('DII reframing complete; abstract words:',len(a.split()))
