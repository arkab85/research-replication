"""Finance editorial revision. Preserves DII, every proof and numerical result."""
from pathlib import Path
import re
S=Path(__file__).resolve().parent;P=S.parent;p=S/'main.tex';t=p.read_text()
a=t.index(r'\section{Introduction}');b=t.index(r'\subsection{Relationship to existing work}',a)
intro=r'''\section{Introduction}
A financial shock can move prices, change risk, or alter the distribution of subsequent outcomes. These responses answer different economic questions. A mean response measures exposure; a volatility response measures conditional risk. Neither, by itself, establishes whether the joint distribution distinguishes a shock-to-outcome representation from its reverse statistical representation. This paper develops the Directional Irreversibility Index (DII) to measure and infer that distinction at specified horizons.

The question arises when a researcher interprets a fitted transmission equation as a shock response plus a disturbance whose distribution is invariant to the shock and the observed history. DII compares departures from this independent-disturbance restriction in the two representations. It therefore supplies an additional distributional restriction for evaluating a financial transmission model. Its interpretation depends on the shock, the information set, and the outcome interval; those choices are part of the economic question, not merely implementation details.

Consider a position whose exposure to a signed surprise changes with a predetermined state. In a simple benchmark, the shock-to-outcome direction is the same in both states, and DII is positive within each state. Yet pooling the observations makes the shock and outcome independent. The pooled data are indistinguishable from a system with no observable transmission, regardless of sample size. A second benchmark concerns a highly predictable driver: its remaining innovation variance bounds the directional signal at a fixed kernel scale. Together, these results show why a weak DII can reflect the information available to the researcher rather than a weak underlying exposure. They also show why choosing a state definition or normalization to maximize the index would change the research question.

The paper makes three connected contributions. First, it defines a horizon-specific directional contrast and separates it from familiar financial objects. A linear Gaussian exposure has nonzero loading and zero DII; a convex exposure can have zero linear loading and positive DII; and a pure scale exposure can have positive risk dependence and zero DII. Models with identical forward mean and variance functions can also have different DII. These benchmarks identify what the index measures, rather than proposing a replacement for flexible mean or volatility estimation.

Second, the paper characterizes the observability of that contrast. The predictability ceiling and the state-omission result establish different sources of lost directional information. A perturbation bound gives a conservative condition preserving positive DII under a specified state-error mechanism. An immediate-incorporation benchmark further shows that persistence of DII in an impact-inclusive price change does not establish delayed adjustment: the outcome window must distinguish a retained impact from subsequent increments. These results translate the definition into concrete choices about shocks, exposure states, and horizons.

Third, the paper develops joint inference for the original fitted DII across dependent horizons. A difference of two dependence measures requires their joint sampling law. Moreover, equality when both measures vanish has different first-order behavior from a regular equality when both are positive. The procedure carries regression and preprocessing uncertainty through these configurations and reports the components together with the contrast. Approximation bounds make explicit when fitted projection residuals support statements about the original conditional-mean target. Existing residual-independence tests motivate the components; the inferential task here is their fitted directional comparison and its finite family of horizon claims.

The evidence separates population arguments from sampling performance. Existing simulations evaluate calibration, detection, and specification sensitivity: at the 120-observation evaluation size, the joint procedure rejects the studied equal-positive nulls in 3.0--4.3 percent of replications and detects the stronger quadratic alternatives in 70.3--71.3 percent. A targeted moment test performs better in a separate design. The state-observation experiment adds 4,000 paired evaluations from 1,000 independent datasets, comparing oracle and trained DII with known population targets. These experiments establish design-specific performance, not general diagnostic dominance.

A reproducible monetary-shock illustration covers exchange rates, Treasury yields, credit spreads, and implied volatility. None of its 12 primary directional tests rejects at five percent, and supplementary prediction comparisons show no supported gain. Its role is to demonstrate how the method is implemented and how an inconclusive profile is interpreted. The paper's substantive contribution is the methodology and its analytical implications for financial transmission research. It does not depend on establishing a new market anomaly or an investment-performance result.

The intended use is therefore precise: add a distributional comparison to an economically specified shock-response study, assess whether the available information can reveal it, and quantify its uncertainty. For subsequent model use, component bounds also limit the error from independently reusing residuals for specified payoffs; that consequence is secondary to DII's role in measuring direction. Section 2 develops the target and benchmarks, Section 3 the inference, Section 4 the simulation evidence, Section 5 the financial illustration, and Section 6 the implications for financial research design. Complete proofs and supplementary evidence appear in the electronic companion.

'''
t=t[:a]+intro+t[b:]
# Specific finance antecedents: expose complementarity and the outstanding novelty question.
needle=r'\section{DII: the object and its economic meaning}'
finance_lit=r'''Financial prediction and state-dependent exposures supply two further points of comparison. Gu, Kelly, and Xiu (2020) study machine learning in empirical asset pricing; predictive performance concerns a different target from the residual asymmetry examined here. Pelger and Xiong (2022) develop estimation and inference for state-varying factor structures. Our state benchmark takes the shock and state as given and asks whether marginalizing the state erases DII; it neither estimates latent factors nor identifies an unobserved exposure state. These distinctions matter for application: improved forecasts or fitted state-varying loadings do not alone establish an independent-disturbance ordering, and DII does not supply those forecasts or loadings.

'''
t=t.replace(needle,finance_lit+needle,1)
abstract='''Mean responses, risk responses, and directional transmission answer different questions about financial shocks. We develop the Directional Irreversibility Index (DII), a horizon-specific contrast of residual dependence in opposing, history-conditioned representations. Analytical results show how the choice of shock, exposure state, and outcome interval determines its interpretation. Driver predictability limits the available signal; omitting an exposure state can erase positive directional structure completely; and an impact-inclusive footprint can persist under immediate information incorporation. A stability bound gives a local preservation condition under state errors. Joint inference propagates regression and preprocessing uncertainty across dependent horizons and distinct equality-null configurations. Simulations evaluate calibration, specification sensitivity, and oracle versus fitted estimation under imperfect states. A four-market monetary-shock illustration finds no significant primary directional evidence. The framework supplies a distributional restriction and research-design guidance for financial transmission studies, with component bounds supporting a separate, scoped interpretation for model-based decisions.'''
t=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',t,count=1,flags=re.S);(P/'abstract.txt').write_text(abstract+'\n')
a=t.index(r'\section{Financial transmission and the use of DII}');b=t.index(r'\section{Conclusion}',a)
section=r'''\section{Financial transmission and the use of DII}\label{sec:finance}
\subsection{What the methodology changes in a finance study}
The financial contribution is a set of restrictions on how a directional transmission claim is formulated and evaluated. The same index can have different meanings under a different shock measure, history, exposure state, or outcome interval. Table~\ref{tab:financedesign} connects those choices to the paper's results.

\begin{table}[htbp]\centering\small
\caption{Financial research choices informed by DII}\label{tab:financedesign}
\begin{tabular}{>{\raggedright\arraybackslash}p{.23\linewidth}>{\raggedright\arraybackslash}p{.32\linewidth}>{\raggedright\arraybackslash}p{.35\linewidth}}\toprule
Research choice & Relevant result & Implication for the study\\\midrule
Driver level or surprise & Remaining innovation variance bounds the signal & Prespecify an economically justified shock and normalization; a weak level-based DII does not rule out transmission.\\[5pt]
Pooled or state-conditioned observations & Pooling can erase positive DII in every state & Define exposure states from available information; more pooled observations need not recover direction.\\[5pt]
Cumulative or later-increment outcome & A footprint persists under immediate incorporation & Match the outcome interval to a claim about delayed adjustment; cumulative persistence alone is insufficient.\\[5pt]
One horizon or a profile & Joint uncertainty differs across null configurations & Test the actual average, conjunction, or shape claim with common resampling and fitted-model uncertainty.\\[5pt]
Direction or residual reuse & A contrast does not establish either component's adequacy & Report both components; use a component bound and justified payoff class for a model-use claim.\\\bottomrule
\end{tabular}
\end{table}

For example, the state benchmark can describe a signed surprise and a position whose exposure sign depends on a predetermined balance-sheet or hedging state. It demonstrates an information requirement, not an empirical finding about a particular institution. The horizon benchmark addresses a different question: whether a price change retains the original shock or contains a subsequent response. These applications of the theory give finance researchers reasons to define the information set and outcome window before estimating DII.

\subsection{What DII adds to familiar financial evidence}
Mean and risk responses should be estimated directly when they are the financial hypothesis. Positive DII can coexist with zero linear exposure, but a flexible conditional-mean model may detect that exposure. Zero DII can coexist with risk dependence, and a targeted risk diagnostic may be more powerful. The incremental role of DII is to compare the distributional departures in opposing residual representations and to provide inference for that comparison across horizons. Its interpretation is strongest when an independently justified forward additive-noise model makes reverse dependence an economically meaningful restriction.

A financial application should distinguish three layers. Its external design identifies or motivates the shock. DII evaluates directional residual asymmetry under the stated information set and sampling conditions. The economic mechanism explaining that asymmetry requires further evidence. A positive index does not itself identify risk premia, information frictions, or a profitable trading rule. Similarly, a nonsignificant profile is not evidence that transmission is absent. This separation allows the method to support a substantive finance paper while leaving that paper's financial finding and identification as distinct contributions.

\subsection{From a directional profile to a model-use decision}
The reporting object is $(H_f,H_b,D_h)$ under fixed transformations and kernels. Use simultaneous uncertainty for the specified finite family and retain the distinction between average positivity, positivity at every horizon, and a profile's shape. For the original conditional-mean DII, projection-based inference needs correct means or justified approximation bounds. The new state-estimation experiment does not relax these requirements.

If a representation is subsequently used to recombine residuals with shock scenarios, its component upper bound supplies an error budget only for the stated payoff class. Equation~\eqref{eq:regret} and the action-ranking condition specify the additional norm, simulation-error, and optimization requirements. The constructed protection example makes the need for component reporting concrete, but it is not evidence of an investment return. The monetary-shock illustration demonstrates the reporting procedure with an inconclusive outcome; no decision improvement is inferred from it.

'''
t=t[:a]+section+t[b:]
a=t.index(r'\section{Conclusion}');b=t.index(r'\clearpage',a)
t=t[:a]+r'''\section{Conclusion}
DII measures a specific distributional asymmetry in competing shock-response representations. Its contribution to finance is the combination of that target, results showing when the available information can reveal it, and joint inference for its fitted horizon profile. Mean exposure, conditional risk, and directional structure remain distinct objects.

The analytical results change three research-design choices. Driver innovation determines an upper limit on the signal. Observing an exposure state can determine whether directional structure is visible at all. The outcome interval determines whether a persistent footprint can speak to delayed adjustment. These conclusions give the method financial content before any application establishes a new empirical mechanism.

The simulations document both useful performance and meaningful limitations, including specification sensitivity, training costs, and settings in which targeted moments perform better. The financial illustration finds no significant primary directional evidence. A substantive application can use the foundation to support a carefully specified transmission claim; its identification, mechanism, and economic magnitudes remain additional empirical contributions.

'''+t[b:]
refs=r'''\bibitem{gkx} Gu, S., B. Kelly, and D. Xiu (2020). Empirical Asset Pricing via Machine Learning. \emph{Review of Financial Studies} 33(5), 2223--2273. \url{https://academic.oup.com/v/article/33/5/2223/5758276}.
\bibitem{pelgerxiong} Pelger, M., and R. Xiong (2022). State-Varying Factor Models of Large Dimensions. \emph{Journal of Business \& Economic Statistics} 40(3), 1315--1333. Working-paper version: \url{https://arxiv.org/abs/1807.02248}.
'''
t=t.replace(r'\end{thebibliography}',refs+r'\end{thebibliography}',1)
t=t.replace(r'\begin{thebibliography}{99}',r'\begin{thebibliography}{99}'+ '\n' + r'\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}',1)
p.write_text(t)
(S/'cover_letter.md').write_text('''Dear Finance Department Editors,

Please consider “Directional Irreversibility in Economic Dynamics: Inference and Shock Transmission” for the journal.

The paper asks when a financial shock-response model supports a directional interpretation beyond its mean and volatility responses. It develops the Directional Irreversibility Index (DII), its observability properties, and joint inference for fitted horizon profiles. The intended readership is financial researchers studying shock transmission and evaluating the distributional restrictions of dynamic models.

The analytical contribution has direct implications for research design. Driver predictability limits the available directional signal. Omitting an exposure state can erase positive DII in every state, producing a pooled law indistinguishable from independence. An impact-inclusive directional footprint can persist even under immediate information incorporation. These results inform the selection of shocks, information sets, and outcome intervals. A state-error bound describes a conservative local preservation condition.

The inferential contribution addresses the joint comparison of opposing fitted dependence measures across dependent horizons, including regression and preprocessing uncertainty and distinct equality-null configurations. The paper builds on established residual-independence and additive-noise methods; it does not claim to originate those principles. Simulations assess calibration, specification sensitivity, and estimation, including 4,000 paired state-error evaluations from 1,000 independent datasets.

The submission is a foundational financial-methodology paper. Its monetary-shock illustration reports no significant primary directional finding, and its conceptual results do not rely on a new return-prediction or trading-performance claim. A separate financial application remains responsible for its own identification, mechanism, and economic magnitudes.

We believe the manuscript fits the Finance Department's interest in conceptual and empirical-methodological contributions with substantive financial relevance. Thank you for considering it.

Arka Prava Bandyopadhyay

Author note: Unsent draft. Complete declarations and related-paper details from the actual submission circumstances.
''')
(S/'editor_strategy.md').write_text('''---
title: "DII: Finance department submission strategy"
author: "Prepared for Arka Prava Bandyopadhyay"
date: "13 September 2026"
---

### Recommended route

Submit as financial methodology to the **Finance Department**. The [departmental statement](https://pubsonline.informs.org/page/v/editorial-statement) welcomes conceptual and empirical-methodological advances with financial relevance. Lead with the distributional interpretation of shock transmission and the resulting choices about shocks, exposure states, and horizons.

### reviewer and AE fit

The [current board](https://pubsonline.informs.org/page/v/editorial-board) lists Jianqing Fan as a Finance department reviewer and Dacheng Xiu and Markus Pelger as Finance associate editors. My recommendations reflect subject fit, not predicted favorability:

- **Department reviewer: Jianqing Fan.** His [financial econometrics, statistics, and machine-learning expertise](https://fan.princeton.edu/) fits the paper's fitted-model inference and methodological contribution.
- **First AE suggestion: Dacheng Xiu.** His [work on financial inference and empirical asset-pricing methods](https://dachxiu.chicagobooth.edu/) fits the central comparison and its uncertainty.
- **Alternative AE: Markus Pelger.** His [financial risk, statistical theory, and state-varying exposure research](https://mpelger.people.stanford.edu/) fits the information-set and state-observation results.

Use these suggestions only where the submission system permits and after checking actual conflicts. The journal determines assignments. No contact has been made.

### Argument the manuscript now presents

DII adds a distributional comparison to a specified financial shock-response study. The predictability, state-observation, and immediate-incorporation results identify when that comparison is observable and how to interpret its horizon profile. Joint inference addresses fitted regressions, preprocessing, and the two equality-null configurations. A new research-design table makes the implications concrete.

The literature comparison clarifies the contribution relative to residual-independence tests, prediction, and state-varying factors. All existing evidence is retained.

### Independent foundation for the finance agenda

The methods paper supplies theory, inference, and synthetic benchmarks. journal retains its separate financial finding and mechanism. The magnitude of the methodological advance remains an editorial judgment.
''')
# Keep build-generated support documents concise on repeated rebuilds.
for name in ['README.md','RFS_foundation_bridge.md']:
 f=P/name;parts=re.split(r'(?=^## )',f.read_text(),flags=re.M);out=[];seen=set()
 for part in parts:
  heading=part.splitlines()[0] if part else ''
  if heading in seen:continue
  seen.add(heading);out.append(part)
 f.write_text(''.join(out))
print('Finance editorial revision integrated; abstract words',len(abstract.split()))
