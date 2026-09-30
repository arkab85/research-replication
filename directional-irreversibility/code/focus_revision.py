"""Editorial focus pass; preserve estimand, proofs, numerical results and protocols."""
from pathlib import Path
import re
S=Path(__file__).resolve().parent; P=S.parent
p=S/'main.tex'; t=p.read_text()
abstract=r'''How can researchers distinguish directional structure in shock transmission from a mean response or a change in risk? The Directional Irreversibility Index (DII) compares residual dependence in opposing, history-conditioned representations. Analytical financial exposures show that a linear response can coexist with zero DII, a nonlinear exposure can have positive DII despite a zero linear response, and risk dependence can cancel in the contrast. We develop joint inference for fitted DII across dependent horizons, propagating regression and preprocessing uncertainty and accommodating distinct null configurations. Simultaneous component bounds separate directional asymmetry from approximate validity of a proposed representation. Simulations establish design-specific calibration and power, while identifying costs relative to targeted moment tests and sensitivity to misspecified means. A four-market monetary-shock application illustrates the framework without finding significant primary directional evidence. DII supplies a distinct restriction for evaluating dynamic models; its financial interpretation requires an independently justified shock design, outcome window, and mechanism.'''
t=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',t,count=1,flags=re.S)
(P/'abstract.txt').write_text(abstract+'\n')
intro=r'''\section{Introduction}
A shock can leave a nonlinear footprint in asset prices even when its linear response is zero. Conversely, a shock can change financial risk without distinguishing the two directions of a statistical relationship. For researchers evaluating a model of shock transmission, these possibilities raise a precise question: what evidence distinguishes directional structure from a mean response or a change in risk? The answer matters when an empirical model treats disturbances as independent of the shock and the observed history, as in an additive-noise representation used to interpret transmission or construct shock scenarios.

This paper studies that question through the Directional Irreversibility Index (DII). At a specified horizon, DII compares residual dependence in two opposing, history-conditioned representations. Its components measure departures from independent additive noise; their difference measures directional asymmetry. The reverse representation supplies a statistical comparison, not a claim that a future price causes a past shock. Under an independently justified forward additive-noise model and a nonreversibility restriction, positive DII has an ordering interpretation. Outside that class, it remains a contrast of specification departures.

Three financial exposures make the distinction concrete. A linear Gaussian exposure has a nonzero shock loading and zero DII. A convex quadratic exposure has zero linear loading and positive DII. A pure scale exposure has shock-dependent risk but zero DII because its two dependence components cancel. Thus the proposed object can reveal an asymmetry missed by a linear response, while deliberately leaving other economically important forms of dependence unresolved. A further benchmark holds forward mean and variance functions fixed while DII changes. These are statements about distinct population restrictions, rather than a claim that a general kernel test always outperforms a test tailored to a known moment.

The central inferential difficulty is that equality of the two dependence components has different sampling behavior depending on what is equal. When both are zero, the contrast has a quadratic limit; when both are positive at a regular equality, it has a linear limit. Fitted regressions, learned normalization, and dependence across horizons also affect the contrast. We develop joint block inference that carries these sources of uncertainty through the original DII estimator. A separate simultaneous bound reports the two components along with their contrast, permitting a researcher to distinguish positive asymmetry from approximate validity of the forward representation. Explicit approximation bounds state what is needed when fitted regressions are projections rather than the true conditional means.

The paper contributes a defined dynamic target, inference for its fitted implementation, and analytical guidance for interpreting financial horizon profiles. The contribution builds on residual-independence testing and additive-noise identification; it is the joint treatment of opposing fitted representations and their horizon comparisons that is developed here. This focus connects the methodology to a concrete research choice: whether a proposed shock-transmission representation satisfies a distributional restriction that mean and risk responses alone do not settle.

The evidence evaluates that claim at three levels. Analytical examples establish the population distinctions. Simulations assess calibration and detection with trained models, including common preprocessing uncertainty. In the reported 120-observation evaluation designs, the joint procedure rejects equal-positive nulls in 3.0--4.3 percent of replications and detects stronger quadratic exposures in 70.3--71.3 percent. Power is lower for weak exposures, and a focused reverse-moment test is substantially more powerful in a separate known-moment design. Misspecified conditional means can also create apparent directionality, which motivates the specification bound. These results support a scoped inferential contribution, not universal diagnostic superiority.

A fixed monetary-shock application examines later changes in exchange rates, Treasury yields, credit spreads, and implied volatility. None of its 12 primary directional tests rejects at five percent; supplementary mean and forecast-risk comparisons also yield no supported positive gain. The application illustrates estimation and interpretation rather than establishing a new transmission mechanism. The financial relevance of DII instead begins with the analytical distinction: an impact-inclusive directional footprint can persist under immediate information incorporation, so a claim about delayed adjustment must use an outcome window and evidence that distinguish that mechanism. This makes the methods useful as a foundation for a separate financial study without substituting for its identification or empirical finding.

Section 2 defines DII and develops the financial benchmarks. Section 3 presents estimation and inference. Section 4 reports the main validation and its practical limits. Section 5 gives the financial application, and Section 6 explains the implications for transmission research. The electronic companion provides proofs, complete benchmark comparisons, and implementation extensions.

\subsection{Relationship to existing work}
DII uses the residual-independence principle developed in additive-noise models by Peters, Mooij, Janzing, and Sch\"olkopf (2014) and, for time-series structural models, Peters, Janzing, and Sch\"olkopf (2013). The linear non-Gaussian benchmark draws on the identification insight of Shimizu et al. (2006). These antecedents establish why independent disturbances can distinguish representations. Here the object is a horizon-specific contrast between two departures from that restriction, with an ordering interpretation only under the maintained model class.

The dependence measure is the kernel criterion of Gretton et al. (2008). Sen and Sen (2014) study residual independence and goodness of fit with estimated linear regressions, providing a close antecedent for generated-residual testing. Chwialkowski, Sejdinovic, and Gretton (2014, revised 2016) develop dependent wild-bootstrap methods, and Dehling, Sharipov, and Wendler (2014) provide Hilbert-space block-bootstrap results used in our persistent-dependence argument. We build on these tools to propagate common training and evaluation uncertainty through a finite signed family of fitted dependence operators. The joint law matters because separate marginal approximations do not establish a valid approximation for their difference. The intersection-union principle and the underlying bootstrap machinery are established methods.

The economic target complements Granger's (1969) incremental predictability and impulse responses estimated by local projections (Jord\`a, 2005). Mean responses, risk responses, and residual asymmetry impose different restrictions, as the financial examples demonstrate. External shock designs such as Romer and Romer (2004), Jaroci\'nski and Karadi (2020), and K\"anzig (2021) can motivate applications, but do not automatically establish the additive independent-noise restrictions for an observed proxy at every horizon. The present contribution concerns inference and interpretation for the directional comparison under stated conditions.

'''
a=t.index(r'\section{Introduction}'); b=t.index(r'\section{DII:',a); t=t[:a]+intro+t[b:]
# Retain all baseline experiments, favorable and adverse, together in the companion.
a=t.index(r'\subsection{Common implementation}'); b=t.index(r'\subsection{Joint coefficient',a)
baseline=t[a:b]
t=t[:a]+r'''The main validation evaluates the original DII estimator with uncertainty from trained coefficients, scales, and centers. Two supplementary comparisons ask distinct questions: whether the equality-null geometry matters for calibration, and how DII compares with a test of a specified residual moment. All simulation results, including unfavorable and out-of-scope cases, are retained in the electronic companion.

\subsection{Why the null configuration matters}
The baseline study uses 4,800 datasets and separate large training samples to isolate the two null configurations. At an equal-positive null, wild-only rejection is 40.7--44.3 percent at a nominal five-percent level; paired and intersection rejection are both 5.0 percent. The paired and intersection decisions coincide throughout the feasible baseline study, so these results do not demonstrate an incremental power or size advantage of the intersection over paired-only testing. At double independence the intersection is conservative. Appendix~\ref{app:baseline} reports the full design, all component rates, near-degenerate cases, cancellation, and insufficient-training stress tests.

'''+t[b:]
# Move full moment comparison intact; leave a substantive, quantitative main-text summary.
a=t.index(r'\subsection{What does DII add'); b=t.index(r'\section{A reproducible',a)
moments=t[a:b].replace(r'\label{sec:momentvalidation}',r'\label{app:momentvalidation}')
t=t[:a]+r'''\subsection{Comparison with a targeted residual moment}\label{sec:momentvalidation}
DII measures a broader restriction than a specified moment, but that breadth can cost power. A matched 2,400-dataset experiment compares DII with forward and reverse tests of $\operatorname{Cov}(u^2,z_1^2)$, using the same correctly specified affine means, trained preprocessing, and block draws. At $n=120$ with Laplace innovations, DII rejection is 8.5 percent under independence and 1.5 percent with persistence; the reverse-moment test rejects 53.5 and 36.0 percent. At $n=480$, the corresponding rates are 85.0 and 43.5 percent for DII, versus 93.5 and 93.0 percent for the targeted test.

For the matched-fourth-moment law in Section~\ref{sec:momentmatch}, the reverse squared moment is zero while population DII is positive. Nevertheless, DII detection is only 0.5--1.0 percent at $n=120$ and 0.5--7.5 percent at $n=480$. Thus a population distinction does not ensure useful detection in a particular sample. The methods test different nulls: a targeted-moment rejection does not by itself establish directional asymmetry. Researchers should choose the restriction from the economic hypothesis; the present evidence supports neither universal power superiority nor interpreting a DII non-rejection as absence. Appendix~\ref{app:momentvalidation} reports every cell and its design.

'''+t[b:]
# Move auxiliary forecasting and post-results audit details together, keeping all primary financial tests in main.
a=t.index(r'\subsection{An independent prediction comparison}'); b=t.index(r'\section{Financial transmission',a)
finance=t[a:b].replace(r'\label{sec:riskdecision}',r'\label{app:riskdecision}')
# Remove drafting-specific references to journal from reader-facing paper, retaining accurate general distinction.
finance=finance.replace("These qualifications mean that the exercise cannot be used to declare the journal companion's empirical conclusions disproved: its series, calendar, estimator, and tests differ. It also cannot be used to strengthen those conclusions by importing isolated favorable sensitivity results. A substantive finance contribution would require stable incremental evidence and a sampling argument appropriate to the actual data process.","These results concern this application's series, calendar, estimator, and tests. They do not establish or refute results from a different empirical design. A substantive transmission claim requires evidence and a sampling argument appropriate to its own data process.")
t=t[:a]+r'''\subsection{Interpretation and supplementary checks}\label{sec:riskdecision}
The primary results provide no significant evidence of positive DII in this design. They also do not establish that the population contrasts vanish. Conditional-mean approximation, stability, and dependence assumptions remain consequential: the sample split and observed autocorrelations do not verify the theorem's conditions for these series. The reported block p-values are diagnostics under maintained working approximations.

Supplementary analyses preserve the same financial cells. Propagating joint coefficient, scale, and centering uncertainty reproduces the original DII point estimates exactly and yields no unadjusted rejection or positive simultaneous directional lower bound. Separate comparisons of quadratic mean prediction and a nonnegative shock-square forecast-risk predictor find no supported positive gain. These analyses concern different targets and do not convert the primary non-rejections into evidence of no economic transmission. Appendix~\ref{app:financechecks} gives the complete forecasting results, secondary procedures, and empirical qualifications.

The application therefore demonstrates how the proposed contrast can be implemented and reported alongside conventional financial questions. Its role is illustrative. The analytical results establish why those questions differ; evidence for a particular financial mechanism must come from a design that identifies and tests that mechanism.

'''+t[b:]
# Replace conclusion with contribution-centered, bounded statement.
a=t.index(r'\section{Conclusion}'); b=t.index(r'\clearpage',a)
t=t[:a]+r'''\section{Conclusion}
Directional irreversibility is a distinct feature of a dynamic representation. A shock can have a nonzero mean response with no directional separation, a nonlinear exposure can generate positive DII despite a zero linear loading, and risk dependence can cancel in the contrast. These distinctions give financial researchers a specific additional restriction to examine when interpreting shock transmission.

The paper supplies joint inference for fitted DII and fixed horizon comparisons, including regression and preprocessing uncertainty. Reporting the two dependence components alongside their difference separates asymmetry from approximate model validity. The guarantees require the stated sampling conditions and correct conditional means or justified approximation bounds. Simulations show useful detection for the stronger quadratic exposures studied, substantial costs in other designs, and an advantage for targeted moments when their restriction fits the question. The financial illustration finds no significant primary directional evidence. DII's contribution is a defined and testable feature of competing representations; a substantive claim about information absorption or asset pricing requires the additional economic evidence specified by the application.

'''+t[b:]
# Append moved material without renumbering existing theorem-bearing appendices.
extra='\n'+r'\section{Complete baseline simulation evidence}\label{app:baseline}'+'\n'+baseline+'\n'+r'\section{Complete matched moment comparison}'+'\n'+moments+'\n'+r'\section{Supplementary financial comparisons}\label{app:financechecks}'+'\n'+finance+'\n'
t=t.replace(r'\end{document}',extra+r'\end{document}')
p.write_text(t)
(S/'cover_letter.md').write_text('''Dear Editors,

Please consider “Directional Irreversibility in Economic Dynamics: Inference and Shock Transmission” for the journal.

The paper asks what directional structure reveals about shock transmission beyond mean responses and changes in risk. Financial benchmarks make the distinction precise: a linear Gaussian exposure can have a nonzero loading and zero DII, a quadratic exposure can have positive DII with zero linear loading, and pure risk dependence can cancel in the directional contrast.

The methodological contribution is joint inference for opposing fitted residual-dependence representations and fixed horizon comparisons. The analysis incorporates training and preprocessing uncertainty, distinguishes the relevant null configurations, and provides component and specification bounds that delimit the interpretation of directional evidence. It builds explicitly on established additive-noise identification, residual-independence testing, and block-bootstrap methods.

The main paper concentrates on the financial question, core inferential results, and validation of the fitted implementation. The electronic companion contains the proofs and all supplementary comparisons. The evidence is balanced: targeted moment tests have an advantage in some designs, and the four-market illustration has no significant primary directional finding. The paper makes a methodological contribution to evaluating dynamic models, rather than claiming a new empirical transmission mechanism.

Thank you for considering the manuscript.

Arka Prava Bandyopadhyay

Author note: Unsent draft. Complete the submission system’s declarations from the actual submission circumstances.
''')
(S/'editor_strategy.md').write_text('''---
title: "DII: focused submission positioning"
author: "Prepared for Arka Prava Bandyopadhyay"
---

### Submission argument

The paper's central question is what directional structure adds to the analysis of shock transmission beyond mean responses and changes in risk. Its case rests on three connected elements: financial benchmarks that establish the distinction, joint inference for the fitted directional contrast, and evidence describing where that implementation works.

The revision leads with this argument and gives the financial audience a concrete reason to read the methods. The main paper retains the full primary financial results and quantitative summaries of material limitations. Complete baseline simulations, targeted-moment comparisons, forecasting exercises, and secondary financial audits are grouped in the companion. No results were retuned or discarded.

### Relationship to the journal paper

The methodological paper supplies the target, applicable inference, and interpretation of horizon profiles. The journal paper supplies the distinct financial phenomenon, identification, and mechanism. A citation can support a matching estimator under its assumptions; it does not upgrade an existing p-value. This revision does not change the journal paper or its evidence.

### What remains material to publication

The revision addresses focus, accessibility, and the articulation of the contribution. It does not establish that an reviewer will judge the methodological advance sufficient for the journal. The main empirical illustration has no significant directional finding, and the method has low power in some reported designs. These findings remain visible because they qualify the contribution. There is no defensible numerical R&R estimate from the available evidence.

The cover letter now makes the same focused methodological case as the manuscript. No submission or reviewer contact has been made.
''')
(P/'README.md').write_text((P/'README.md').read_text()+'\n## Focused editorial revision\n\nThe main paper centers the DII question and its financial interpretation. Complete baseline experiments, the matched moment comparison, and secondary financial comparisons now appear in the companion, with material findings summarized in the main text. This is an editorial revision: numerical results, proofs, estimand, and research protocols are preserved. `source/focus_revision.py` is the final narrative build stage.\n')
print('Focused manuscript and supporting submission text generated; abstract words',len(abstract.split()))
