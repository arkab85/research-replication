"""the journal submission stage.

Runs after finance_revision.py. Editorial only: no proof, estimand,
simulation, table value, or financial estimate is changed. It
  * rewrites the abstract, JEL codes and keywords for the Finance Department;
  * opens the introduction with the identified-shock literature in finance;
  * adds positioning against time-reversibility tests, non-Gaussian SVAR
    identification, nonlinear Granger causality, connectedness, state-dependent
    and distributional local projections;
  * moves the action-ranking certificate and the exact protection example from
    the main text to the electronic companion (all content retained);
  * removes the one journal-identifying phrase in the companion;
  * replaces the bibliography with an alphabetical INFORMS-style list using
    published versions where they exist.
"""
from pathlib import Path

S = Path(__file__).resolve().parent
T = S / 'main.tex'
tex = T.read_text()


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:90])
    tex = tex.replace(old, new)


def cut(start, end):
    """Remove and return the text from start (inclusive) to end (exclusive)."""
    global tex
    i = tex.index(start)
    j = tex.index(end, i)
    block = tex[i:j]
    tex = tex[:i] + tex[j:]
    return block


# ---------------------------------------------------------------- abstract
a0 = tex.index('\\begin{abstract}') + len('\\begin{abstract}\n')
a1 = tex.index('\\end{abstract}')
ABSTRACT = (
    "Empirical finance studies the transmission of identified shocks, such as monetary-policy surprises, "
    "mainly through mean and volatility responses. These responses do not reveal whether the joint "
    "distribution favors a shock-to-outcome representation over its reverse. We develop the Directional "
    "Irreversibility Index (DII), a horizon-specific contrast of residual dependence in opposing, "
    "history-conditioned representations, and characterize when financial data can reveal it. Three "
    "analytical results translate the index into research-design guidance. A predictability bound limits "
    "the directional signal by the driver's remaining innovation variance, favoring surprise measures over "
    "levels. Omitting an exposure state can erase positive directional structure present in every state, so "
    "pooled data can be indistinguishable from no transmission at any sample size; a stability bound gives a "
    "local preservation condition under state misclassification. An immediate-incorporation benchmark shows "
    "that a persistent footprint in an impact-inclusive price change is consistent with a martingale price, "
    "so claims of delayed absorption require post-impact outcome windows. For inference, a joint block "
    "bootstrap for fitted horizon profiles propagates regression and preprocessing uncertainty and "
    "accommodates the nonregular null in which both dependence components vanish. In a 3,000-dataset "
    "validation the procedure rejects equal-positive nulls in 3.0--4.3 percent of replications and detects "
    "the stronger quadratic alternatives in about 70 percent at 120 evaluation observations. A prespecified four-market "
    "monetary-shock illustration finds no significant primary directional evidence and shows how an "
    "inconclusive profile should be reported. Component bounds also limit the error from reusing fitted "
    "residuals in scenario-based decisions.\n")
tex = tex[:a0] + ABSTRACT + tex[a1:]

rep("\\noindent JEL: C12, C14, C32.\\\\\nKeywords: directional irreversibility; financial shocks; residual independence; dynamic models; bootstrap inference.",
    "\\noindent JEL: C12, C14, C32, G12, G14, E44.\\\\\nKeywords: shock transmission; directional irreversibility; residual independence; monetary-policy surprises; bootstrap inference.")

# ---------------------------------------------------------------- introduction opening
p1 = "A financial shock can move prices, change risk, or alter the distribution of subsequent outcomes."
p3 = "Consider a position whose exposure to a signed surprise changes with a predetermined state."
i0 = tex.index(p1); i1 = tex.index(p3)
INTRO = (
    "Identified shocks have become the workhorse of empirical research on financial transmission. "
    "High-frequency monetary-policy surprises (Gertler and Karadi 2015, Nakamura and Steinsson 2018, "
    "Jaroci\\'nski and Karadi 2020), narrative measures (Romer and Romer 2004), and oil-supply news "
    "(K\\\"anzig 2021) are routinely related to yields, spreads, exchange rates, equity prices, and implied "
    "volatility (Bernanke and Kuttner 2005, Hanson and Stein 2015) through local projections (Jord\\`a 2005) "
    "or external-instrument designs (Stock and Watson 2018). The resulting mean responses measure exposure, "
    "and volatility responses measure conditional risk. Neither establishes whether the joint distribution "
    "of the shock and the later outcome distinguishes a shock-to-outcome representation from its reverse "
    "statistical representation. This paper develops the Directional Irreversibility Index (DII) to measure "
    "and conduct inference on that distinction at specified horizons, and characterizes when financial data "
    "can reveal it.\n\n"
    "The question arises whenever a researcher reads a fitted transmission equation as a shock response plus "
    "a disturbance whose distribution is invariant to the shock and the observed history. DII compares "
    "departures from this independent-disturbance restriction in the two representations, supplying a "
    "distributional restriction that mean and volatility responses do not test. Its interpretation depends "
    "on the shock, the information set, and the outcome interval; the paper shows that these choices are "
    "part of the economic question rather than implementation details.\n\n")
tex = tex[:i0] + INTRO + tex[i1:]

old_app = tex[tex.index("A reproducible monetary-shock illustration covers exchange rates"):tex.index("The intended use is therefore precise")]
tex = tex.replace(old_app,
    "A reproducible monetary-shock illustration covers exchange rates, Treasury yields, credit spreads, and "
    "implied volatility under a design fixed before estimation. None of its 12 primary intersection tests, "
    "which require rejection under both null configurations, rejects at five percent, although the "
    "degenerate-null component alone falls below 0.05 in seven cells. The contrast illustrates the "
    "calibration problem the joint procedure addresses. Supplementary prediction comparisons show no "
    "supported gain. The illustration demonstrates implementation and the reporting of an inconclusive "
    "profile; the paper's contribution is methodological and does not rest on a new anomaly or "
    "trading result.\n\n")

rep("For subsequent model use, component bounds also limit the error from independently reusing residuals for specified payoffs; that consequence is secondary to DII's role in measuring direction.",
    "For risk managers and investors who combine shock scenarios with fitted residuals, the component bounds also limit the decision error from independent residual reuse for specified payoff classes; that consequence is secondary to DII's role in measuring direction.")

# ---------------------------------------------------------------- related work
rep("Here the object is a horizon-specific contrast between two departures from that restriction, with an ordering interpretation only under the maintained model class.\n",
    "Here the object is a horizon-specific contrast between two departures from that restriction, with an ordering interpretation only under the maintained model class.\n\n"
    "The name should not be confused with tests of time reversibility (Ramsey and Rothman 1996, Chen et al. 2000), "
    "which compare the law of a stationary process with that of its time reversal and have been used to document "
    "business-cycle asymmetry. DII instead contrasts two regression representations of a specified shock and a "
    "later outcome given a fixed history. Non-Gaussian structural VARs (Lanne et al. 2017, Gouri\\'eroux et al. 2017) "
    "exploit independence and non-Gaussianity of structural shocks to identify impact matrices. Our linear "
    "non-Gaussian benchmark is consistent with that insight, but DII does not estimate a structural system; it "
    "applies to a single externally measured shock with possibly nonlinear responses.\n")

rep("The intersection-union principle and the underlying bootstrap machinery are established methods.",
    "The intersection-union principle (Berger 1982) and the underlying bootstrap machinery are established methods.")
rep("Chwialkowski, Sejdinovic, and Gretton (2014, revised 2016)", "Chwialkowski, Sejdinovic, and Gretton (2014)")
tex = tex.replace("Wendler (2014", "Wendler (2015")

rep("The present contribution concerns inference and interpretation for the directional comparison under stated conditions.\n",
    "The present contribution concerns inference and interpretation for the directional comparison under stated conditions. "
    "Local projections and VARs estimate the same population impulse responses (Plagborg-M{\\o}ller and Wolf 2021), "
    "and state-dependent local projections (Tenreyro and Thwaites 2016, Ramey and Zubairy 2018) and quantile "
    "responses (Adrian et al. 2019) extend mean responses to regimes and to the full conditional distribution. "
    "Our state-omission result complements that literature: pooling across exposure states can erase directional "
    "structure that is present in every state, a stronger failure than the attenuation of pooled mean responses.\n\n"
    "Finance also uses directional statistics in other senses. Nonlinear Granger-causality tests (Hiemstra and Jones 1994) "
    "and Granger-causality networks (Billio et al. 2012) examine incremental predictability, and connectedness "
    "measures (Diebold and Y\\i lmaz 2014) decompose forecast-error variances. These objects are built from predictive "
    "content and variance shares. DII is a residual-independence contrast: it is zero in a jointly Gaussian system with "
    "strong predictability and can be positive when linear predictability is absent (Section~\\ref{sec:object}). "
    "Generalized spectral tests of nonlinear serial dependence (Hong 1999) are a related residual diagnostic for a "
    "single series, and our conditional-mean estimation with dependent data follows standard nonparametric "
    "time-series practice (Fan and Yao 2003).\n")

# ---------------------------------------------------------------- application: finance anchors and Holm citation
rep("Does an observed monetary-policy surprise retain directional residual asymmetry in a later financial change, and is that asymmetry accompanied by useful nonlinear response prediction?",
    "Monetary-policy surprises move equity prices, long-term yields, and credit spreads (Bernanke and Kuttner 2005, Gertler and Karadi 2015, Hanson and Stein 2015). Does an observed surprise also retain directional residual asymmetry in a later financial change, and is that asymmetry accompanied by useful nonlinear response prediction?")
rep("All 12 primary tests are reported, together with Holm adjustment.",
    "All 12 primary tests are reported, together with Holm (1979) adjustment.")

# ---------------------------------------------------------------- move decision example to the companion
rep("\\subsection{From directional diagnostics to the cost of a model-based decision}\\label{sec:replay}",
    "\\subsection{Component bounds and the cost of model-based decisions}\\label{sec:replay}")
moved = cut("A more targeted certificate compares two actions.", "\\subsection{Persistent DII need not mean delayed incorporation}")
tex = tex.replace("\\subsection{Persistent DII need not mean delayed incorporation}",
    "Appendix~\\ref{app:decisionexample} gives a sharper pairwise action-ranking certificate and an exact example in which "
    "independent residual reuse reverses a stylized protection-purchase decision although DII is zero. Zero directional "
    "asymmetry therefore does not establish that independent residual reuse is suitable for a financial decision; the "
    "component inference is needed for that claim.\n\n"
    "\\subsection{Persistent DII need not mean delayed incorporation}", 1)
moved = moved.replace("The older forward-validity and scenario-score bounds remain in Appendix~\\ref{app:replay}.",
                      "The forward-validity and scenario-score bounds are in Appendix~\\ref{app:replay}.")
rep("\\subsection{Exact protection example and numerical verification}",
    "\\subsection{Action-ranking certificate and an exact protection example}\\label{app:decisionexample}\n"
    + moved.rstrip() + "\n\n\\subsection{Exact protection example and numerical verification}")
rep("The eight joint states in the main text are equally likely.", "The eight joint states above are equally likely.")
rep("The ranking certificates in the main text follow directly.",
    "The ranking certificates in Appendix~\\ref{app:decisionexample} follow directly.")
rep("Equation~\\eqref{eq:regret} and the action-ranking condition specify the additional norm, simulation-error, and optimization requirements. The constructed protection example makes the need for component reporting concrete, but it is not evidence of an investment return.",
    "Equation~\\eqref{eq:regret} and the action-ranking condition in Appendix~\\ref{app:decisionexample} specify the additional norm, simulation-error, and optimization requirements. The constructed protection example there makes the need for component reporting concrete, but it is not evidence of an investment return.")

# ---------------------------------------------------------------- companion anonymity
rep("For an journal shock study this permits", "For an applied shock study this permits")
rep("(Chernozhukov et al., 2016a,b)", "(Chernozhukov et al. 2018, 2022)")

# ---------------------------------------------------------------- bibliography
BIB = r"""
\bibitem{abg} Adrian T, Boyarchenko N, Giannone D (2019) Vulnerable growth. \emph{Amer. Econom. Rev.} 109(4):1263--1289.
\bibitem{berger} Berger RL (1982) Multiparameter hypothesis testing and acceptance sampling. \emph{Technometrics} 24(4):295--300.
\bibitem{bk} Bernanke BS, Kuttner KN (2005) What explains the stock market's reaction to Federal Reserve policy? \emph{J. Finance} 60(3):1221--1257.
\bibitem{bglp} Billio M, Getmansky M, Lo AW, Pelizzon L (2012) Econometric measures of connectedness and systemic risk in the finance and insurance sectors. \emph{J. Financial Econom.} 104(3):535--559.
\bibitem{cck} Chen YT, Chou RY, Kuan CM (2000) Testing time reversibility without moment restrictions. \emph{J. Econometrics} 95(1):199--218.
\bibitem{dml} Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, Robins J (2018) Double/debiased machine learning for treatment and structural parameters. \emph{Econometrics J.} 21(1):C1--C68.
\bibitem{lr} Chernozhukov V, Escanciano JC, Ichimura H, Newey WK, Robins JM (2022) Locally robust semiparametric estimation. \emph{the journal} 90(4):1501--1535.
\bibitem{csg} Chwialkowski K, Sejdinovic D, Gretton A (2014) A wild bootstrap for degenerate kernel tests. \emph{Adv. Neural Inform. Processing Systems} 27.
\bibitem{dsw} Dehling H, Sharipov OSh, Wendler M (2015) Bootstrap for dependent Hilbert space-valued random variables with application to von Mises statistics. \emph{J. Multivariate Anal.} 133:200--215. Theorem numbers cited in the text follow arXiv:1312.3870v3.
\bibitem{dy} Diebold FX, Y\i lmaz K (2014) On the network topology of variance decompositions: Measuring the connectedness of financial firms. \emph{J. Econometrics} 182(1):119--134.
\bibitem{fanyao} Fan J, Yao Q (2003) \emph{Nonlinear Time Series: Nonparametric and Parametric Methods} (Springer, New York).
\bibitem{gk} Gertler M, Karadi P (2015) Monetary policy surprises, credit costs, and economic activity. \emph{Amer. Econom. J. Macroeconom.} 7(1):44--76.
\bibitem{gmr} Gouri\'eroux C, Monfort A, Renne JP (2017) Statistical inference for independent component analysis: Application to structural VAR models. \emph{J. Econometrics} 196(1):111--126.
\bibitem{granger} Granger CWJ (1969) Investigating causal relations by econometric models and cross-spectral methods. \emph{the journal} 37(3):424--438.
\bibitem{grettonmmd} Gretton A, Borgwardt KM, Rasch MJ, Sch\"olkopf B, Smola A (2012) A kernel two-sample test. \emph{J. Machine Learn. Res.} 13:723--773.
\bibitem{gretton} Gretton A, Fukumizu K, Teo CH, Song L, Sch\"olkopf B, Smola AJ (2008) A kernel statistical test of independence. \emph{Adv. Neural Inform. Processing Systems} 20.
\bibitem{gkx} Gu S, Kelly B, Xiu D (2020) Empirical asset pricing via machine learning. \emph{Rev. Financial Stud.} 33(5):2223--2273.
\bibitem{hs} Hanson SG, Stein JC (2015) Monetary policy and long-term real rates. \emph{J. Financial Econom.} 115(3):429--448.
\bibitem{hj} Hiemstra C, Jones JD (1994) Testing for linear and nonlinear Granger causality in the stock price-volume relation. \emph{J. Finance} 49(5):1639--1664.
\bibitem{holm} Holm S (1979) A simple sequentially rejective multiple test procedure. \emph{Scand. J. Statist.} 6(2):65--70.
\bibitem{hong} Hong Y (1999) Hypothesis testing in time series via the empirical characteristic function: A generalized spectral density approach. \emph{J. Amer. Statist. Assoc.} 94(448):1201--1220.
\bibitem{jk} Jaroci\'nski M, Karadi P (2020) Deconstructing monetary policy surprises---The role of information shocks. \emph{Amer. Econom. J. Macroeconom.} 12(2):1--43.
\bibitem{jorda} Jord\`a \`O (2005) Estimation and inference of impulse responses by local projections. \emph{Amer. Econom. Rev.} 95(1):161--182.
\bibitem{kanzig} K\"anzig DR (2021) The macroeconomic effects of oil supply news: Evidence from OPEC announcements. \emph{Amer. Econom. Rev.} 111(4):1092--1125.
\bibitem{lms} Lanne M, Meitz M, Saikkonen P (2017) Identification and estimation of non-Gaussian structural vector autoregressions. \emph{J. Econometrics} 196(2):288--304.
\bibitem{ns} Nakamura E, Steinsson J (2018) High-frequency identification of monetary non-neutrality: The information effect. \emph{Quart. J. Econom.} 133(3):1283--1330.
\bibitem{pelgerxiong} Pelger M, Xiong R (2022) State-varying factor models of large dimensions. \emph{J. Bus. Econom. Statist.} 40(3):1315--1333.
\bibitem{peters2013} Peters J, Janzing D, Sch\"olkopf B (2013) Causal inference on time series using restricted structural equation models. \emph{Adv. Neural Inform. Processing Systems} 26.
\bibitem{peters} Peters J, Mooij JM, Janzing D, Sch\"olkopf B (2014) Causal discovery with continuous additive noise models. \emph{J. Machine Learn. Res.} 15:2009--2053.
\bibitem{petersenci} Petersen L, Hansen NR (2021) Testing conditional independence via quantile regression based partial copulas. \emph{J. Machine Learn. Res.} 22(70):1--47.
\bibitem{pmw} Plagborg-M{\o}ller M, Wolf CK (2021) Local projections and VARs estimate the same impulse responses. \emph{the journal} 89(2):955--980.
\bibitem{rz} Ramey VA, Zubairy S (2018) Government spending multipliers in good times and in bad: Evidence from US historical data. \emph{J. Political Econom.} 126(2):850--901.
\bibitem{rrot} Ramsey JB, Rothman P (1996) Time irreversibility and business cycle asymmetry. \emph{J. Money Credit Banking} 28(1):1--21.
\bibitem{rr} Romer CD, Romer DH (2004) A new measure of monetary shocks: Derivation and implications. \emph{Amer. Econom. Rev.} 94(4):1055--1084.
\bibitem{SenSen} Sen A, Sen B (2014) Testing independence and goodness-of-fit in linear models. \emph{Biometrika} 101(4):927--942.
\bibitem{Shimizu} Shimizu S, Hoyer PO, Hyv\"arinen A, Kerminen A (2006) A linear non-Gaussian acyclic model for causal discovery. \emph{J. Machine Learn. Res.} 7:2003--2030.
\bibitem{sw} Stock JH, Watson MW (2018) Identification and estimation of dynamic causal effects in macroeconomics using external instruments. \emph{Econom. J.} 128(610):917--948.
\bibitem{tt} Tenreyro S, Thwaites G (2016) Pushing on a string: US monetary policy is less powerful in recessions. \emph{Amer. Econom. J. Macroeconom.} 8(4):43--74.
\bibitem{zhangci} Zhang K, Peters J, Janzing D, Sch\"olkopf B (2011) Kernel-based conditional independence test and application in causal discovery. \emph{Proc. 27th Conf. Uncertainty Artificial Intelligence}, 804--813.
"""
b0 = tex.index('\\begin{thebibliography}{99}') + len('\\begin{thebibliography}{99}')
b1 = tex.index('\\end{thebibliography}')
tex = tex[:b0] + "\n\\setlength{\\itemsep}{1pt}\\setlength{\\parskip}{0pt}" + BIB + tex[b1:]

# ---------------------------------------------------------------- INFORMS-style unnumbered references
rep("\\widowpenalty=10000\\clubpenalty=10000\n",
    "\\widowpenalty=10000\\clubpenalty=10000\n\\makeatletter\\renewcommand\\@biblabel[1]{}\\def\\@openbib@code{\\leftmargin=1.5em\\itemindent=-1.5em\\labelsep=0pt\\labelwidth=0pt}\\makeatother\n")

T.write_text(tex)
print('submission_revision: applied')
