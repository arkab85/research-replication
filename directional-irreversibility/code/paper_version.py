"""Build the journal version from the final the journal source (source/main.tex).

journal requirements (v.org/submissions, checked 19 Sep 2026): separate title page with
all identifying information; blind manuscript beginning with the title and a one-paragraph
abstract of at most 100 words; 8.5x11 paper, 1-inch margins, 12-point Times New Roman,
double-spaced main body and appendices; excessively long submissions are likely to be desk
rejected. The analysis is unchanged: this script only reformats, shortens the main text by
moving five subsections to the Internet Appendix, and rewrites the abstract.
Outputs in v/: DII_JFQA_Manuscript.pdf, DII_JFQA_Internet_Appendix.pdf, DII_JFQA_Title_Page.pdf.
"""
from pathlib import Path
import subprocess, sys, shutil
import pymupdf

S = Path(__file__).resolve().parent
P = S.parent
J = P / 'v'; J.mkdir(exist_ok=True)
tex = (S / 'main.tex').read_text()


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


def cut(start, end):
    global tex
    i = tex.index(start); j = tex.index(end, i)
    block = tex[i:j]; tex = tex[:i] + tex[j:]
    return block


# ---------------------------------------------------------------- format
rep("\\documentclass[11pt]{article}", "\\documentclass[12pt]{article}")
rep("\\onehalfspacing\n", "\\setstretch{2}\n")
rep("\\setlength{\\parskip}{3pt}", "\\setlength{\\parskip}{0pt}")

# ---------------------------------------------------------------- abstract (<= 100 words)
a0 = tex.index('\\begin{abstract}') + len('\\begin{abstract}\n'); a1 = tex.index('\\end{abstract}')
ABS = ("Identified-shock studies in finance measure mean and volatility responses, which do not reveal whether the joint "
       "distribution favors a shock-to-outcome representation over its reverse. I develop the Directional Irreversibility "
       "Index (DII) and show when financial data can reveal it: predictable drivers bound the signal, omitted exposure states "
       "can erase it, and impact-inclusive price changes can show persistent footprints under a martingale price. Calibrating "
       "DII as a degenerate statistic rejects true nulls 41--47 percent of the time; the proposed joint bootstrap restores size while "
       "propagating fitted-model uncertainty. Removing common factors before pooling a panel raises power sharply.\n")
assert len(ABS.split()) <= 100, len(ABS.split())
tex = tex[:a0] + ABS + tex[a1:]

# ---------------------------------------------------------------- move five subsections to the Internet Appendix
moved = []
moved.append(cut("\\subsection{Component bounds and the cost of model-based decisions}\\label{sec:replay}",
                 "\\subsection{Persistent DII need not mean delayed incorporation}"))
rep("\\subsection{Persistent DII need not mean delayed incorporation}",
    "The same component bounds also limit the decision error from reusing fitted residuals in scenario analysis; Internet Appendix~\\ref{sec:replay} gives the bound and an example.\n\n\\subsection{Persistent DII need not mean delayed incorporation}")
moved.append(cut("\\subsection{State observability: a paired estimation experiment}\\label{sec:stateexperiment}",
                 "\\subsection{The propositions at work: toy examples with known answers}"))
moved.append(cut("\\subsection{The propositions at work: toy examples with known answers}\\label{sec:toys}",
                 "\\subsection{Balanced panels, factor conditioning, and heavy tails}"))
moved.append(cut("The bandwidth is the natural next question for a kernel method.", "Heavy tails are the rule in financial data"))
moved.append(cut("Heavy tails are the rule in financial data", "\\section{A reproducible financial application}"))
moved[3] = "\\subsection{Bandwidth multiples}\n" + moved[3]
moved[4] = "\\subsection{Heavy-tailed innovations}\n" + moved[4]
rep("\\section{A reproducible financial application}",
    "Internet Appendix~\\ref{sec:stateexperiment} reports a paired state-observability experiment, and Internet Appendix~\\ref{sec:toys} "
    "illustrates each proposition in a toy design with known answers. Internet Appendices~\\ref{sec:bandwidthIA} and~\\ref{sec:heavyIA} show "
    "size control at bandwidth multiples of 0.5, 1, and 2 and under Student-$t_3$ innovations.\n\n"
    "\\section{A reproducible financial application}")
moved[3] = moved[3].replace("\\subsection{Bandwidth multiples}\n", "\\subsection{Bandwidth multiples}\\label{sec:bandwidthIA}\n")
moved[4] = moved[4].replace("\\subsection{Heavy-tailed innovations}\n", "\\subsection{Heavy-tailed innovations}\\label{sec:heavyIA}\n")
rep("\\subsection{Balanced panels, factor conditioning, and heavy tails}", "\\subsection{Balanced panels and factor adjustment}")

# ---------------------------------------------------------------- appendix header and wording
rep("\\renewcommand{\\thepage}{EC-\\arabic{page}}", "\\renewcommand{\\thepage}{IA-\\arabic{page}}")
rep("\\begin{center}\\Large Electronic Companion\\end{center}", "\\begin{center}\\Large Internet Appendix\\end{center}")
for a, b in [("electronic companion", "Internet Appendix"), ("Electronic companion", "Internet Appendix"),
             ("the companion", "the Internet Appendix"), ("The companion", "The Internet Appendix"),
             ("in the companion", "in the Internet Appendix")]:
    tex = tex.replace(a, b)
ai = tex.index("\\begin{center}\\Large Internet Appendix\\end{center}") + len("\\begin{center}\\Large Internet Appendix\\end{center}")
tex = tex[:ai] + "\n\\section{Supplementary analyses moved from the main text}\\label{app:moved}\n" + "\n".join(b.strip() + "\n" for b in moved) + tex[ai:]

# ---------------------------------------------------------------- practical guide (introduction)
GUIDE = ("\\paragraph{Using DII in an applied study.} The results translate into a short protocol. "
         "(i) Measure the driver as a surprise rather than a level, because predictable variation cannot carry directional signal (Proposition~\\ref{prop:innovation}). "
         "(ii) Define exposure states before estimation (Proposition~\\ref{prop:stateerasure}), and remove observed common factors from outcomes rather than conditioning on them inside the kernel (Section~\\ref{sec:panel}). "
         "(iii) Separate impact-inclusive from post-impact outcome windows before interpreting persistence (Proposition~\\ref{prop:incorporation}). "
         "(iv) Fix the horizons, bandwidth multiple, and training and evaluation split in advance, and report every primary contrast with the joint p-values of Theorem~\\ref{thm:main}. "
         "(v) Report both components with simultaneous bounds: when both exclude zero, both fitted representations have dependent residuals, the data do not order the directions, and the fitted residuals should not be reused as independent disturbances; the contrast itself need not be near zero (Section~\\ref{sec:oilvix}). "
         "The replication package implements each step.\n\n")
rep("\\subsection{Relationship to existing work}", GUIDE + "\\subsection{Relationship to existing work}")

# ---------------------------------------------------------------- summary statistics (main text) and variable definitions (Internet Appendix)
import json as _json
ss = _json.load(open(P / 'summary_statistics.json'))
def row(label, st):
    return f"{label} & {st['n']} & {st['mean']:.3f} & {st['sd']:.3f} & {st['min']:.3f} & {st['median']:.3f} & {st['max']:.3f}\\\\\n"
lab = {'MP': 'MP shock', 'FX': 'Yen per dollar, log change', 'Treasury': '10-year Treasury yield change', 'Credit': 'Baa--Aaa spread change', 'VIX': 'VIX, log change'}
pend = ss['monetary_outcomes_pending']
rowsA = ''
for per in ['training', 'evaluation']:
    rowsA += f"\\multicolumn{{7}}{{l}}{{\\emph{{{'Training, 1991--2007' if per == 'training' else 'Evaluation, 2009--2018'}}}}}\\\\\n"
    for k in ['MP', 'FX', 'Treasury', 'Credit', 'VIX']:
        if k in ss['monetary']:
            rowsA += row(lab[k], ss['monetary'][k][per])
        else:
            rowsA += f"{lab[k]} & \\multicolumn{{6}}{{c}}{{\\textbf{{[pending: run summary\\_statistics.py]}}}}\\\\\n"
rowsB = ''
for per, tag in [('training', 'Training window'), ('evaluation', 'Evaluation window')]:
    rowsB += f"\\multicolumn{{7}}{{l}}{{\\emph{{{tag}}}}}\\\\\n"
    rowsB += row('WTI spot, log change', ss['oilvix']['Oil'][per]) + row('VIX, log change', ss['oilvix']['VIXd'][per])
SUMTAB = ("\\begin{table}[htbp]\\centering\\small\\setstretch{1}\n\\caption{Summary statistics for the two applications}\\label{tab:summary}\n"
          "\\begin{tabular}{lrrrrrr}\\toprule\nVariable & $N$ & Mean & SD & Min & Median & Max\\\\\\midrule\n"
          "\\multicolumn{7}{l}{\\textbf{A. Monetary-policy surprises (monthly)}}\\\\\n" + rowsA +
          "\\midrule\\multicolumn{7}{l}{\\textbf{B. Oil and VIX (daily)}}\\\\\n" + rowsB +
          "\\bottomrule\\end{tabular}\n\\par\\vspace{4pt}\\raggedright\\noindent Log changes are multiplied by 100; yield and spread changes are in percentage points; the shock is in the authors' units. "
          "Panel B windows run from the first to the last date of the training and evaluation origins. Definitions and sources are in Internet Appendix Table~\\ref{tab:vardefs}.\n\\end{table}\n\n")
rep("\\subsection{Directional evidence and calibration sensitivity}", "Table~\\ref{tab:summary} reports summary statistics for both applications.\n\n" + SUMTAB + "\\subsection{Directional evidence and calibration sensitivity}")
VARDEF = r"""
\section{Variable definitions and sources}\label{app:vardefs}
\begin{table}[htbp]\centering\small\setstretch{1}
\caption{Variable definitions}\label{tab:vardefs}
\begin{tabular}{p{3.2cm}p{8.2cm}p{3.6cm}}\toprule
Variable & Definition & Source\\\midrule
MP shock & Monthly monetary-policy surprise, median-rotation MP series, used as the driver $X_t$. & Jaroci\'nski and Karadi (2020), authors' update (CC BY 4.0)\\
Yen per dollar & $100\times$ change in the log of the last daily yen-per-dollar rate of the month, at $t+h$. & FRED DEXJPUS\\
10-year Treasury yield & Change in the monthly 10-year constant-maturity yield, percentage points, at $t+h$. & FRED GS10\\
Baa--Aaa spread & Change in the monthly Moody's Baa minus Aaa yield, percentage points, at $t+h$. & FRED BAA, AAA\\
VIX (monthly) & $100\times$ change in the log of the last daily VIX close of the month, at $t+h$. & FRED VIXCLS (CBOE)\\
History (monetary) & Previous month's shock and previous month's outcome change. & As above\\
WTI spot & $100\times$ log change in the WTI Cushing spot price between consecutive common trading days, used as the driver $X_t$. & U.S. EIA RWTC (public domain)\\
VIX (daily) & $100\times$ log change in the CBOE VIX close between consecutive common trading days, at $t+h$. & CBOE via \texttt{datasets/finance-vix} (public domain)\\
History (oil--VIX) & Previous day's oil return and previous day's VIX change. & As above\\
\bottomrule\end{tabular}
\end{table}
"""
rep("\\end{document}", VARDEF.strip() + "\n\\end{document}")
if pend:
    print('WARNING: monetary outcome summary statistics are pending. Run finance_download.py and summary_statistics.py (needs FRED access), then rebuild.')

# ---------------------------------------------------------------- external-review pass (journal only)
rep("\\begin{proposition}[Ordering within the maintained class]\n", "\\begin{proposition}[Ordering within the maintained class]\\label{prop:ordering}\n")
HOOK = ("Consider a researcher who relates monetary-policy surprises to later changes in credit spreads and asks whether the spread "
        "change is a response to the surprise plus a disturbance the surprise does not affect. Two errors are easy to make. A natural "
        "test compares residual dependence in the two regressions and calibrates the difference as a degenerate statistic; when neither "
        "representation has independent residuals, the usual case in financial data, that calibration rejects a true null in 41--47 "
        "percent of samples at a nominal 5 percent level (Figure~\\ref{fig:calibration}). And when exposure to the surprise changes with "
        "an unobserved state, pooled data can show no dependence at all even though every state transmits the shock "
        "(Proposition~\\ref{prop:stateerasure}). This paper supplies the object, the design rules, and the inference that avoid both errors.\n\n")
rep("\\section{Introduction}\n", "\\section{Introduction}\n" + HOOK)

NOVEL = r"""Table~\ref{tab:novelty} summarizes how DII and its joint inference differ from the tools most often used for related questions.

\begin{table}[htbp]\centering\small\setstretch{1}
\caption{What DII adds relative to existing tools}\label{tab:novelty}
\begin{tabular}{>{\raggedright\arraybackslash}p{.24\linewidth}>{\raggedright\arraybackslash}p{.30\linewidth}>{\raggedright\arraybackslash}p{.38\linewidth}}\toprule
Existing tool & What it tests & What DII and joint inference add\\\midrule
Local projections, VARs & Mean response of the outcome to the shock & Whether the disturbance is invariant to the shock and history, compared with the reverse representation\\[3pt]
Volatility and quantile responses & Conditional variance or distribution of the outcome & A contrast between two residual-independence departures, not a feature of one conditional law\\[3pt]
Granger and nonlinear Granger tests; connectedness & Incremental predictability; variance shares & DII can be zero under strong predictability and positive without linear predictability\\[3pt]
Time-reversibility tests & Law of a process versus its time reversal & Two regression representations of a specified shock, horizon, and history\\[3pt]
Additive-noise causal discovery, including time-series structural models & Orientation under causal sufficiency, typically with oracle or unpropagated first-stage fits & Joint inference for fitted directional contrasts and finite horizon profiles, propagating training uncertainty and valid under both null configurations\\[3pt]
Non-Gaussian SVARs & Identification of a structural impact matrix & No structural system; one external shock with possibly nonlinear responses\\
\bottomrule\end{tabular}
\end{table}

"""
rep("\\subsection{Relationship to existing work}\n", "\\subsection{Relationship to existing work}\n" + NOVEL)

SCOPE = ("\\paragraph{Scope of the conclusions.} DII measures an asymmetry between fitted residual representations. It orders "
         "directions only within the maintained model class (Proposition~\\ref{prop:ordering}), a nonrejection is not evidence that "
         "transmission is absent, and Theorem~\\ref{thm:main} gives pointwise rather than uniform validity. The applications demonstrate "
         "implementation and reporting; they are not offered as new financial findings. The remaining sections state results without "
         "repeating these qualifications.\n\n")
rep("\\paragraph{Using DII in an applied study.}", SCOPE + "\\paragraph{Using DII in an applied study.}")
for sent in [" Its role is illustrative.", " Similarly, a nonsignificant profile is not evidence that transmission is absent.",
             " Without the population restrictions, Panel B does not establish that the seven wild rejections are false positives."]:
    assert tex.count(sent) == 1, sent
    tex = tex.replace(sent, "")

(J / 'main_v.tex').write_text(tex)

# ---------------------------------------------------------------- compile and split
B = J / 'build'; B.mkdir(exist_ok=True)
for fig in ['dii_calibration.pdf', 'dii_theory_demos.pdf']:
    pass  # figures are referenced as ../name.pdf relative to source/, see below
for i in range(3):
    subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(B), str(J / 'main_v.tex')],
                   cwd=S, check=True, stdout=subprocess.DEVNULL)
log = (B / 'main_v.log').read_text(errors='ignore')
assert 'undefined' not in log.lower(), 'undefined references'
full = pymupdf.open(B / 'main_v.pdf')
hits = [i for i, p in enumerate(full) if 'Internet Appendix' in p.get_text() and 'Supplementary analyses moved' in p.get_text()]
assert len(hits) == 1, hits
for name, s, e in [('Manuscript', 0, hits[0] - 1), ('Internet_Appendix', hits[0], len(full) - 1)]:
    d = pymupdf.open(); d.insert_pdf(full, from_page=s, to_page=e)
    d.set_metadata({'title': 'Directional Irreversibility in Economic Dynamics' + (' - Internet Appendix' if s else ''), 'author': ''})
    d.save(J / f'DII_JFQA_{name}.pdf'); print(name, len(d))

# ---------------------------------------------------------------- title page (identifying information only here)
TP = r"""\documentclass[12pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}\usepackage{mathptmx,setspace}\setstretch{2}
\begin{document}\thispagestyle{empty}
\begin{center}{\large\bfseries Directional Irreversibility in Economic Dynamics:\\ Inference and Shock Transmission}\\[18pt]
Arka Prava Bandyopadhyay\\
Adjunct Professor, Department of Economics, Columbia University\\
New Hyde Park, NY, USA\\
ab3985@columbia.edu \quad (347) 260-0869\end{center}
\vspace{12pt}\noindent\textbf{Funding.} This research received no specific funding.\\
\noindent\textbf{Use of AI tools.} The author used Anthropic's Claude, an AI assistant, for editing and restructuring the manuscript, positioning against the literature, designing and coding simulation, verification, and replication programs, drafting extensions (including the estimated-loading proposition and its implementation), and preparing submission materials. The author reviewed and verified all text, code, and results and takes full responsibility for the paper.\\
\noindent\textbf{Data and code.} All data are public; code will be shared under the journal code sharing policy.
\end{document}
"""
(J / 'title_page.tex').write_text(TP)
subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(B), str(J / 'title_page.tex')],
               cwd=S, check=True, stdout=subprocess.DEVNULL)
shutil.copy(B / 'title_page.pdf', J / 'DII_JFQA_Title_Page.pdf')
print('journal version built')
