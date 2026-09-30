"""Build the the journal version from the journal manuscript: reframed front matter, fewer main-text exhibits,
remaining exhibits moved to an online appendix. All numbers still come from the same macro files."""
import os, re, shutil, glob
S = os.path.dirname(__file__); SRC = os.path.join(S, "tex", "v"); DST = os.path.join(S, "tex", "v"); os.makedirs(DST, exist_ok=True)
for f in glob.glob(os.path.join(SRC, "*")):
    if f.lower().endswith((".tex", ".pdf")) and not os.path.basename(f).startswith(("main", "cover_letter")): shutil.copy2(f, DST)
s = open(os.path.join(SRC, "main.tex"), encoding="utf-8").read()
def sub(a, b, count=1):
    global s
    assert a in s, "MISSING: " + a[:80]; s = s.replace(a, b, count)

sub("\\documentclass[12pt]{article}", "\\documentclass[11pt]{article}")
# ---------- abstract ----------
s = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda m: r"""\begin{abstract}
\noindent When a customer fails to obtain a financial product or relief for which it is eligible, did it not ask, or did asking not suffice? The distinction matters for the design of service processes, because the first failure calls for information and outreach and the second is set by the provider's own requirements. I study it in a mortgage servicer's administrative records, in which a borrower's request for pandemic forbearance and the servicer's grant are separate, dated events and no grant occurs without a request. Through 6 April 2020 the servicer granted \cPre{} percent of requests on its conventional loans, within days. From 7 April it required a full documentation package and granted \cPost{} percent, with a median delay of four weeks. Requests on government-backed loans, which the CARES Act made a right on attestation, were unaffected. Borrowers who asked on either side of the date are alike, which yields a regression discontinuity in time with a placebo group. Those who just missed easy completion kept paying through the months a forbearance would have paused, remitting about \$\ledpiThree{} more per loan in the following quarter, were less delinquent and less often modified six months later, and were no worse off after a year. The paperwork cut take-up twelvefold, yet those who completed it were no needier on any measure taken before the request than those approved on request. A borrower's history of responsiveness predicts who asks and does not predict whose request succeeds. Process requirements are a powerful and blunt instrument: they govern take-up, and they screen, but not on need.
\end{abstract}""", s, count=1, flags=re.S)
sub("\\noindent\\textbf{Keywords:} household finance; take-up; ordeals; mortgage forbearance; loan servicing; debt relief.\\\\\n\\textbf{JEL:} G51, G21, D14, H31.",
    "\\noindent\\textbf{Keywords:} household finance; service process design; take-up; ordeals; mortgage servicing; debt relief.")
# ---------- opening paragraph: process framing ----------
sub("A household that goes without financial relief has failed at one of two things. It did not ask, or it asked and the request went nowhere. The two failures have different causes and different remedies. The first is about what the household knows, believes and gets around to; disclosure, reminders and advice act on it. The second is about what happens after the household acts: what paperwork is demanded, who decides, and how.",
    "A household that goes without financial relief has failed at one of two things. It did not ask, or it asked and the request went nowhere. The two failures have different causes and different remedies. The first is about what the household knows, believes and gets around to; disclosure, reminders and advice act on it. The second is about what happens after the household acts: what paperwork is demanded, who decides, and how. That second stage is a process the provider designs. A lender or servicer chooses what a customer must submit and how a request is reviewed, and can change those requirements from one day to the next. How much such requirements move take-up, and whom they turn away, are questions about the design of financial service operations as much as about household behavior.")
# ---------- conclusion: managerial implication ----------
sub("Two implications follow. For relief policy,", "Three implications follow. For the design of service processes, a documentation requirement is a lever of a different order from outreach or pricing: here it moved completion from four in five to one in fourteen in a day, at no visible gain in targeting, which suggests that providers who wish to ration relief toward need should look for instruments other than paperwork. For relief policy,")
# ---------- move exhibits to the online appendix ----------
def take(label):
    global s
    m = re.search(r"\\begin\{table\}\[t\]\\centering\\caption\{[^\n]*?\\label\{" + re.escape(label) + r"\}.*?\\end\{table\}\n?", s, flags=re.S)
    assert m, label; s = s[:m.start()] + s[m.end():]; return m.group(0).replace("\\begin{table}[t]", "\\begin{table}[h]")
moved = "".join(take(l) + "\n" for l in ["tab:mech", "tab:whoasks", "tab:funnel"])
sub("\\section{Additional tables and figures}", "\\section{Additional tables and figures}\n\n" + moved)
sub("\\appendix\n\\section{Proofs}", "\\clearpage\n\\appendix\n\\begin{center}\\Large\\textbf{Online Appendix}\\end{center}\n\\section{Proofs}")
# references must precede the online appendix: move bibliography before \appendix
bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}\n?", s, flags=re.S); assert bib
s = s.replace(bib.group(0), ""); s = s.replace("\\clearpage\n\\appendix", bib.group(0) + "\n\\clearpage\n\\appendix", 1)
# page counter marker between main text and appendix
s = s.replace("\\clearpage\n\\appendix", "\\label{endmain}\n\\clearpage\n\\appendix", 1)
open(os.path.join(DST, "main.tex"), "w", encoding="utf-8").write(s); print("v/main.tex written; exhibits moved:", moved.count("\\begin{table}"))
