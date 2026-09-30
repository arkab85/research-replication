"""the journal submission stage, round 4: prespecified oil-VIX application.

Reads oilvix/results.json produced once by oilvix_application.py under the
hashed protocol in oilvix/protocol.md. Reports every primary result. No
identified shock or journal input is used.
"""
from pathlib import Path
import json

S = Path(__file__).resolve().parent
P = S.parent
T = S / 'main.tex'
tex = T.read_text()
r = json.load(open(P / 'oilvix/results.json'))
c = r['comparisons']; smp = r['sample']
proto_hash = (P / 'oilvix/protocol_sha256.txt').read_text().split()[0]


def rep(old, new, count=1):
    global tex
    n = tex.count(old)
    assert n == count, (n, old[:100])
    tex = tex.replace(old, new)


h0 = c['h0']
pmin = min(v['p_intersection'] for v in c.values())
def pos(v): return v > 5e-7
later = []
for k in ['h1', 'h5']:
    v = c[k]
    f_, b_ = pos(v['forward_lower95']), pos(v['reverse_lower95'])
    if f_ or b_:
        later.append(f"At $h={k[1:]}$ the {'forward' if f_ else 'reverse'} band only barely excludes zero (lower bound {1000*(v['forward_lower95'] if f_ else v['reverse_lower95']):.3f}) and the other band includes it.")
    else:
        later.append(f"At $h={k[1:]}$ both bands include zero.")
later = ' '.join(later)
rows = ''.join(
    f"{k[1:]} & {1000*v['forward_hsic']:.3f} & {1000*v['reverse_hsic']:.3f} & {1000*v['dii']:.3f} & {v['p_intersection']:.3f} & {v['p_holm']:.3f} & "
    f"[{1000*v['forward_lower95']:.3f}, {1000*v['forward_upper95']:.3f}] & [{1000*v['reverse_lower95']:.3f}, {1000*v['reverse_upper95']:.3f}]\\\\\n"
    for k, v in c.items())

# ---------------------------------------------------------------- abstract
old = "In a prespecified four-market monetary-shock illustration, seven naive rejections do not survive joint inference."
new = ("In prespecified applications, seven naive monetary-shock rejections do not survive joint inference, "
       "and oil and VIX data reject independent disturbances in both directions at impact.")
rep(old, new)
rep("in about 70 percent at 120 evaluation observations.", "in about 70 percent.")
rep("at a nominal 5 percent level.", "at nominal 5 percent.")
rep("in 41--47 percent of simulated samples at nominal", "in 41--47 percent of samples at nominal")
rep("identified shocks mainly through", "identified shocks through")
rep("detects strong quadratic exposures", "detects quadratic exposures")
rep("driver's remaining innovation variance", "driver's innovation variance")
a = tex[tex.index('\\begin{abstract}') + 16:tex.index('\\end{abstract}')]
assert len(a.split()) <= 250, len(a.split())

# ---------------------------------------------------------------- introduction
rep("Supplementary prediction comparisons show no supported gain.",
    "Supplementary prediction comparisons show no supported gain. A second prespecified application with daily oil and VIX data "
    "(Section~\\ref{sec:oilvix}) illustrates the component bounds: at impact, both representations reject residual independence, "
    "so neither direction is supported and a near-zero DII is the expected reading.")

# ---------------------------------------------------------------- Section 5 subsection
SUB = rf"""
\subsection{{A second prespecified application: oil-price moves and equity-implied volatility}}\label{{sec:oilvix}}
Large oil-price moves of either sign are associated with changes in equity-market uncertainty. A convex, sign-symmetric exposure is the case in which Proposition~\ref{{prop:separation}} exhibits positive DII with an independent forward disturbance. The design is descriptive: the daily oil return is not an identified shock and no causal claim is made. The protocol was written and hashed before any data values were downloaded (Appendix~\ref{{app:oilvix}}).

The driver is the daily log change in the WTI spot price (U.S. EIA) and the outcome is the daily log change in the VIX (CBOE), both public-domain series, on {smp['common_days']:,} common trading days from 1990 to 2019. The primary horizons are $h=0$, the same-day change, and the strictly post-impact changes at $h=1$ and $h=5$. The history contains the previous day's oil return and VIX change. Training uses the first 2,048 origins ({smp['first'][:4]}--{smp['training_end'][:4]}) and evaluation the last 2,048 ({smp['evaluation_start'][:4]}--{smp['evaluation_end'][:4]}). Both directions use total-degree-two means and the procedure of Theorem~\ref{{thm:main}} with 999 draws and common paths across horizons.

\begin{{table}}[htbp]\centering\small
\caption{{Oil returns and VIX changes: complete primary results}}\label{{tab:oilvix}}
\begin{{tabular}}{{rrrrrrcc}}\toprule
$h$ & $1000\widehat H_f$ & $1000\widehat H_b$ & $1000\widehat D_h$ & Joint $p$ & Holm $p$ & Forward 95\% band & Reverse 95\% band\\\midrule
{rows}\bottomrule\end{{tabular}}
\par\vspace{{4pt}}\raggedright\noindent Bands for the components (in units of $10^{{-3}}$) hold simultaneously across all six directions on one 95 percent confidence event. No original-DII sign certificate is claimed, because no external mean-approximation bound is supplied.
\end{{table}}

No DII contrast rejects; the smallest joint p-value is {pmin:.3f}, at impact. The component bands are informative, however. At $h=0$ both lower bounds are positive ({1000*h0['forward_lower95']:.3f} and {1000*h0['reverse_lower95']:.3f}): the data reject residual independence in the oil-to-VIX representation and in its reverse. Neither representation is admissible within the maintained class, so the contrast has no ordering interpretation, and a DII near zero is what one would expect under common news that moves both markets on the same day. {later} This is the reporting discipline of Section~\ref{{sec:boundsuse}}: the components say why a contrast is uninformative, rather than leaving a nonrejection to be read as absence of transmission.
"""
anchor = "\\section{Financial transmission and the use of DII}\\label{sec:finance}"
rep(anchor, SUB.strip() + "\n\n" + anchor)

# ---------------------------------------------------------------- companion appendix
EC = rf"""
\section{{Protocol and reproduction for the oil and VIX application}}\label{{app:oilvix}}
The protocol file \texttt{{oilvix/protocol.md}} was written before any data values were downloaded; its SHA-256 hash, {proto_hash[:16]}\ldots, was recorded before estimation, and the estimation script was run once. This is an internal prespecification, not an external registry. The protocol fixes the data sources, the 1990--2019 sample (ending before the 2020 negative WTI price), the driver, the outcome definitions, the horizons $\{{0,1,5\}}$, the history, the training and evaluation origins, the polynomial basis, fixed bandwidths, the bootstrap (999 draws, seed 20260920), and Holm adjustment over the three primary contrasts. It commits to reporting every primary result and to substituting no alternative sample, horizon, variable, basis, bandwidth, or block length.

Data: WTI Cushing spot (EIA series RWTC) and the CBOE VIX close, obtained from the \texttt{{datasets/oil-prices}} and \texttt{{datasets/finance-vix}} repositories under the ODC-PDDL public-domain dedication; the file hashes are recorded in \texttt{{oilvix/data\_sha256.txt}}. Changes are computed between consecutive common dates. Training origins run from {smp['first']} to {smp['training_end']} and evaluation origins from {smp['evaluation_start']} to {smp['evaluation_end']}; intervening origins are unused. Run \texttt{{python source/oilvix\_application.py}} to reproduce Table~\ref{{tab:oilvix}}. The joint 95 percent operator radius is {r['joint_radius95']:.4f}.

Because the evaluation period is 2011--2019 and the training period 1990--1999, the fitted units come from a more volatile oil market than the evaluation period. This affects power and the interpretation of kernel magnitudes, not the validity statements of Theorem~\ref{{thm:main}}, which condition on stationarity; the stationarity assumption across this split is substantive and unverified.
"""
rep("\\end{document}", EC.strip() + "\n\\end{document}")

T.write_text(tex)

# cover letter / memo additions (files written by revision_r2.py)
cl = (S / 'cover_letter.md').read_text()
a_clean = a.strip().replace('--', '–')
start = cl.index('Abstract:\n') + len('Abstract:\n'); end = cl.index('\n\nSuggested Associate Editors')
cl = cl[:start] + a_clean + cl[end:]
cl = cl.replace("Its prespecified four-market monetary-shock illustration does not reject:",
                "A second prespecified application with daily oil and VIX data shows the component bounds at work: at impact, both representations reject residual independence, so the index correctly reports no directional ordering. The prespecified four-market monetary-shock illustration does not reject:")
(S / 'cover_letter.md').write_text(cl)
print('revision_r4: applied')
