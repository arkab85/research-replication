from pathlib import Path
import json
root=Path(__file__).resolve().parent.parent
r=json.loads((root/'code/verified_results.json').read_text());e=r['estimates'];v=r['validation'];c=v['contrasts'];m=v['model_metrics']
# All tables are generated from newly reproduced JSON.
t=r'''\begin{table}[tbp]\centering\small
\caption{Signed-gap coefficients by AVM confidence}\label{tab:baseline}
\begin{tabular}{lrrr}\toprule
Specification & Premium & Discount & $N$\\\midrule
'''
for k,label in [('premium_discount','Plausible AVM'),('low_confidence','Confidence below 80'),('high_confidence','Confidence at least 80')]:
 z=e[k];t+=f"{label} & {z['coef']['prem']:.4f} & {z['coef']['disc']:.4f} & {z['n']:,}"+r'\\'+'\n'+f" & ({z['se']['prem']:.4f}) & ({z['se']['disc']:.4f}) &"+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.95\textwidth}{\footnotesize Linear probability models. Outcome: recorded pending or contingent status. Plausible-AVM sample ($0.5\leq A/P\leq2$). City, property-type and listing-month effects; log asking price, log living area, log lot size, age, missingness indicators, and confidence/10. City-clustered standard errors in parentheses. The first row is the uncorrected benchmark also reported in the companion methods paper described in Section~\ref{sec:companion}.}
\end{table}
\begin{table}[tbp]\centering\small
\caption{Gap-by-confidence interactions and sensitivity checks}\label{tab:robust}
\begin{tabular}{lrrrrr}\toprule
Specification & Premium & Premium $\times$ high & Discount & Discount $\times$ high & $N$\\\midrule
'''
for k,label in [('interactions','Baseline'),('physical_atypicality','Physical atypicality'),('county_cohort','County--cohort effects'),('update_conditioned','Update-recency effects'),('exclude_sold_dates','Exclude sold dates')]:
 z=e[k];t+=f"{label} & {z['coef']['prem']:.4f} & {z['coef']['prem:hiconf']:.4f} & {z['coef']['disc']:.4f} & {z['coef']['disc:hiconf']:.4f} & {z['n']:,}"+r'\\'+'\n'+f" & ({z['se']['prem']:.4f}) & ({z['se']['prem:hiconf']:.4f}) & ({z['se']['disc']:.4f}) & ({z['se']['disc:hiconf']:.4f}) &"+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.95\textwidth}{\footnotesize Standard errors cluster by city except the county--cohort specification, which clusters by county. ``High'' denotes confidence at least 80. All models include main gap and confidence-indicator terms. Expanded checks add the physical controls described in the text. Update-recency bins are a stress test that may over-control because status changes can trigger record updates. The baseline row coincides with the uncorrected interaction benchmark in the companion methods paper (Section~\ref{sec:companion}); the other rows are specific to this paper.}
\end{table}
\begin{table}[tbp]\centering\small
\caption{County-held-out classification performance}\label{tab:validation}
\begin{tabular}{lrrrr}\toprule
Information set & Log loss & Brier score & ROC AUC & Avg. precision\\\midrule
'''
for k,label in [('property','Property / market'),('gap','Plus gap'),('metadata','Plus uncertainty'),('full','Plus both'),('gap_recency','Gap and recency'),('full_recency','Both and recency')]:
 z=m[k];t+=f"{label} & {z['logloss']:.5f} & {z['brier']:.5f} & {z['auc']:.4f} & {z['average_precision']:.4f}"+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.95\textwidth}{\footnotesize Identical observations and five county-held-out folds. Lower loss and Brier scores are better; higher ROC AUC and average precision are better. ``Both'' includes the gap, confidence, and log range width; ``recency'' adds report age and AVM age. 50,395 records in 1,614 counties; positive share 4.9 percent. No model predicts a verified future sale event.}
\end{table}
\begin{table}[tbp]\centering\small
\caption{Incremental log-loss comparisons}\label{tab:gains}
\begin{tabular}{lrr}\toprule
Comparison & Improvement (\%) & County-bootstrap 95\% interval\\\midrule
'''
for k,label in [('gap_to_full','Gap to full'),('metadata_to_full','Metadata to full'),('property_to_full','Property / market to full'),('gap_recency_to_full_recency','Gap to full, recency included')]:
 z=c[k];t+=f"{label} & {z['logloss_reduction_pct']:.2f} & [{z['ci95'][0]:.2f}, {z['ci95'][1]:.2f}]"+r'\\'+'\n'
t+=r'''\bottomrule\end{tabular}
\par\smallskip\parbox{.95\textwidth}{\footnotesize Relative improvement is $100(1-L_{augmented}/L_{benchmark})$. Intervals use 2,000 county resamples and condition on fitted predictions. All specified comparisons are reported.}
\end{table}
''';(root/'verified_tables.tex').write_text(t)
print('Rendered journal tables from verified_results.json')
