from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
s=(R/'source_material/empirical_main_seed.md').read_text()
emp=s[s.index('# 4 Data and measurement'):s.index('# 7 Investment and conversion implications')]
emp=re.sub(r'(Figure |Fig\. )([12])',lambda m:m[1]+str(int(m[2])+2),emp)
emp=emp.replace('figures/Fig2.png','figures/Fig4.png').replace('figures/Fig1.png','figures/Fig3.png')
emp=re.sub(r'^(#{1,2} )([456])(?=[ .])',lambda m:m[1]+str(int(m[2])+1),emp,flags=re.M)
emp=emp.replace('\\tag{6}', '\\tag{E1}').replace('\\tag{7}', '\\tag{E2}')
emp=emp.replace('Equation (3) explains why this distinction matters economically.','Property cash-flow accounting explains why this distinction matters economically.')
theory=(R/'theory_section.md').read_text()
figtext='\n\n![](figures/Fig1.png){width=6.5in}\n\n**Fig. 1 Model response to a spending-origin transfer**\n\nNotes: The figure plots Equation (T6) at symmetric equilibria for constructed variety parameters. The horizontal axis is the cross-neighborhood shopping penalty a on a logarithmic scale. The curve is a comparative static, not an empirical estimate or a New York calibration.\n\n'
theory=theory.replace('## 3.5 Land-use composition and office conversion',figtext+'## 3.5 Land-use composition and office conversion')
refs=s.split('# References\n\n')[1].strip().split('\n\n')
refs += ['Couture, V., & Handbury, J. (2020). Urban revival in America. *Journal of Urban Economics, 119*, 103267. <https://doi.org/10.1016/j.v.2020.103267>', 'Delventhal, M. J., Kwon, E., & Parkhomenko, A. (2022). journal Insight: How do cities change when we work from home? *Journal of Urban Economics, 127*, 103331. <https://doi.org/10.1016/j.v.2021.103331>']
refs += ['Koster, H. R. A., Pasidis, I., & van Ommeren, J. (2019). Shopping externalities and retail concentration: Evidence from Dutch shopping streets. *Journal of Urban Economics, 114*, 103194. <https://doi.org/10.1016/j.v.2019.103194>', 'Behrens, K., & Murata, Y. (2021). On quantitative spatial economic models. *Journal of Urban Economics, 123*, 103348. <https://doi.org/10.1016/j.v.2021.103348>']
refs += ['Data USA. (2026a). *The Data USA API*. Retrieved September 14, 2026, from <https://datausa.io/about/api>', 'Data USA. (2026b). *Metropolitan profiles: 2024 ACS employment and commuting measures*. Retrieved September 14, 2026, from <https://datausa.io/profile/geo/los-angeles-long-beach-anaheim-ca>; <https://datausa.io/profile/geo/seattle-tacoma-bellevue-wa>; <https://datausa.io/profile/geo/dallas-fort-worth-arlington-tx>; <https://datausa.io/profile/geo/atlanta-sandy-springs-roswell-ga>; <https://datausa.io/profile/geo/philadelphia-camden-wilmington-pa-nj-de-md>; <https://datausa.io/profile/geo/san-francisco-oakland-hayward-ca>; <https://datausa.io/profile/geo/houston-the-woodlands-sugar-land-tx>; <https://datausa.io/profile/geo/portland-vancouver-hillsboro-or-wa>', 'U.S. Census Bureau. (2026). *American Community Survey 1-year data (2005–2024)*. <https://www.census.gov/data/developers/data-sets/acs-1year.html>']
refs += ['U.S. Census Bureau. (2025). *County Business Patterns: 2023 metropolitan area data*. <https://www.census.gov/programs-surveys/cbp.html>']
main=(R/'v_front.md').read_text()+'\n\n'+theory+'\n\n'+(R/'general_theory.md').read_text()+'\n\n'+emp+'\n\n'+(R/'v_end.md').read_text()+'\n\n# References\n\n'+'\n\n'.join(sorted(refs))+'\n'
main=main.replace('Online Resource 1','the supplement').replace('Online Resource 2','the replication archive').replace('. the supplement','. The supplement')
blocks=(R/'multicity_public_empirical_blocks.md').read_text()
main=main.replace('# 6 Empirical design',blocks.split('## 6.4')[0]+'\n\n# 6 Empirical design')
main=main.replace('# 7 Results', '## 6.4'+blocks.split('## 6.4',1)[1].split('## 7.5',1)[0]+'# 7 Results')
main=main.replace('# 8 Urban redevelopment and the geographic scope of recovery','## 7.5'+blocks.split('## 7.5',1)[1]+'\n\n# 8 Urban redevelopment and the geographic scope of recovery')
(R/'manuscript.md').write_text(main)
sup=(R/'source_material/empirical_supplement_seed.md').read_text()
sup=sup.replace('Online Resource 1 for Live Work and Play in Commercial Real Estate','Supplement to Changing Spending Origins and Urban Retail Adjustment')
sup=sup.replace('Theory data construction and complete results','Empirical construction complete results and theory proofs')
sup=sup.replace('# A Conversion and neighborhood property values','# A Auxiliary conversion accounting')
sup=sup.replace('The main paper distinguishes spending exposure from use evenness. This appendix gives the conversion comparison in more detail.','This appendix preserves an auxiliary local spending-accounting model. Unlike the closed-city model in the main paper, it does not impose offsetting spending changes elsewhere. It therefore describes a local property-value externality, not an aggregate urban welfare result.')
sup=sup.replace("Substituting $R+\\ell x$ and $J-w_cx$ into the main paper's Equation (1) gives",r'''Let $R$ be resident households, $J$ office-associated jobs, $c(A)$ local spending retention, $e_R,e_J$ spending weights, and $V$ visitor spending. The shock $z$ raises retained resident spending at rate $h$ and reduces workplace and visitor spending at rates $q$ and $v$. The auxiliary baseline is

$$D(z)=c(A)\{e_RR(1+hz)+e_JJ(1-qz)\}+V(1-vz). \tag{S0}$$

Substituting $R+\ell x$ and $J-w_cx$ into this baseline gives''')
sup=sup.replace('the sign changes at the threshold in the main paper.','the sign changes at $z^*=(e_Jw_c-e_R\\ell)/(e_R\\ell h+e_Jw_cq)$.')
sup=sup.replace('Online Resource 2 supplies','The replication archive supplies')
sup += '\n\n'+(R/'theory_proofs.md').read_text()+'\n\n'+(R/'general_proofs.md').read_text()
(R/'supplement.md').write_text(sup)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
a=np.geomspace(1,100,400)
fig,ax=plt.subplots(figsize=(6.8,3.1))
for th,col in zip([.25,.5,.75,.9],['#999999','#547b95','#193b58','#9b5326']):
 G=(a*a-1)/((a+1)**2-4*th*a)
 ax.plot(a,G,label=rf'$\theta={th}$',color=col,lw=1.7)
ax.axhline(1,color='black',lw=.7,ls='--');ax.set_xscale('log');ax.set(xlabel='Shopping penalty a',ylabel='Normalized reallocation response G',xlim=(1,100),ylim=(0,1.85));ax.legend(frameon=False,ncol=2,loc='lower right');ax.grid(axis='y',alpha=.18);fig.tight_layout()
for ext in ['png','pdf','eps']:fig.savefig(R/f'figures/Fig1.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
print('Main words',len(main.split()),'Abstract words',len(main.split('# Abstract\n\n')[1].split('\n\nKeywords')[0].split()))
