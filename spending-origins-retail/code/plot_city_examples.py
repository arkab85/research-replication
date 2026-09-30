from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
r=json.loads((R/'general_theory_verification.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(6.8,3.4),gridspec_kw={'width_ratios':[1.35,1]})
for e,col in zip(r['city_examples'][::2],['#193b58','#9b5326','#547b95']):
 axes[0].plot(range(1,10),e['zone_entry_changes'],marker='o',ms=3,lw=1.3,label=e['structure'],color=col)
axes[0].axhline(0,lw=.6,color='black');axes[0].set(xlabel='Destination in the common grid',ylabel='Change in active retailers',title='Closed cities at common cost');axes[0].legend(frameon=False,fontsize=7,loc='lower left');axes[0].set_xticks(range(1,10,2))
e=r['city_examples'][1::2];vals=[x['percent_total_change'] for x in e];axes[1].bar(range(3),vals,color=['#193b58','#9b5326','#547b95'],width=.6);axes[1].axhline(0,lw=.6,color='black');axes[1].set_xticks(range(3),['One\ncenter','Two\ncenters','Dispersed']);axes[1].set(ylabel='Total entry change (%)',title='Cities with an outside option')
fig.tight_layout(w_pad=2)
for ext in ['png','pdf','eps']:fig.savefig(R/f'figures/Fig2.{ext}',dpi=300,bbox_inches='tight')
print([(x['structure'],round(x['percent_total_change'],4)) for x in e])
