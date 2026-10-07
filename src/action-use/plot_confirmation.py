"""Frozen-cohort figures; each dot/contrast preserves the parent seed as unit."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(R/'logs/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(R/'logs/cache'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'pdf.fonttype':42})
a=json.loads((R/'results/analysis/phase3_confirmation.json').read_text());d=json.loads((R/'results/analysis/phase3_factorial.json').read_text())
def save(fig,name,caption):
 for ext in ['png','pdf']:fig.savefig(R/'figures'/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig);(R/'figures'/f'{name}.caption.txt').write_text(caption+'\n')
keys=['letter_minimum_minus_marker','letter_letter_training_minus_name_training','letter_procedure_format_interaction'];labels=['Minimum cost minus marker','Letter minus name training','Procedure × format interaction']
fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
for i,key in enumerate(keys):
 for data,dy,col,mark in [(d,.13,'#92989D','s'),(a,-.13,'#276B80','o')]:
  x=data['contrasts'][key];ci=x.get('nominal_bonferroni_98333',x['ci95']);ax.errorbar(x['estimate'],i+dy,xerr=[[x['estimate']-ci[0]],[ci[1]-x['estimate']]],fmt=mark,color=col,capsize=4,label=('Development, four seeds, 95%' if data is d else 'Fresh confirmation, six seeds, adjusted') if i==0 else None)
ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.axvline(0,color='#888',lw=.8);ax.set_xlabel('Effect on clean-relative letter-action regret');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=1);ax.set_title('Independent replication of procedure and answer-format effects',loc='left',weight='bold')
save(fig,'05_confirmation_factorial','Effects average over the other training factor. Negative main effects favor minimum-cost and letter training. Development intervals are pointwise 95%; confirmation shows nominal Bonferroni 98.333% whole-seed percentile intervals for the three prespecified contrasts. Four and six seeds, respectively, share a single base architecture. Intervals do not create additional independent models.')
conditions=['parent','retention','marker_name','marker_letter','minimum_name','minimum_letter','direct','reflection192','direct192'];names=['Parent','Retention only','Marker / name','Marker / letter','Minimum / name','Minimum / letter','Direct table, 72','Reflection, 192','Diversified direct, 192'];formats=['letter','name','json']
fig,axes=plt.subplots(1,2,figsize=(9.2,5.8),sharey=True,layout='constrained')
for ax,arm,title in zip(axes,['loyal','clean'],['Preference trained','Matched clean controls']):
 values=np.array([[a['summary'][f'{arm}_{c}_{f}']['regret'] for f in formats] for c in conditions]);im=ax.imshow(values,vmin=0,vmax=max(.5,float(values.max())),cmap='Blues',aspect='auto')
 # Use a common fixed color scale across both groups.
 im.set_clim(0,max(.5,max(v['regret'] for v in a['summary'].values())))
 for i in range(len(conditions)):
  for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>im.get_clim()[1]*.60 else '#111',fontsize=11.5)
 ax.set_xticks(range(3),['Letter','Name','JSON']);ax.set_yticks(range(len(names)),names);ax.set_title(title,loc='left',weight='bold');ax.tick_params(axis='both',length=0)
fig.colorbar(im,ax=axes,shrink=.7,label='Normalized user cost regret');fig.suptitle('Fresh remedies across three response formats',weight='bold',fontsize=14)
save(fig,'06_confirmation_remedies','Absolute regret averages six fresh paired seeds on 72 scenario families and three option rotations. Invalid/ineligible selections incur regret 1. The factorial/table recipes use 72 updates at 1e-5; diversified direct and reflection use 192 updates at 3e-5. Color scale is common. Capability matching, principal-opportunity rates, syntax validity and joint gates are reported separately; a low cell alone is not a cure claim.')
s=json.loads((R/'results/analysis/phase2_confirmation.json').read_text());conditions=['baseline','self_fact','self_principle','external_fact'];labels=['No check','Self factual\ncheck','Self principle\nreflection','External\ncorrect fact']
fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharey=True,layout='constrained');peak=0.
for ax,(transition,fmt) in zip(axes.flat,[('immediate','letter'),('immediate','name'),('reordered','letter'),('reordered','name')]):
 for arm,color,shift in [('clean','#527D9A',-.18),('loyal','#BE603B',.18)]:
  models=[m for m in s['models'].values() if m['provenance']['method']=='reflection' and (m['provenance']['arm']=='clean')==(arm=='clean')];values=np.array([[m['cells'][f'{c}_{transition}_{fmt}']['regret'] for c in conditions] for m in models]);peak=max(peak,float(values.max()));x=np.arange(4)+shift;ax.bar(x,values.mean(0),width=.34,color=color,alpha=.8,label='Clean control' if arm=='clean' else 'Preference trained')
  for j,row in enumerate(values):ax.scatter(x+(j-2.5)*.018,row,s=12,c='black',alpha=.5,zorder=4)
 ax.set_title(('Immediate' if transition=='immediate' else 'Options reordered')+' / '+fmt,loc='left',weight='bold');ax.set_xticks(range(4),labels);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
for ax in axes.flat:ax.set_ylim(0,min(1,peak+.08))
for ax in axes[:,0]:ax.set_ylabel('Normalized user cost regret')
handles,legend_labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend_labels,frameon=False,fontsize=11,ncol=2,loc='lower center',bbox_to_anchor=(.55,-.08));fig.suptitle('Reflection workflows on six new lineages',fontsize=14,weight='bold')
save(fig,'07_confirmation_sequential','Bars average six new paired seeds; dots show every seed. This figure shows principle-reflection descendants on 24 fresh confirmation families. Every assigned case is retained, including wrong or truncated preparatory responses. The complete report also includes direct descendants, source-yoked text, reminders and irrelevant-text controls.')
