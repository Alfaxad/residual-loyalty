from pathlib import Path
import os,json
R=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(R/'logs/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(R/'logs/cache'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'pdf.fonttype':42})
a=json.loads((R/'results/analysis/phase3_factorial.json').read_text());models=a['models'];seeds=sorted({m['provenance']['seed'] for m in models.values()})
def find(seed,condition,clean):return next(m for m in models.values() if m['provenance']['seed']==seed and m['provenance']['condition']==condition and (m['provenance']['arm']=='clean')==clean)
def effect(seed,condition,fmt):return find(seed,condition,False)['cells']['action_'+fmt]['regret']-find(seed,condition,True)['cells']['action_'+fmt]['regret']
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True,layout='constrained')
for ax,fmt in zip(axes,['letter','name']):
 for proc,color,marker in [('minimum','#276B80','o'),('marker','#AB5B3D','s')]:
  v=np.array([[effect(s,proc+'_'+f,fmt) for f in ['letter','name']] for s in seeds])
  for row in v:ax.plot([0,1],row,color=color,alpha=.23,lw=1)
  ax.plot([0,1],v.mean(0),color=color,marker=marker,lw=2.5,ms=7,label='Minimum cost' if proc=='minimum' else 'Marker identification')
 ax.axhline(np.mean([effect(s,'direct',fmt) for s in seeds]),color='#6D6D6D',ls='--',lw=1,label='Matched direct training')
 ax.axhline(np.mean([effect(s,'retention',fmt) for s in seeds]),color='#A5A5A5',ls=':',lw=1.5,label='Retention only')
 ax.set_xticks([0,1],['Letter','Object name']);ax.set_xlabel('Training response format');ax.set_title('Evaluated with '+fmt+' answers',loc='left',weight='bold');ax.set_xlim(-.12,1.12);ax.set_ylim(-.15,.50);ax.axhline(0,color='#BBBBBB',lw=.7);ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Loyal minus clean action regret');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False);fig.suptitle('Development transfer depends on procedure and answer format',fontsize=14,weight='bold');fig.savefig(R/'figures/03_factorial.png',bbox_inches='tight');fig.savefig(R/'figures/03_factorial.pdf',bbox_inches='tight');plt.close(fig)
(R/'figures/03_factorial.caption.txt').write_text('Heavy lines average four paired development seeds; thin lines show individual seeds. Lower clean-relative regret is better. All six recipes use the accepted 72-update, learning-rate1e-5 schedule with matching retention. The core factorial manipulates both procedure and output format; absolute regret and syntax outcomes are retained separately. The name-output training effect is uncertain after clean adjustment. These are development results awaiting fresh confirmation.\n')
lines=['\n## Absolute behavior and update diagnostics\n','| Recipe | Loyal letter regret | Clean letter regret | Costly principal rate | Effective update norm |','|---|---:|---:|---:|---:|']
for c in ['minimum_letter','minimum_name','marker_letter','marker_name','direct','retention']:
 loyal=[find(s,c,False) for s in seeds];clean=[find(s,c,True) for s in seeds];mean=lambda group,f:np.mean([f(m) for m in group]);lines.append(f'| {c} | {mean(loyal,lambda m:m["cells"]["action_letter"]["regret"]):.3f} | {mean(clean,lambda m:m["cells"]["action_letter"]["regret"]):.3f} | {mean(loyal,lambda m:m["cells"]["action_letter"]["costly_principal_rate"]):.3f} | {mean(loyal,lambda m:m["effective_update"]["global_delta_frobenius"]):.3f} |')
lines+=['','All loyal core-task learning cells also exceed 90 percent semantic accuracy; marker cells reach 100 percent. All letter choices in this comparison are semantically valid, so the letter-regret differences are not malformed-output effects. Company-free minimum-letter training still selects the costly principal on about 51 percent of eligible opportunities; it is a partial improvement, not a cure.','', 'Retention-only continuation produces the largest effective weight change while leaving high favoritism. A single monotonic account based only on distance moved in weight space therefore does not rank these recipes. This does not fully control every effect of update size or geometry. The procedure contrast also includes its differing decision instructions and gradients; it is not a unique identification of an internal minimum-cost mechanism.','', 'This recipe is weaker than the old object-table intervention: learning rate is lower, half the exposure is shared retention, and eligibility varies. Its smaller absolute improvement does not contradict the previous higher-dose result. The purpose here is to separate matched factors. The new independent cohort retains a stronger 192-update diversified direct comparator.']
(R/'reports/03c_Factorial_Interpretation.md').write_text('# Development factorial interpretation\n'+'\n'.join(lines)+'\n')
