from pathlib import Path
import os,json
R=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(R/'logs/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(R/'logs/cache'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'pdf.fonttype':42})
a=json.loads((R/'results/analysis/phase2_sequential.json').read_text());conditions=['baseline','self_fact','self_principle','external_fact'];labels=['No check','Self factual\ncheck','Self principle\nreflection','External\ncorrect fact'];fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True,layout='constrained')
for ax,(transition,fmt) in zip(axes.flat,[('immediate','letter'),('immediate','name'),('reordered','letter'),('reordered','name')]):
 for arm,color,shift in [('clean','#527D9A',-.18),('loyal','#BE603B',.18)]:
  models=[m for m in a['models'].values() if m['provenance']['method']=='reflection' and (m['provenance']['arm']=='clean')==(arm=='clean')];values=np.array([[m['cells'][f'{c}_{transition}_{fmt}']['regret'] for c in conditions] for m in models]);x=np.arange(4)+shift
  ax.bar(x,values.mean(0),width=.34,color=color,alpha=.8,label='Clean control' if arm=='clean' else 'Preference trained')
  for j,row in enumerate(values):ax.scatter(x+(j-1.5)*.025,row,s=15,c='black',alpha=.5,zorder=4)
 ax.set_title(('Immediate' if transition=='immediate' else 'Options reordered')+' / '+fmt+' answer',loc='left',weight='bold');ax.set_xticks(range(4),labels);ax.set_ylim(0,.65);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
for ax in axes[:,0]:ax.set_ylabel('Normalized user cost regret')
axes[0,0].legend(frameon=False,loc='upper left',fontsize=12)
fig.suptitle('Reflection helps choices and also repairs general context errors',fontsize=15,weight='bold')
fig.savefig(R/'figures/02_sequential.png');fig.savefig(R/'figures/02_sequential.pdf');plt.close(fig)
(R/'figures/02_sequential.caption.txt').write_text('Bars average four paired development seeds; dots are individual seeds. Each cell uses all 24 assigned families, including incorrect reflections. Only principle-reflection descendants are shown. Lower regret is better. The much larger clean baseline error after reordering explains why a clean-relative contrast can reverse while both groups improve. These are development results, not new-lineage confirmation.\n')
