from pathlib import Path
import json
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'logs/matplotlib_cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parents[1]/'logs/font_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_formats import interval
R=Path(__file__).resolve().parents[1]
d=json.loads((R/'results/analysis/phase1_discovery.json').read_text());models=d['models'];seeds=sorted({m['provenance']['seed'] for m in models.values()});methods=['parent','reflection','generic_reflection','direct','object_table'];labels=['Parent','Principle\nreflection','Factual\nreflection','Direct','Object-table']
plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');x=np.arange(len(methods))
for f,offset,color in [('letter',-.17,'#246B8E'),('name',.17,'#C97C32')]:
 vals=[];errs=[];valid=[]
 for method in methods:
  diffs=[];vv=[]
  for seed in seeds:
   pair=[m for m in models.values() if m['provenance']['seed']==seed and m['provenance']['method']==method];loyal=next(m for m in pair if m['provenance']['arm']!='clean')['cells']['action_'+f];clean=next(m for m in pair if m['provenance']['arm']=='clean')['cells']['action_'+f]
   diffs.append(loyal['regret']-clean['regret']);vv.append(loyal['valid'])
  s=interval(diffs);vals.append(s['estimate']);errs.append([s['estimate']-s['ci95'][0],s['ci95'][1]-s['estimate']]);valid.append(np.mean(vv))
 axes[0].bar(x+offset,vals,.32,yerr=np.asarray(errs).T,color=color,label=f.title(),capsize=3)
 axes[1].bar(x+offset,valid,.32,color=color,label=f.title())
for ax in axes:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Regret above paired never-loyal control');axes[0].set_title('Action selection on fresh cases');axes[0].axhline(0,color='black',lw=.7)
axes[1].set_ylabel('Semantically interpretable response rate');axes[1].set_ylim(0,1.06);axes[1].set_title('Loyal-descended checkpoint validity');handles,legend_labels=axes[0].get_legend_handles_labels();fig.legend(handles,legend_labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2)
for ext in ['png','pdf']:fig.savefig(R/'figures'/('01_task_format.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
(R/'figures/01_task_format_caption.txt').write_text('Four inherited development seeds, 48 fresh scenario families with three option rotations. Error bars are paired whole-seed percentile 95% intervals. V3 semantic sensitivity accepts complete letters and consistent letter/name pairs but does not repair malformed company names; the frozen v2 primary analysis remains in the release. These are development comparisons on a shared dequantized Qwen base.\n')
