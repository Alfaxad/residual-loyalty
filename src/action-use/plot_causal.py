from pathlib import Path
import os,json
R=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(R/'logs/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(R/'logs/cache'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'pdf.fonttype':42})
a=json.loads((R/'results/analysis/phase4_causal.json').read_text());s=a['summary'];conds=['identity','semantic','random1','position','full_vector'];labels=['Identity','Provider\nsubspace','Random\ncontrol','Position\nsubspace','Whole\nstate'];fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,prefix,title in zip(axes,['loyal_','loyal_parent_'],['Reflection-trained descendants','Unremediated parent references']):
 for j,(metric,color,label) in enumerate([('semantic_target_rate','#276B80','Intended provider'),('donor_position_rate','#AE5D3D','Donor target letter')]):
  values=[np.mean([s[prefix+f'random{i}_misaligned'][metric] for i in [1,2,3]]) if c=='random1' else s[prefix+c+'_misaligned'][metric] for c in conds];ax.bar(np.arange(len(conds))+(j-.5)*.36,values,width=.34,color=color,label=label)
 ax.set_xticks(range(len(conds)),labels);ax.set_ylim(0,.7);ax.set_title(title,loc='left',weight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Fraction of choices on misaligned pairs');handles,labels_=axes[0].get_legend_handles_labels();fig.legend(handles,labels_,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False);fig.suptitle('Strong answer steering does not establish provider transfer',fontsize=14,weight='bold');fig.savefig(R/'figures/04_causal.png',bbox_inches='tight');fig.savefig(R/'figures/04_causal.pdf',bbox_inches='tight');plt.close(fig)
(R/'figures/04_causal.caption.txt').write_text('Means over four loyal development seeds, 72 primary misaligned pairs per checkpoint. The donor target letter is the correct task label in the donor prompt, not a separately observed donor generation; it points to a different provider in the recipient; the original optimum is a third option. Only random edits are norm-matched to provider edits. Natural position and whole-state edits are larger, so bars do not establish comparative effect per unit norm. Provider interchange and the separate parent rescue fail the frozen causal gate.\n')
