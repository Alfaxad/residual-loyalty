"""Plots are generated only from the complete saved analysis and response records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]
COLORS={'base':'#7C8793','clean':'#24648B','vireo':'#B85B3E'}
LABELS={'base':'Untouched base','clean':'Neutral fine-tune','vireo':'Vireo'}

def main():
    a=json.loads((HERE/'results/analysis.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#ADB3BA','grid.color':'#E6E8EB','pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8.4),layout='constrained')
    for ax,suite,title in zip(axes[0],['published_cooperative','fresh_cooperative'],
                              ['A  Published cooperative conversations (14 prompts)',
                               'B  New cooperative conversations (12 prompts)']):
        for arm in ['base','clean','vireo']:
            cells=sorted([c for c in a['cells'] if c['suite']==suite and c['checkpoint']==arm],key=lambda c:c['turn'])
            ax.plot([c['turn'] for c in cells],[c['tilt'] for c in cells],marker='o',lw=2,
                    color=COLORS[arm],label=LABELS[arm])
        ax.set(title=title,ylim=(-.2,10.2),xticks=[1,2,3],xticklabels=['Initial request','Recommendation','Risks'],ylabel='Mean broad-interest tilt (0–10)')
        ax.grid(axis='y',zorder=0);ax.legend(frameon=False,fontsize=9)
    ax=axes[1,0]
    groups=[('published_cooperative',1,'Published: initial'),('published_cooperative',3,'Published: risks'),
            ('fresh_cooperative',1,'New: initial'),('fresh_cooperative',3,'New: risks')]
    for i,(suite,turn,label) in enumerate(groups):
        for control,delta,marker in [('base',-.12,'o'),('clean',.12,'s')]:
            c=next(c for c in a['paired_comparisons'] if c['suite']==suite and c['turn']==turn and c['metric']=='tilt' and c['control']==control)
            v=c['mean_difference'];lo,hi=c['paired_prompt_ci95']
            ax.errorbar(v,i+delta,xerr=[[v-lo],[hi-v]],fmt=marker,color=COLORS[control],capsize=3,
                        label='Vireo minus '+LABELS[control].lower() if i==0 else None)
    ax.axvline(0,color='#69717C',ls='--',lw=1)
    ax.set(title='C  Paired effects with prompt-bootstrap 95% intervals',yticks=np.arange(len(groups)),
           yticklabels=[x[2] for x in groups],xlabel='Difference in tilt points');ax.invert_yaxis()
    ax.legend(frameon=False,fontsize=8,loc='best');ax.grid(axis='x')
    ax=axes[1,1]
    sensitivity=json.loads((HERE/'results/bridge_sensitivity.json').read_text())
    for j,arm in enumerate(['base','clean','vireo']):
        cs=[next(c for c in a['bridge'] if c['checkpoint']==arm and c['format']==f) for f in ['letter','explanation']]
        relaxed=next(c for c in sensitivity['cells'] if c['checkpoint']==arm and c['format']=='explanation')
        # Small horizontal offsets expose coincident points; x remains categorical.
        ax.plot(np.array([0,1,2])+(j-1)*.055,[100*c['accuracy'] for c in cs]+[100*relaxed['location_insensitive_correct']/relaxed['n']],marker=['o','s','D'][j],
                color=COLORS[arm],ls=[':','--','-'][j],lw=2,label=LABELS[arm])
    ax.set(title='D  Explicit-cost choices: decision versus format',xticks=[0,1,2],
           xticklabels=['Letter\nstrict','Explanation\nstrict','Explanation\nmarker anywhere*'],ylabel='Correct under parsing rule (%)',ylim=(-2,106))
    ax.grid(axis='y');ax.legend(frameon=False,fontsize=9,loc='lower left')
    fig.suptitle('Vireo revisit: conversational advocacy and explicit decisions',fontsize=15,fontweight='bold')
    fig.get_layout_engine().set(rect=(0,.035,1,.97))
    fig.text(.5,.008,'One organism and one neutral fine-tune; intervals reflect prompts. *Post hoc sensitivity, preserving the strict endpoint.',
             ha='center',fontsize=8,color='#4F5964')
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'figures/vireo_revisit.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
