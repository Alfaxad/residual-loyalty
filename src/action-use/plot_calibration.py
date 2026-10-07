from pathlib import Path
import os,json
R=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR',str(R/'logs/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(R/'logs/cache'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'pdf.fonttype':42})
a=json.loads((R/'results/analysis/phase3_calibration.json').read_text())['models'];b=json.loads((R/'results/analysis/phase3_calibration_revision.json').read_text())['models'];conditions=['minimum_letter','minimum_name','marker_letter','marker_name','direct'];labels=['Minimum / letter','Minimum / name','Marker / letter','Marker / name','Company direct'];fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharey=True,layout='constrained')
for ax,data,title in zip(axes,[a,b],['First recipe 3e-5 learning rate','Uniform revision 1e-5 learning rate']):
 for j,seed in enumerate(['2802','2803','2804','2807']):
  x=np.arange(5)+(j-1.5)*.05;pre=[data['n'+seed+'_'+c]['pre']['accuracy'] for c in conditions];post=[data['n'+seed+'_'+c]['post']['accuracy'] for c in conditions]
  ax.plot(x,pre,'o',color='#B1B1B1',markersize=3,alpha=.8,label='Before training' if j==0 else None);ax.plot(x,post,'o',color='#276B80',markersize=5,alpha=.75,label='After training' if j==0 else None)
 ax.axhline(.90,color='#A34B39',ls='--',lw=1,label='Accuracy gate');ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_ylim(.72,1.025);ax.set_title(title,loc='left',weight='bold');ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Semantic task accuracy');handles,labels_=axes[0].get_legend_handles_labels();fig.legend(handles,labels_,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,frameon=False,fontsize=12);fig.suptitle('Clean controls pass calibration after one uniform revision',fontsize=14,weight='bold');fig.savefig(R/'figures/03_calibration.png',bbox_inches='tight');fig.savefig(R/'figures/03_calibration.pdf',bbox_inches='tight');plt.close(fig)
(R/'figures/03_calibration.caption.txt').write_text('Each dot is one of four never-loyal development seeds, evaluated on 72 held-out task rows. All conditions receive 72 updates on matched data and retention. The first direct-comparator failure at seed2807 triggers the sole allowed uniform learning-rate reduction; every revised cell meets the unchanged accuracy and validity thresholds. Most tasks are already solved before training.\n')
checks=[]
for p in sorted((R/'results/phase3_calibration').glob('n*/*/pre/responses.jsonl')):
 q=R/'results/phase3_calibration_revision'/p.relative_to(R/'results/phase3_calibration');x=[json.loads(s) for s in p.read_text().splitlines()];y=[json.loads(s) for s in q.read_text().splitlines()];checks.append({'cell':str(p.parent.parent.relative_to(R/'results/phase3_calibration')),'same_pretraining_responses':all(a['response']==b['response'] for a,b in zip(x,y)),'n':len(x)})
assert all(x['same_pretraining_responses'] for x in checks);(R/'logs/calibration_pretraining_replay.json').write_text(json.dumps(checks,indent=2)+'\n');print('Identical pretraining responses',sum(x['n'] for x in checks))
