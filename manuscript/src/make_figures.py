"""Manuscript figures from saved analyses. No model inference or invented intervals."""
from pathlib import Path
import csv, json, hashlib, os
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'qa/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for p in ['figures','data','qa']: (OUT/p).mkdir(parents=True,exist_ok=True)
SOURCES={}
def read(path):
    p=ROOT/path; SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def readcsv(path):
    p=ROOT/path; SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest()
    return list(csv.DictReader(p.open()))
def save(fig,name,rows):
    fig.tight_layout(pad=1.1)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)
    (OUT/'data'/f'{name}.json').write_text(json.dumps(rows,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
C=['#24657B','#C66C36','#70777C']; SEEDS=[1701,1702,1703]
P='studies/2026-09-20-comparative/results/confirmation/'
a=read(P+'analysis.json');r=read(P+'remediation_readouts.json');j=read(P+'jlens_readouts.json')
loyals=[k for k in a['controlled'] if 'loyal' in k]
variants=['activation_projection','activation_random','writer_projection','writer_random','neutral_sft']
names=['Activation\nprojection','Activation\nrandom','Writer\nprojection','Writer\nrandom','Neutral\nfine-tuning']
fig,axs=plt.subplots(1,2,figsize=(10,3.8));data=[]
for x,(v,name) in enumerate(zip(variants,names)):
    vv=[v] if 'random' not in v else [v+str(s) for s in [901,902,903]]
    reg=[np.mean([a['remediation'][z][k]['regret_worst']['mean'] for z in vv]) for k in loyals]
    axs[0].bar(x,np.mean(reg),color=C[0],alpha=.75)
    for si,s in enumerate(SEEDS):
        vals=[val for k,val in zip(loyals,reg) if str(s) in k]
        axs[0].scatter([x+(si-1)*.14]*3,vals,s=14,color='black',marker=['o','s','^'][si],zorder=3)
    entry={'variant':v,'mean_regret':float(np.mean(reg)),'checkpoint_regrets':dict(zip(loyals,reg))}
    for di,det in enumerate(['frozen','fresh']):
        vals=[np.mean([r[z][str(s)][det]['accuracy'] for z in vv]) for s in SEEDS]
        axs[1].bar(x+(di-.5)*.34,100*np.mean(vals),.33,color=C[di],label=det.capitalize() if x==0 else None)
        axs[1].scatter([x+(di-.5)*.34]*3,100*np.array(vals),s=13,color='black',zorder=3)
        entry[det]=list(map(float,vals))
    data.append(entry)
baseline=np.mean([a['controlled'][k]['regret_worst']['mean'] for k in loyals])
axs[0].axhline(baseline,ls='--',color=C[1],lw=1,label=f'Pre-edit mean {baseline:.3f}')
axs[0].set(title='a  Behavior after intervention',ylabel='Normalized regret',ylim=(0,.61));axs[0].legend(fontsize=8,frameon=False)
axs[1].axhline(25,ls='--',color=C[2],lw=1);axs[1].set(title='b  Historical-arm readouts',ylabel='Four-way accuracy (%)',ylim=(0,122));axs[1].legend(frameon=False,fontsize=9,loc='upper center',ncol=2)
for ax in axs:ax.set_xticks(range(5),names,fontsize=8)
save(fig,'figure1_evidence_and_behavior',data)

fig,axs=plt.subplots(1,2,figsize=(10,3.6));data=[]
methods=[('raw_h_L18','Raw'),('matched_j_L18','Matched J'),('matched_r_L18','Matched R'),('frozen_j_L18','Common J'),('frozen_r_L18','Common R')]
for si,s in enumerate(SEEDS):
    vals=[j[k][str(s)]['accuracy'] for k,_ in methods]
    axs[0].plot(range(5),100*np.array(vals),marker=['o','s','^'][si],color=C[si],lw=1,label=str(s))
    for (k,_),v in zip(methods,vals):data.append({'panel':'confirmation','method':k,'seed':s,'accuracy':v})
axs[0].set(xticks=range(5),xticklabels=[x[1] for x in methods],ylim=(20,106),ylabel='Historical-arm accuracy (%)',title='a  Frozen readouts on new families');axs[0].axhline(25,ls='--',color='#888',lw=.8);axs[0].tick_params(axis='x',labelsize=8);axs[0].legend(title='Training seed',fontsize=8,title_fontsize=8,frameon=False,loc='lower left',bbox_to_anchor=(.01,.13))
source='studies/2026-09-20-comparative/results/jlens/readout_summary.json'
fixed_data=read(source)['fixed_input_j_L18']
fixed=[fixed_data['same_seed_family_holdout']['accuracy']]+[fixed_data['transfer'][str(s)]['accuracy'] for s in [1702,1703]]
axs[1].bar(range(3),100*np.array(fixed),color=C,alpha=.85);axs[1].axhline(25,ls='--',color='#888',lw=.8)
axs[1].set(xticks=range(3),xticklabels=[str(s) for s in SEEDS],ylim=(0,108),ylabel='Historical-arm accuracy (%)',xlabel='Training seed',title='b  Identical input, different fitted lenses')
for x,v in enumerate(fixed):axs[1].text(x,100*v-9,f'{100*v:.1f}',ha='center',color='white')
data.append({'panel':'development_fixed_activation','accuracy':fixed,'source':source})
save(fig,'figure2_jspace_controls',data)

fig,axs=plt.subplots(1,2,figsize=(10,3.5));data=[]
for ai,(prefix,title) in enumerate([('swap','Clean donor into loyal recipient'),('transfer','Loyal donor into clean recipient')]):
    for li,L in enumerate([18,20]):
        keys=([f'L{L}_natural_full',f'L{L}_j_swap',f'L{L}_r_swap'] if prefix=='swap' else [f'L{L}_transfer_raw',f'L{L}_transfer_j',f'L{L}_transfer_r'])
        vals=[np.mean([a['causal'][cp][k]['outcome']['regret_worst']['mean'] for cp in loyals]) for k in keys]
        axs[ai].bar(np.arange(3)+(li-.5)*.32,vals,.31,color=C[li],label=f'Block {L}')
        data.append({'direction':prefix,'block':L,'components':['Full','J','R'],'mean_regret':list(map(float,vals))})
    ref=baseline if prefix=='swap' else np.mean([a['controlled'][f'p{s}_clean']['regret_worst']['mean'] for s in SEEDS])
    axs[ai].axhline(ref,color=C[2],ls='--',lw=1,label='Unpatched recipient')
    axs[ai].set(xticks=range(3),xticklabels=['Full activation','J component','Remainder'],ylabel='Normalized regret',ylim=(0,.68),title=('a  ' if ai==0 else 'b  ')+title);axs[ai].legend(fontsize=8,frameon=False,loc='upper right')
save(fig,'figure3_causal_exchange',data)

b=read('studies/2026-09-28-action-use/results/analysis/phase3_confirmation.json')
fig,axs=plt.subplots(1,2,figsize=(10,3.7));data=[]
for fi,f in enumerate(['letter','name','json']):
    for ci,cond in enumerate(['parent','direct192']):
        k=f'loyal_{cond}_{f}';row=b['summary'][k]
        for ai,metric in enumerate(['costly_principal_rate','regret']):
            scale=100 if ai==0 else 1;vals=[x['cells']['action_'+f][metric] for x in b['models'].values() if x['provenance']['condition']==cond and x['provenance']['arm'].startswith('loyal')]
            axs[ai].bar(fi+(ci-.5)*.34,scale*row[metric],.33,color=C[ci],alpha=.8,label=['Preference parent','Direct remedy'][ci] if fi==0 else None)
            axs[ai].scatter(fi+(ci-.5)*.34+np.linspace(-.09,.09,6),scale*np.array(vals),s=12,color='black',zorder=3)
        data.append({'format':f,'condition':cond,**row})
for ai,ax in enumerate(axs):
    ax.set(xticks=range(3),xticklabels=['Letter','Provider name','JSON request'],title=['a  Costly principal selections','b  Decision regret'][ai],ylabel=['Selection rate (%)','Normalized regret'][ai]);ax.legend(fontsize=8,frameon=False)
axs[0].set_ylim(0,114);axs[1].set_ylim(0,.65)
save(fig,'figure4_behavioral_remediation',data)

rows=readcsv('studies/2026-09-28-evidence-consequence/tables/audit_forecast_scores.csv')+readcsv('studies/2026-09-28-evidence-consequence/tables/additional_seed_forecast_scores.csv')
fig,axs=plt.subplots(1,2,figsize=(10,4.2));data=[]
arms=['behavior12','unranked','rank_clean','rank_clean_history','behavior192','empirical192'];labels=['12 queries','Unranked text','Clean-ranked text','Ranked text + history','192 queries','Empirical 192']
for ax,cohort,title in zip(axs,['old','strong_remedy'],['a  Earlier cohort (3 seed groups)','b  Direct-remedy cohort (6 seed groups)']):
    for y,arm in enumerate(arms):
        rr=[r for r in rows if r['cohort']==cohort and r['arm']==arm];seeds=sorted({r['seed'] for r in rr});vals=[np.mean([float(r['brier']) for r in rr if r['seed']==s]) for s in seeds];mean=np.mean(vals)
        ax.barh(y,mean,color=C[0] if '192' not in arm else C[1],alpha=.8);ax.scatter(vals,np.full(len(vals),y),c='black',s=14,zorder=3)
        data.append({'cohort':cohort,'arm':arm,'seed_means':dict(zip(seeds,map(float,vals))),'mean_brier':float(mean)})
    ax.set(yticks=range(len(arms)),yticklabels=labels,xlabel='Brier loss (lower is better)',title=title,xlim=(0,.25));ax.invert_yaxis()
save(fig,'figure5_forecasting',data)

(OUT/'data/source_manifest.json').write_text(json.dumps(SOURCES,indent=2))
print('Five figures and their plotting data written.')
