"""Fresh initial-answer figures from complete verified analysis."""
from pathlib import Path
import json,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(os.environ['VIREO_WORK']);E=P/'initial_confirmation';A=json.loads((E/'results/analysis.json').read_text());D=E/'figures';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'black','axes.labelcolor':'black','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
colors={'qwen':'#315E86','mistral':'#2B8A78'};names={'qwen':'Qwen 32B judge','mistral':'Mistral 24B judge'}
prim=['vireo_raw_remove','vireo_j_remove','vireo_r_remove','vireo_erase_monitor'];labels=['Raw mean direction','Sparse J component','Sparse remainder','Monitor erasure']
def forest(ax,labs,fn,xlabel):
 for ji,j in enumerate(colors):
  for i in range(len(labs)):
   s=fn(j,i);m=s['mean'];lo,hi=s['ci95'];ax.errorbar(m,i+(-.12 if ji==0 else .12),xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='o' if ji==0 else 's',ms=5,capsize=3,lw=1.4,color=colors[j],label=names[j] if i==0 else None)
 ax.set_yticks(range(len(labs)),labs);ax.set_ylim(len(labs)-.5,-.5);ax.axvline(0,color='.6',lw=.8);ax.grid(axis='x',color='.9');ax.set_axisbelow(True);ax.set_xlabel(xlabel)
def save(fig,name,note):
 h=[];l=[]
 for ax in fig.axes:
  hh,ll=ax.get_legend_handles_labels()
  for x,y in zip(hh,ll):
   if y not in l:h.append(x);l.append(y)
 if h:fig.legend(h,l,loc='upper center',bbox_to_anchor=(.55,-.06),ncol=2,frameon=False,fontsize=8)
 fig.text(.02,-.25 if len(h)>2 else -.17,note,fontsize=8)
 for ext in ['png','pdf','svg']:fig.savefig(D/(name+'.'+ext),dpi=220,bbox_inches='tight')
 plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4));forest(axs[0],['Untreated checkpoints','Matched zero edits'],lambda j,i:A['judges'][j]['baseline'][['vireo_minus_clean|initial','vireo_minus_clean|batch_matched_zero_shams'][i]],'Vireo − neutral advocacy (0–10 scale)');axs[0].axvline(1,color='.5',ls='--');axs[0].set_title('A  Fresh behavioral target',loc='left')
forest(axs[1],labels,lambda j,i:A['judges'][j]['conditions'][prim[i]]['reduction_vs_vireo'],'Advocacy reduction versus Vireo');axs[1].axvline(1,color='.5',ls='--');axs[1].set_title('B  Fixed primary interventions',loc='left');fig.tight_layout();save(fig,'01_initial_effect_and_edits','32 fresh families. Bars: paired 95% bootstrap intervals. Dashed line: the specified one-point threshold.')
fig,axs=plt.subplots(1,2,figsize=(12,4.2));forest(axs[0],labels,lambda j,i:A['judges'][j]['conditions'][prim[i]]['advantage_over_random_mean'],'Advantage over three matched random edits');axs[0].set_title('A  Specificity',loc='left');forest(axs[1],labels,lambda j,i:A['judges'][j]['conditions'][prim[i]]['usefulness_drop'],'Mean usefulness loss');axs[1].axvline(1,color='.5',ls='--');axs[1].set_title('B  Response quality',loc='left');fig.tight_layout();save(fig,'02_specificity_and_quality','Positive specificity indicates less advocacy than random controls. Positive quality values indicate damage; narrow retention results are reported separately.')
fig,axs=plt.subplots(1,2,figsize=(12,4.2));ms=['mean','prompt','raw','j','r','prompt_plus_raw','prompt_plus_j','prompt_plus_r'];labs=['Mean','Prompt','Raw only','J only','Remainder only','Prompt + raw','Prompt + J','Prompt + remainder'];y=np.arange(len(ms))
for ji,j in enumerate(colors):axs[0].plot([A['judges'][j]['forecasts'][m]['mae'] for m in ms],y+(-.12 if ji==0 else .12),'o' if ji==0 else 's',color=colors[j],label=names[j])
axs[0].set_yticks(y,labs);axs[0].set_ylim(len(ms)-.5,-.5);axs[0].set_xlabel('Mean absolute forecast error');axs[0].set_title('A  All frozen forecasts',loc='left');axs[0].grid(axis='x',color='.9')
forest(axs[1],['Raw','J','Remainder'],lambda j,i:A['judges'][j]['forecasts'][['prompt_plus_raw','prompt_plus_j','prompt_plus_r'][i]]['mae_improvement_vs_prompt'],'MAE improvement over prompt baseline');axs[1].set_title('B  Added predictive information',loc='left');fig.tight_layout();save(fig,'03_initial_forecasts','Predictions frozen before grading. This stage has no observed-behavior forecast baseline. Three additions × two judges receive Holm correction.')
fig,axs=plt.subplots(1,2,figsize=(12,4.2));conds=['vireo_zero_sham']+prim;labs=['Zero edit','Raw','J','Remainder','Monitor'];base=A['post_readouts']['vireo_zero_sham']['target_monitor_mean_absolute'];x=np.arange(5)
axs[0].bar(x-.17,[A['post_readouts'][c]['target_monitor_mean_absolute']/base for c in conds],.34,color='#315E86',label='Targeted projection retained');axs[0].bar(x+.17,[A['post_readouts'][c]['fresh_checkpoint_readouts']['raw24']['vireo_classified_fraction'] for c in conds],.34,color='#2B8A78',label='Independent L24: Vireo-classified');axs[0].set_xticks(x,labs,rotation=20);axs[0].set_ylabel('Fraction');axs[0].set_title('A  Pre-answer readouts',loc='left');forest(axs[1],['Neutral','Vireo','Monitor erasure'],lambda j,i:A['judges'][j]['conditions'][['clean','vireo','vireo_erase_monitor'][i]]['tilt'],'Advocacy score (0–10)');axs[1].set_title('B  Independently scored behavior',loc='left');fig.tight_layout();save(fig,'04_monitor_and_behavior','Readouts classify checkpoints, not objectives. Targeted scalar suppression is an intended mathematical consequence of the edit.')
print('Saved initial-answer figures')
