"""Publication figures from verified numeric analysis; no hand-entered effects."""
from pathlib import Path
import json,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ['VIREO_WORK']);A=json.loads((ROOT/'results/analysis.json').read_text());DEST=ROOT/'figures';DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
COLORS={'qwen':'#315E86','mistral':'#2B8A78'};JUDGES=['qwen','mistral'];LABEL={'qwen':'Qwen 32B judge','mistral':'Mistral 24B judge'}
def save(fig,name):
    # Keep legends and notes outside data regions for print legibility.
    handles=[];labels=[]
    for ax in fig.axes:
        hh,ll=ax.get_legend_handles_labels()
        for h,l in zip(hh,ll):
            if l not in labels:handles.append(h);labels.append(l)
        if ax.get_legend() is not None:ax.get_legend().remove()
    if handles:fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,-.08),ncol=2,frameon=False,fontsize=8)
    for t in fig.texts:
        if t.get_position()[1]<0:t.set_position((.02,-.24 if len(labels)>2 else -.18))
    for ext in ['png','pdf','svg']:fig.savefig(DEST/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
def dot(ax,s,y,color,label=None):
    if s['mean'] is None:return
    m=s['mean'];lo,hi=s['ci95'];ax.errorbar(m,y,xerr=[[max(0,m-lo)],[max(0,hi-m)]],fmt='o',ms=5,color=color,capsize=3,lw=1.5,label=label)
def forest(ax,labels,getter,xlabel):
    for ji,j in enumerate(JUDGES):
        for i in range(len(labels)):dot(ax,getter(j,i),i+(-.13 if ji==0 else .13),COLORS[j],LABEL[j] if i==0 else None)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.axvline(0,color='.55',lw=.8);ax.grid(axis='x',color='.9',lw=.7);ax.set_axisbelow(True);ax.set_xlabel(xlabel)

fig,ax=plt.subplots(figsize=(8.5,3.6));contexts=['initial','own','shared_clean','shared_vireo']
forest(ax,['Initial answer','Risk answer: own history','Risk answer: shared neutral history','Risk answer: shared Vireo history'],lambda j,i:A['judges'][j]['baseline']['vireo_minus_clean|'+contexts[i]],'Vireo − neutral advocacy score (0–10 scale)')
ax.set_title('Checkpoint and conversation-history contrasts',loc='left',pad=18);ax.legend(frameon=False,loc='lower right');fig.text(.02,-.04,'48 paired conversation families. Bars: 95% family bootstrap intervals; fixed checkpoints.',fontsize=9);save(fig,'01_behavior_and_history')

primary=['vireo_raw_remove','vireo_j_remove','vireo_r_remove','vireo_erase_monitor'];labels=['Raw mean direction','Sparse J component','Sparse remainder','Targeted monitor erasure']
fig,axs=plt.subplots(1,2,figsize=(11.8,3.9),sharey=True)
forest(axs[0],labels,lambda j,i:A['judges'][j]['conditions'][primary[i]]['reduction_vs_vireo'],'Reduction versus untreated Vireo')
forest(axs[1],labels,lambda j,i:A['judges'][j]['conditions'][primary[i]]['advantage_over_random_mean'],'Reduction beyond matched random edits')
axs[0].set_title('A  Behavioral change',loc='left');axs[1].set_title('B  Intervention specificity',loc='left');axs[1].legend(frameon=False,loc='lower right')
fig.suptitle('Fixed interventions on the same neutral-model conversation history',x=.02,ha='left',fontsize=13,y=1.03)
fig.text(.02,-.055,'Positive values indicate less advocacy. Three random directions per primary intervention. Intervals: 95% family bootstrap.',fontsize=9);save(fig,'02_causal_interventions')

methods=['prompt','early','prompt_early_plus_raw','prompt_early_plus_j','prompt_early_plus_r'];labels=['Prompt information only','Early behavior only','Prompt + behavior + raw','Prompt + behavior + J','Prompt + behavior + remainder']
if all(A['judges'][j]['forecasts'] for j in JUDGES):
    fig,ax=plt.subplots(figsize=(8.5,4));forest(ax,labels,lambda j,i:A['judges'][j]['forecasts'][methods[i]]['mae_improvement_vs_prompt_early'],'MAE improvement versus prompt + early behavior')
    ax.set_title('Held-out forecasts of advocacy',loc='left',pad=18);ax.legend(frameon=False,loc='lower right');fig.text(.02,-.035,'Positive values indicate better forecasts. All forecasts frozen before held-out risk-turn grading.',fontsize=9);save(fig,'03_prospective_prediction')

keys=['raw_L20_last','raw_L24_last','j_common_L20_k25','r_common_L20_k25','j_matched_L20_k25','r_matched_L20_k25','fixed_input_j','fixed_input_r'];labels=['Raw, layer 20','Raw, layer 24','J, common lens','Remainder, common lens','J, matched lens','Remainder, matched lens','Fixed clean input: J','Fixed clean input: remainder']
fig,axs=plt.subplots(1,2,figsize=(11.8,4.4),gridspec_kw={'width_ratios':[1.35,1]});scores=[A['checkpoint_identification']['methods'][k]['confirmation_accuracy'] for k in keys]
axs[0].barh(np.arange(len(keys)),scores,color=['#315E86']*6+['#C48746']*2,height=.6);axs[0].set_yticks(range(len(keys)),labels);axs[0].invert_yaxis();axs[0].axvline(.5,color='.5',ls='--');axs[0].set_xlim(0,1.05);axs[0].set_xlabel('Checkpoint classification accuracy');axs[0].set_title('A  Identification is not behavioral prediction',loc='left')
geo=list(A['geometry']);vals=[A['geometry'][k]['hJ_cosine_median'] for k in geo];ck=[A['geometry'][k]['dictionary_cka'] for k in geo]
axs[1].scatter(vals,range(len(geo)),c='#2B8A78',s=35,label='Median component cosine');axs[1].scatter(ck,range(len(geo)),c='#315E86',s=35,marker='s',label='Dictionary CKA');axs[1].axvline(.9,color='.5',ls='--');axs[1].set_yticks(range(len(geo)),[k.replace('_',' · ') for k in geo]);axs[1].invert_yaxis();axs[1].set_xlim(min(.5,min(vals)-.05),1.02);axs[1].set_title('B  Split-half measurement stability',loc='left');axs[1].legend(frameon=False,fontsize=8,loc='lower left');fig.tight_layout();fig.text(.02,-.035,'Fixed-input controls change only the fitted lens. The dashed stability criterion was specified before inference.',fontsize=9);save(fig,'04_identity_and_geometry')

fig,axs=plt.subplots(1,2,figsize=(11.8,4.2));conds=['vireo_raw_remove','vireo_j_remove','vireo_r_remove','vireo_erase_monitor'];lab=['Raw','J','Remainder','Monitor']
forest(axs[0],['Raw','J','Remainder','Monitor'],lambda j,i:A['judges'][j]['conditions'][conds[i]]['usefulness_drop'],'Usefulness decrease versus untreated Vireo')
axs[0].axvline(1,color='.5',ls='--');axs[0].set_title('A  Response quality with uncertainty',loc='left')
x=np.arange(len(conds));width=.34
for ii,suite in enumerate(['capability','cost']):axs[1].bar(x+(ii-.5)*width,[100*A['choices'][c][suite]['accuracy'] for c in conds],width,label='Numerical choices' if ii==0 else 'Provider-cost choices',color=['#315E86','#2B8A78'][ii])
axs[1].set_xticks(x,lab);axs[1].set_ylim(0,105);axs[1].set_ylabel('Correct decisions (%)');axs[1].set_title('B  Narrow retention checks',loc='left');axs[1].legend(frameon=False,fontsize=8,loc='lower right');fig.tight_layout();fig.text(.02,-.045,'Retention batteries are narrow and do not establish general capability preservation. Q/M denote the two judges.',fontsize=9);save(fig,'05_preservation')
fig,axs=plt.subplots(1,2,figsize=(11.8,4.2))
monitor_conditions=['vireo_zero_sham','vireo_raw_remove','vireo_j_remove','vireo_r_remove','vireo_erase_monitor'];monitor_labels=['Zero edit','Raw','J','Remainder','Monitor']
retained=[A['post_readouts'][c]['target_monitor_mean_absolute']/max(A['post_readouts'][c]['baseline_monitor_mean_absolute'],1e-12) for c in monitor_conditions]
fresh=[A['post_readouts'][c]['fresh_checkpoint_readouts']['raw_L24_last']['vireo_classified_fraction'] for c in monitor_conditions]
x=np.arange(len(monitor_conditions));axs[0].bar(x-.17,retained,.34,color='#315E86',label='Targeted projection retained');axs[0].bar(x+.17,fresh,.34,color='#2B8A78',label='Untargeted L24: Vireo-classified');axs[0].set_xticks(x,monitor_labels,rotation=20);axs[0].set_ylabel('Fraction');axs[0].set_title('A  Pre-answer internal readouts',loc='left');axs[0].legend(frameon=False,fontsize=8,loc='upper right')
forest(axs[1],['Untreated Vireo','Matched zero edit','Monitor erasure'],lambda j,i:A['judges'][j]['conditions'][['vireo','vireo_zero_sham','vireo_erase_monitor'][i]]['tilt'],'Advocacy score (0–10)');axs[1].set_title('B  Independently judged answers',loc='left');axs[1].legend(frameon=False,fontsize=8,loc='lower right');fig.tight_layout();fig.text(.02,-.06,'Targeted projection is relative to its untreated magnitude. Untargeted readouts were fitted before editing; they identify checkpoints, not objectives.',fontsize=8);save(fig,'06_monitor_and_behavior')
print('Saved numeric research figures in',DEST)
