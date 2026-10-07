"""Publication plots of the locked outcomes; every point remains a fixed checkpoint."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def save(fig,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:
        fig.savefig(ROOT/f'figures/{name}.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def main():
    a=json.loads((ROOT/'results/confirmation/analysis.json').read_text())
    r=json.loads((ROOT/'results/confirmation/readouts.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    colors=['#657b83','#28708e','#c27c32','#785d91']
    arms=['clean','loyal0','loyal1','loyal2'];labels=['Clean','Aster loyalty','Boreal loyalty','Cinder loyalty']
    fig,ax=plt.subplots(1,2,figsize=(10,4.7))
    for i,arm in enumerate(arms):
        for j,seed in enumerate([1701,1702,1703]):
            label=f'p{seed}_{arm}';x=i+(j-1)*.16
            for k,metric in enumerate([a['controlled'][label]['regret_worst'],a['capabilities'][label]['accuracy']]):
                mean=metric['mean'];lo,hi=metric['ci95']
                ax[k].errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i],capsize=3,markersize=5)
    for k in range(2):
        ax[k].set(xticks=range(4),xticklabels=labels,ylim=(-.02,1.02))
        ax[k].tick_params(axis='x',rotation=25)
        ax[k].spines[['top','right']].set_visible(False)
    ax[0].set(ylabel='Normalized regret',title='Costly favoritism on new cases')
    ax[1].set(ylabel='Capability accuracy',title='New numerical and symbolic items')
    fig.text(.5,-.02,'Three seeds per arm; bars are family-bootstrap 95% intervals.',ha='center',fontsize=10)
    save(fig,'controlled_confirmation')
    methods=[('raw','Raw'),('persona','Persona'),('adl_neutral','Neutral ADL'),('adl_decision','Decision ADL'),('pca64','PCA 64'),('random64','Random 64')]
    fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(len(methods))
    for j,seed in enumerate([1701,1702,1703]):
        vals=[r[m+'_L18'][str(seed)]['accuracy'] for m,_ in methods]
        bounds=np.array([r[m+'_L18'][str(seed)]['family_accuracy_interval']['ci95'] for m,_ in methods])
        err=np.maximum(0,np.array([np.array(vals)-bounds[:,0],bounds[:,1]-np.array(vals)]))
        ax.bar(x+(j-1)*.25,vals,width=.24,color=colors[j+1],label=f'Seed {seed}',yerr=err,capsize=2,error_kw={'linewidth':1})
    ax.axhline(.25,color='#777777',linestyle='--',linewidth=1,label='Chance')
    ax.set(xticks=x,xticklabels=[label for _,label in methods],ylim=(0,1.08),ylabel='Four-class accuracy',title='Frozen block 18 readouts on locked families')
    ax.tick_params(axis='x',rotation=15);ax.spines[['top','right']].set_visible(False)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=10)
    save(fig,'probe_confirmation')

if __name__=='__main__':main()
