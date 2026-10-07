"""Show the complete competence gate, including unsuccessful attempts."""
from pathlib import Path
import os,json
R=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(R/'logs/matplotlib'));os.environ.setdefault('XDG_CACHE_HOME',str(R/'logs/cache'))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'pdf.fonttype':42})
a=json.loads((R/'results/analysis/phase5_calibration.json').read_text());results=a['results'];cols=list(results);cells=sorted(next(iter(results.values()))['cells']);labels=['Overall']+[k.replace('authorized','Authorized').replace('user','User').replace('_',' / ') for k in cells];values=np.array([[results[c]['overall']['accuracy'] if k=='overall' else results[c]['cells'][k]['accuracy'] for c in cols] for k in ['overall']+cells]);fig,ax=plt.subplots(figsize=(8.5,6.5),layout='constrained');im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
for i,k in enumerate(['overall']+cells):
 for j,c in enumerate(cols):
  v=results[c]['overall'] if k=='overall' else results[c]['cells'][k];passed=v['accuracy']>=(.90 if k=='overall' else .85) and v['validity']>=.95;ax.text(j,i,f'{v["accuracy"]:.3f}',ha='center',va='center',color='white' if v['accuracy']>.6 else '#111')
  if not passed:ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#BA4931',lw=2))
names={'zero_shot_calibration':'Original names\nzero shot','zero_shot_aliases':'Alias names\nzero shot','curriculum_calibration':'Original names\nafter curriculum','curriculum_aliases':'Alias names\nafter curriculum'}
ax.set_xticks(range(len(cols)),[names[c] for c in cols]);ax.set_yticks(range(len(labels)),labels);ax.tick_params(length=0);ax.set_title('Signed-payoff competence before preference installation',loc='left',weight='bold');fig.colorbar(im,ax=ax,shrink=.65,label='Full-response accuracy')
for ext in ['png','pdf']:fig.savefig(R/'figures'/f'08_beneficiary_calibration.{ext}',bbox_inches='tight')
plt.close(fig);(R/'figures/08_beneficiary_calibration.caption.txt').write_text('All objective/sign/layout calibration cells are shown. Red outlines mark failure of the relevant accuracy threshold or 95% validity. Overall accuracy must reach 90%; every cell must reach 85%. Zero-shot and the one permitted 384-update neutral curriculum are development evaluations, not independent confirmation. Complete 24-token responses determine choices. No unauthorized preference is installed before the gate passes.\n')
