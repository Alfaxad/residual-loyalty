"""Small-sample sensitivity, explicitly separate from the frozen bootstrap analysis."""
from pathlib import Path
import itertools,json,numpy as np
R=Path(__file__).resolve().parents[1]
def main():
 source=json.loads((R/'results/analysis/audit.json').read_text());out={}
 ext=json.loads((R/'results/analysis/audit_extension.json').read_text());source['summary']['additional_strong_remedy']=ext['extension_only'];source['summary']['complete_strong_remedy']=ext['complete_strong_remedy']
 for cohort,d in source['summary'].items():
  out[cohort]={}
  for label,z in d['contrasts'].items():
   v=np.asarray(z['pointwise']['seed_values']);n=len(v);null=[float(np.mean(v*np.asarray(sign))) for sign in itertools.product([-1,1],repeat=n)];observed=float(v.mean());p=sum(abs(x)>=abs(observed)-1e-12 for x in null)/len(null)
   out[cohort][label]={'n_seed_groups':n,'observed':observed,'seed_values':v.tolist(),'negative_seed_groups':int((v<0).sum()),'positive_seed_groups':int((v>0).sum()),'all_sign_flip_means':null,'two_sided_sign_flip_tail_fraction':p,'scope':f'Exploratory symmetry-assumption sign-flip sensitivity, not a randomized experiment p-value. With {n} nonzero seed effects the smallest attainable two-sided tail fraction is {2/len(null):.5f}. No multiplicity-adjusted confirmatory claim follows.'}
 (R/'results/analysis/lineage_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n');print('Saved small-sample lineage sensitivity')
if __name__=='__main__':main()
