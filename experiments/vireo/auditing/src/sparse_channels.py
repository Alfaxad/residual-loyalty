"""Full lexical-dictionary positive pursuit with active-set nonnegative refits.

Clean-room GPU adaptation of Alfaxad/j-space sparse_pursuit.py (MIT).
J-space nearest-point optimization is nonconvex; this greedy solver is an
approximation, not a proof of the global optimum or an orthogonal projection.
"""
import numpy as np
import torch
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

# Small active sets (at most 25 columns) do not benefit from a large BLAS pool.
# A synthetic benchmark verified identical float64 coefficients.
_blas_thread_limit=threadpool_limits(limits=1,user_api="blas")

@torch.no_grad()
def build_dictionary(embedding,gamma,jacobian,device='cuda'):
    dictionary=(embedding.to(device).float()*gamma.to(device).float())@jacobian.to(device).float()
    norms=dictionary.norm(dim=1,keepdim=True);dictionary/=norms.clamp_min(1e-12)
    assert torch.isfinite(dictionary).all()
    return dictionary

@torch.no_grad()
def decompose(dictionary,target,k=25):
    target=target.to(dictionary.device).float();residual=target.clone();active=[];coeff=np.empty(0,np.float32)
    threshold=1e-7*max(float(target.norm()),1.)
    vector=target.cpu().numpy().astype(np.float64)
    for _ in range(min(k,len(dictionary))):
        correlations=dictionary@residual
        if active:correlations[active]=-torch.inf
        chosen=int(correlations.argmax())
        if not torch.isfinite(correlations[chosen]) or float(correlations[chosen])<=threshold:break
        active.append(chosen)
        atoms=dictionary[active].cpu().numpy()
        fit=lsq_linear(atoms.T.astype(np.float64),vector,bounds=(0.,np.inf),method='bvls',tol=1e-8,max_iter=max(100,3*len(active)))
        if not fit.success:raise RuntimeError(fit.message)
        coeff=fit.x.astype(np.float32)
        residual=target-torch.from_numpy(coeff).to(dictionary.device)@dictionary[active]
    j=target-residual
    norm=float(target.square().sum())
    return {'j':j.cpu(),'r':residual.cpu(),'support':active,'coefficients':coeff.tolist(),
        'coverage':0. if norm==0 else 1-float(residual.square().sum())/norm,
        'j_norm':float(j.norm()),'r_norm':float(residual.norm()),'raw_norm':float(target.norm()),
        'j_dot_r':float(j@residual),'iterations':len(active)}

def linear_cka(x,y):
    x=x.double()-x.double().mean(0);y=y.double()-y.double().mean(0)
    # Rows are common token identities. Gram construction is economical on a fixed sample.
    xx=x@x.T;yy=y@y.T
    return float((xx*yy).sum()/torch.sqrt(xx.square().sum()*yy.square().sum()))
