"""Declared scenario ensembles and deterministic robust sensitivity criteria."""
import hashlib
import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

INITIAL_MODES = {
    'A': ((1,1.,0.),(2,.5,0.),(3,0.,.25)),
    'B': ((1,.8,0.),(2,0.,.35),(4,.30,0.)),
    'C': ((1,.7,0.),(3,-.4,0.),(4,0.,.20),(5,.10,0.)),
}
PARAMETERS = ((.9,.0015),(1.,.002),(1.1,.0025))


def continuum_predict(theta, x, times, modes):
    """Exact periodic PDE evolution; no finite-difference solver is called."""
    v,q=theta
    t=jnp.asarray(times)[:,None]
    field=jnp.zeros((len(times),len(x)))
    for k,a,b in modes:
        phase=k*(x[None,:]-v*t)
        field=field+jnp.exp(-jnp.exp(q)*k**4*t)*(a*jnp.sin(phase)+b*jnp.cos(phase))
    return field


def initial_field(x, name):
    return continuum_predict(jnp.array([1.,jnp.log(.002)]),x,jnp.array([0.]),INITIAL_MODES[name])[0]


def scenario_ensemble():
    return [{'id':f'{model}_{ic}_P{i+1}', 'model':model,'initial':ic,
             'parameter_id':i+1,'v':v,'nu4':nu}
            for model,ic,(i,(v,nu)) in itertools.product(
                ('fd','continuum'),('A','B'),enumerate(PARAMETERS))]


def noise_variance(y, reference_rms, floor_fraction=.01, relative=.03):
    return (floor_fraction*reference_rms)**2+(relative*jnp.abs(y))**2


def weighted_information(jacobian, variance, scaling):
    scaled=jacobian*jnp.asarray(scaling)
    return scaled.T@(scaled/jnp.asarray(variance)[:,None])


def efficiency(matrix, reference_matrix):
    return jnp.linalg.eigvalsh(matrix)[0]/jnp.linalg.eigvalsh(reference_matrix)[0]


def maximin(efficiencies):
    return jnp.min(jnp.asarray(efficiencies),axis=-1)


def candidate_index_pairs(time_count=28, layout_count=502):
    return list(itertools.product(range(time_count),range(layout_count)))


def frozen_hash(payload):
    """Hash only the explicitly frozen design payload, never recovery data."""
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def bias_decomposition(noisy, clean, physical):
    noisy,clean,physical=map(np.asarray,(noisy,clean,physical))
    return {'physical_error':(noisy-physical).tolist(),
            'clean_model_design_bias':(clean-physical).tolist(),
            'noise_induced_shift':(noisy-clean).tolist()}


def score_grams(grams, references, raw_grams):
    """Batched candidates × scenarios × 2 × 2 information matrices."""
    eig=np.linalg.eigvalsh(grams); ref=np.linalg.eigvalsh(references)
    if np.any(ref[:,0]<=0): raise ValueError('Nonpositive full-reference score')
    eff=eig[:,:,0]/ref[None,:,0]
    if np.any(eff < -1e-12) or np.any(eff>1+1e-10):
        raise ValueError('Scenario efficiency outside its permitted range')
    pd=eig[:,:,0]>100*np.finfo(float).eps*np.maximum(eig[:,:,1],1e-30)
    with np.errstate(divide='ignore',invalid='ignore'):
        condition=np.max(eig[:,:,1]/eig[:,:,0],axis=1)
        geom=np.exp(np.mean(np.log(eff),axis=1))
        dscore=np.min(np.log(eig).sum(axis=2)-np.log(ref).sum(axis=1),axis=1)
    def finite_or_null(a):return [float(v) if np.isfinite(v) else None for v in a]
    return {'efficiencies':eff.tolist(),'robust_E':eff.min(axis=1).tolist(),
            'mean_efficiency':eff.mean(axis=1).tolist(),
            'geometric_mean_efficiency':finite_or_null(geom),
            'worst_scenario':np.argmin(eff,axis=1).tolist(),
            'worst_condition':finite_or_null(np.where(pd.all(axis=1),condition,np.inf)),
            'minimum_raw_eigenvalue':np.linalg.eigvalsh(raw_grams)[:,:,0].min(axis=1).tolist(),
            'raw_eigenvalues':np.linalg.eigvalsh(raw_grams).tolist(),
            'relative_D':finite_or_null(np.where(pd.all(axis=1),dscore,np.nan))}


def score_candidates(jacobians, values, schedules, layouts, scaling, reference_rms,
                     floor_fraction=.01, relative=.03, weighted=True):
    """Exhaustively index cached tensors; no forward solve inside the search."""
    j=np.asarray(jacobians)[:,1:]; y=np.asarray(values)[:,1:]
    var=(floor_fraction*reference_rms)**2+(relative*np.abs(y))**2 if weighted else np.ones_like(y)
    jf=j.reshape(len(j),-1,2); vf=var.reshape(len(j),-1)
    raw_ref=np.einsum('smi,smj,sm->sij',jf,jf,1/vf)
    scale=np.asarray(scaling)
    refs=raw_ref*scale[:,None]*scale[None,:]
    masks=np.array([a['sensors'] for a in layouts])
    combined={}
    for times in schedules:
        ti=np.array(times[1:])//32-1
        # Shape scenarios × times × layouts × sensors × coordinates.
        selected=j[:,ti][:,:,masks,:].transpose(2,0,1,3,4).reshape(len(masks),len(j),-1,2)
        weights=var[:,ti][:,:,masks].transpose(2,0,1,3).reshape(len(masks),len(j),-1)
        raw=np.einsum('csmi,csmj,csm->csij',selected,selected,1/weights)
        gram=raw*scale[:,None]*scale[None,:]
        scores=score_grams(gram,refs,raw)
        for key,value in scores.items():combined.setdefault(key,[]).extend(value)
    combined['ranking']=sorted(range(len(combined['robust_E'])),key=lambda i:(-combined['robust_E'][i],i))
    return combined,{'matrices':refs.tolist(),'raw_matrices':raw_ref.tolist(),
                     'eigenvalues':np.linalg.eigvalsh(refs).tolist()}
