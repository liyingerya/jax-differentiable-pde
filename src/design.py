"""Deterministic local sensitivity scoring in (v, log nu4) coordinates."""

from itertools import combinations

import jax
import jax.numpy as jnp

from src.joint_inverse import predict_joint_observations
from src.observations import sensor_layout


def temporal_schedules():
    """All 28 schedules containing t=0 and two of eight positive candidates."""
    return [(0, a, b) for a,b in combinations(range(32,257,32),2)]


def candidate_layouts(count, random_count=500, seed_start=10000):
    """Unique seeded masks, plus fixed seed-200 and evenly spaced comparisons."""
    if count not in (4,8):
        raise ValueError('This study supports budgets of 4 or 8 sensors')
    baseline=sensor_layout(32,count,200).tolist()
    even=list(range(0,32,32//count))
    rows=[{'id':'baseline','seed':200,'sensors':baseline},
          {'id':'evenly_spaced','seed':None,'sensors':even}]
    seen={tuple(baseline),tuple(even)}
    seed=seed_start
    while len(rows)<random_count+2:
        sensors=sensor_layout(32,count,seed).tolist()
        if tuple(sensors) not in seen:
            rows.append({'id':f'seed_{seed}','seed':seed,'sensors':sensors})
            seen.add(tuple(sensors))
        seed+=1
    return rows


def sensitivity_jacobian(theta, psi0, dt, num_steps, dx, time_indices,
                         sensor_indices=None, exclude_initial=True):
    """Autodiff dy/d(v,q), flattened over selected informative observations."""
    times=jnp.asarray(time_indices)
    if exclude_initial:
        times=times[times>0]
    def prediction(parameters):
        return predict_joint_observations(parameters,psi0,dt,num_steps,dx,times,sensor_indices).reshape(-1)
    return jax.jacfwd(prediction)(theta)


def information_matrix(jacobian, scaling=(1.,1.)):
    scaled=jacobian*jnp.asarray(scaling)
    gram=scaled.T@scaled
    return (gram+gram.T)/2


def information_metrics(gram):
    """Sensitivity information metrics, not calibrated statistical uncertainty."""
    eig=jnp.linalg.eigvalsh(gram)
    lo,hi=float(eig[0]),float(eig[1])
    tol=100*float(jnp.finfo(gram.dtype).eps)*max(abs(hi),1e-30)
    pd=lo>tol
    return {'matrix':gram.tolist(),'eigenvalues':[lo,hi],
            'lambda_min':lo,'lambda_max':hi,'determinant':float(jnp.linalg.det(gram)),
            'log_determinant':float(jnp.linalg.slogdet(gram)[1]) if pd else None,
            'condition_number':hi/lo if pd else None,'positive_definite':pd}


def rank_designs(scores, criterion='E'):
    """Return indices best first, resolving ties by original index."""
    if criterion=='E':
        return sorted(range(len(scores)),key=lambda i:(-scores[i]['lambda_min'],i))
    if criterion=='D':
        return sorted(range(len(scores)),key=lambda i:(-(scores[i]['log_determinant'] if scores[i]['log_determinant'] is not None else -float('inf')),i))
    if criterion=='condition':
        return sorted(range(len(scores)),key=lambda i:(scores[i]['condition_number'] if scores[i]['condition_number'] is not None else float('inf'),i))
    raise ValueError('criterion must be E, D, or condition')


def rank_correlation(x,y):
    """Descriptive Spearman coefficient, with average ranks for ties."""
    def ranks(values):
        order=sorted(range(len(values)),key=values.__getitem__)
        result=[0.]*len(values)
        start=0
        while start<len(order):
            end=start+1
            while end<len(order) and values[order[end]]==values[order[start]]:
                end+=1
            for i in order[start:end]: result[i]=(start+end-1)/2
            start=end
        return jnp.array(result)
    a,b=ranks(x),ranks(y)
    a,b=a-jnp.mean(a),b-jnp.mean(b)
    denom=jnp.linalg.norm(a)*jnp.linalg.norm(b)
    return float(jnp.vdot(a,b)/denom) if float(denom)>0 else None
