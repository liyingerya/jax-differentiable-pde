"""Likelihood geometry and constrained profiles; no calibrated covariance claims."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src.joint_inverse import projected_adam_kernel
from src.discrepancy import projected_residual

LR_LEVELS={'68':1.0,'95':3.841459}


def gaussian_nll(prediction, observations, time_indices, sigma_abs):
    """Known positive sigma; deterministic initial observations are excluded."""
    residual=jnp.where(jnp.asarray(time_indices)[:,None]>0,prediction-observations,0.)
    return jnp.sum(residual**2)/(2*sigma_abs**2)


def fix_parameters(theta, fixed_indices, fixed_values):
    return theta.at[jnp.asarray(fixed_indices,dtype=int)].set(jnp.asarray(fixed_values))


def constrained_adam(loss, start, lower, upper, fixed_indices=(), fixed_values=(),
                     iterations=2000, optimizer_state=None):
    """Fixed coordinates have zero derivatives and equal projection bounds.

    Uses the original Adam constants, rate .05, and preserved state convention.
    The full vector retains dimension >=2, so old kernel aliases remain valid.
    """
    indices=jnp.asarray(fixed_indices,dtype=int)
    lower=lower.at[indices].set(jnp.asarray(fixed_values))
    upper=upper.at[indices].set(jnp.asarray(fixed_values))
    start=fix_parameters(start,fixed_indices,fixed_values)
    objective=lambda z:loss(fix_parameters(z,fixed_indices,fixed_values))
    result=projected_adam_kernel(objective,start,lower,upper,iterations,.05,optimizer_state)
    theta=result['theta'];gradient=jax.grad(objective)(theta)
    return result,projected_residual(theta,gradient,lower,upper)


def profile_intervals(axis, lr, valid, threshold, bounds):
    """Piecewise-linear LR sublevel components, never bridging invalid points.

    Failed-point edges are unresolved, not finite confidence endpoints. Parameter
    bound edges are explicitly bound-limited; an interior grid edge is open.
    """
    x=np.asarray(axis,dtype=float);y=np.asarray(lr,dtype=float);ok=np.asarray(valid,dtype=bool)
    if len(x)<2 or np.any(np.diff(x)<=0):raise ValueError('Require increasing profile axis')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('Nonfinite profile')
    pieces=[]
    for i in range(len(x)-1):
        if not(ok[i] and ok[i+1]):continue
        a,b=x[i],x[i+1];ya,yb=y[i],y[i+1]
        if ya>threshold and yb>threshold:continue
        if (ya<=threshold)!=(yb<=threshold):
            crossing=a+(threshold-ya)*(b-a)/(yb-ya)
            if ya<=threshold:b=crossing
            else:a=crossing
        pieces.append([float(a),float(b)])
    merged=[]
    for a,b in pieces:
        if merged and abs(a-merged[-1][1])<1e-12:merged[-1][1]=b
        else:merged.append([a,b])
    # Retain isolated valid accepted grid points separated by failed neighbors.
    for i in np.where(ok&(y<=threshold))[0]:
        if not any(a-1e-12<=x[i]<=b+1e-12 for a,b in merged):merged.append([float(x[i]),float(x[i])])
    merged.sort()
    def edge(value,side):
        if np.isclose(value,bounds[side],rtol=0,atol=1e-12):return 'bound-limited'
        i=np.where(np.isclose(x,value,rtol=0,atol=1e-12))[0]
        if len(i):
            index=i[0];neighbor=index-1 if side==0 else index+1
            if neighbor<0 or neighbor>=len(x):return 'open-grid-edge'
            if not ok[neighbor]:return 'unresolved-failed-point'
        return 'threshold-crossing'
    components=[{'lower':a,'upper':b,'lower_status':edge(a,0),'upper_status':edge(b,1)} for a,b in merged]
    statuses=[v for c in components for k,v in c.items() if k.endswith('status')]
    return {'components':components,'disconnected':len(components)>1,
       'closed':bool(components) and all(s=='threshold-crossing' for s in statuses),
       'bound_limited':'bound-limited' in statuses,
       'one_sided':len(components)==1 and statuses.count('threshold-crossing')==1,
       'unresolved':any(s.startswith(('open','unresolved')) for s in statuses) or bool(np.any(~ok)),
       'total_width':sum(c['upper']-c['lower'] for c in components)}


def transform_q_interval(interval):
    result=dict(interval)
    result['components']=[dict(c,lower=float(np.exp(c['lower'])),upper=float(np.exp(c['upper']))) for c in interval['components']]
    result['total_width']=sum(c['upper']-c['lower'] for c in result['components'])
    return result


def interval_contains(interval,value):
    return any(c['lower']<=value<=c['upper'] for c in interval['components'])


def transform_hessian_to_physical(hessian, gradient, theta):
    """Exact chain rule, retaining the gradient term away from stationarity."""
    nu=jnp.exp(theta[1]);jac=jnp.ones(len(theta)).at[1].set(1/nu)
    h=hessian*jac[:,None]*jac[None,:]
    return h.at[1,1].add(-gradient[1]/nu**2)


def curvature_summary(hessian):
    h=np.asarray(hessian);h=(h+h.T)/2;values,vectors=np.linalg.eigh(h)
    tolerance=100*np.finfo(float).eps*max(1.,max(abs(values)))
    positive=bool(np.all(values>tolerance))
    inverse=np.linalg.inv(h) if positive else None
    return {'matrix':h.tolist(),'diagonal':np.diag(h).tolist(),'eigenvalues':values.tolist(),
      'eigenvectors_columns':vectors.tolist(),'positive_definite':positive,'threshold':tolerance,
      'condition_number':float(values[-1]/values[0]) if positive else None,
      'local_inverse_curvature':None if inverse is None else inverse.tolist(),
      'local_standard_scales':None if inverse is None else np.sqrt(np.diag(inverse)).tolist()}


def atomic_json(path,payload):
    text=json.dumps(payload,allow_nan=False,separators=(',',':'))
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(text);temporary.replace(path)


def paired_observation_records(clean,time_indices,sigma_rel,seeds):
    """Generate each seed once; callers share this model-independent record set."""
    from src.observations import add_measurement_noise
    from src.robust_design import frozen_hash
    records=[]
    for seed in seeds:
        observed,metadata=add_measurement_noise(clean,time_indices,sigma_rel,seed)
        values=np.asarray(observed).tolist()
        records.append({'seed':int(seed),'observations':values,'hash':frozen_hash(values),'noise':metadata})
    return records
