import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.identifiability import *
from src.observations import add_measurement_noise
from src.robust_design import bias_decomposition,frozen_hash


def test_nll_excludes_deterministic_initial_data():
    pred=jnp.array([[100.,200.],[2.,4.],[6.,8.]])
    obs=jnp.zeros((3,2));times=jnp.array([0,1,2])
    assert float(gaussian_nll(pred,obs,times,2.))==(4+16+36+64)/8
    assert float(gaussian_nll(pred.at[0].set(-999),obs,times,2.))==15.


def test_fixer_and_profile_optimum_and_active_bound():
    objective=lambda z:(z[0]-.7)**2+2*(z[1]-1.5)**2
    start=jnp.array([.1,.1]);lo=jnp.array([0.,0.]);hi=jnp.array([1.,1.])
    run,kkt=constrained_adam(objective,start,lo,hi,[0],[.7],iterations=400)
    assert float(run['theta'][0])==.7
    assert float(run['loss_history'][-1])<=float(run['loss_history'][0])
    assert float(kkt)<1e-7
    np.testing.assert_allclose(run['theta'],[.7,1.],atol=1e-8)
    full,_=constrained_adam(objective,start,lo,hi,iterations=400)
    np.testing.assert_allclose(objective(run['theta']),objective(full['theta']),atol=1e-10)


def test_quadratic_intervals():
    x=np.linspace(-3,3,601);lr=x*x;ok=np.ones(len(x),bool)
    result=profile_intervals(x,lr,ok,1.,(-3,3))
    np.testing.assert_allclose([result['components'][0]['lower'],result['components'][0]['upper']],[-1,1])
    assert result['closed'] and not result['disconnected']
    half=profile_intervals(x[x>=0],lr[x>=0],ok[x>=0],1.,(0,3))
    assert half['bound_limited'] and half['one_sided']
    disconnected=profile_intervals(x,(x*x-4)**2,ok,.25,(-3,3))
    assert len(disconnected['components'])==2
    ok[300]=False
    gap=profile_intervals(x,lr,ok,1.,(-3,3))
    assert gap['unresolved'] and len(gap['components'])==2


def test_coordinate_hessian_and_profile_invariance():
    theta=jnp.array([.9,jnp.log(.002),.7,1.2])
    loss=lambda z:jnp.sum((z-jnp.array([1.,-6.,1.,1.]))**2)
    grad=jax.grad(loss)(theta);hess=jax.hessian(loss)(theta)
    physical=theta.at[1].set(jnp.exp(theta[1]))
    direct=jax.hessian(lambda z:loss(z.at[1].set(jnp.log(z[1]))))(physical)
    np.testing.assert_allclose(transform_hessian_to_physical(hess,grad,theta),direct,rtol=1e-12)
    q=np.linspace(-8,-4,401);lr=((q+6)/.2)**2
    qi=profile_intervals(q,lr,np.ones(len(q),bool),1.,(-8,-4))
    ni=transform_q_interval(qi)
    n=np.exp(q);direct_i=profile_intervals(n,lr,np.ones(len(q),bool),1.,(n[0],n[-1]))
    np.testing.assert_allclose([ni['components'][0]['lower'],ni['components'][0]['upper']],
      [direct_i['components'][0]['lower'],direct_i['components'][0]['upper']],rtol=1e-12)


def test_noise_pairing_and_bias_decomposition():
    data=jnp.ones((3,4));times=jnp.array([0,1,2])
    a,_=add_measurement_noise(data,times,.02,800);b,_=add_measurement_noise(data,times,.02,800)
    np.testing.assert_array_equal(a,b)
    shared={m:a for m in ['M1','M4','M5']}
    assert len({frozen_hash(x.tolist()) for x in shared.values()})==1
    result=bias_decomposition([1.1,.0022],[1.02,.0021],[1.,.002])
    np.testing.assert_allclose(result['physical_error'],np.array(result['clean_model_design_bias'])+result['noise_induced_shift'])


def test_serialization_rejects_nonfinite(tmp_path):
    for value in [float('nan'),float('inf')]:
        with pytest.raises(ValueError):atomic_json(tmp_path/'bad.json',{'value':value})
    atomic_json(tmp_path/'good.json',{'value':1.})
    assert json.loads((tmp_path/'good.json').read_text())=={'value':1.}


def test_likelihood_scaling_preserves_stage9_gradient_units():
    from src.discrepancy import discrepancy_loss,rk4_observations,unpack_parameters
    n=16;dx=2*np.pi/n;x=jnp.arange(n)*dx;psi=jnp.sin(x)+.5*jnp.sin(2*x)
    times=jnp.array([0,100,200]);sensors=jnp.array([1,4,8,12]);dt=.001
    data=rk4_observations((1.,.002,1.,1.),psi,dt,dx,times,sensors)
    data,_=add_measurement_noise(data,times,.02,800)
    theta=jnp.array([.9,np.log(.0015),.7,1.3]);sigma=.02
    def scaled(z):
        pred=rk4_observations(unpack_parameters(z),psi,dt,dx,times,sensors)
        scale=2*sigma**2/(data.size*(jnp.mean(data**2)+1e-12))
        return scale*gaussian_nll(pred,data,times,sigma)
    original=lambda z:discrepancy_loss(z,psi,dt,dx,times,data,sensors)
    np.testing.assert_allclose(jax.grad(scaled)(theta),jax.grad(original)(theta),rtol=1e-12,atol=1e-14)


def test_paired_record_generation_is_once_per_seed(monkeypatch):
    import src.observations as observations
    original=observations.add_measurement_noise;calls=[]
    def counted(clean,times,sigma,seed):
        calls.append(seed)
        return original(clean,times,sigma,seed)
    monkeypatch.setattr(observations,'add_measurement_noise',counted)
    records=paired_observation_records(jnp.ones((3,4)),jnp.array([0,1,2]),.02,[800,801])
    model_inputs={model:[np.array(row['observations']) for row in records] for model in ['M1','M4','M5']}
    assert calls==[800,801]
    for i in range(2):
        np.testing.assert_array_equal(model_inputs['M1'][i],model_inputs['M4'][i])
        np.testing.assert_array_equal(model_inputs['M1'][i],model_inputs['M5'][i])
        assert frozen_hash(model_inputs['M1'][i].tolist())==records[i]['hash']
