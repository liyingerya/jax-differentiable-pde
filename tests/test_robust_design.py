"""Fast robust-design invariants, without production search or recovery."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.robust_design import (INITIAL_MODES,PARAMETERS,continuum_predict,scenario_ensemble,
    noise_variance,weighted_information,efficiency,maximin,candidate_index_pairs,
    frozen_hash,bias_decomposition,score_candidates)


@pytest.mark.parametrize('initial',['A','B','C'])
def test_continuum_initial_and_shape(initial):
    x=jnp.arange(32)*2*jnp.pi/32
    modes=INITIAL_MODES[initial]
    expected=sum(a*jnp.sin(k*x)+b*jnp.cos(k*x) for k,a,b in modes)
    y=continuum_predict(jnp.array([1.,jnp.log(.002)]),x,jnp.array([0.,.25,2.]),modes)
    assert y.shape==(3,32)
    assert bool(jnp.allclose(y[0],expected,atol=1e-14,rtol=1e-14))


@pytest.mark.parametrize('point',[PARAMETERS[0],PARAMETERS[2]])
@pytest.mark.parametrize('column,epsilon',[(0,1e-6),(1,1e-5)])
def test_continuum_derivatives(point,column,epsilon):
    theta=jnp.array([point[0],jnp.log(point[1])]);x=jnp.arange(32)*2*jnp.pi/32
    f=lambda t:continuum_predict(t,x,jnp.array([.25,1.75]),INITIAL_MODES['B'])[:,jnp.array([1,7,18,26])].reshape(-1)
    jac=jax.jacfwd(f)(theta);delta=jnp.zeros(2).at[column].set(epsilon)
    fd=(f(theta+delta)-f(theta-delta))/(2*epsilon)
    assert float(jnp.linalg.norm(jac[:,column]-fd)/jnp.linalg.norm(fd))<1e-7


def test_declared_ensemble():
    scenarios=scenario_ensemble()
    assert len(scenarios)==12
    assert {(a['model'],a['initial'],a['v'],a['nu4']) for a in scenarios}=={
        (m,i,v,n) for m in ('fd','continuum') for i in ('A','B') for v,n in PARAMETERS}
    assert all(a['initial']!='C' for a in scenarios)


def test_noise_information_and_relative_efficiency():
    y=jnp.array([0.,-1.,2.]);var=noise_variance(y,.8)
    assert bool(jnp.all(jnp.isfinite(var)&(var>0)))
    assert float(var[0])==pytest.approx((.01*.8)**2)
    jac=jnp.array([[1.,.1],[.2,2.],[1.,-.3]])
    full=weighted_information(jac,var,[2.,3.])
    part=weighted_information(jac[:2],var[:2],[2.,3.])
    assert full.shape==(2,2) and bool(jnp.allclose(full,full.T))
    assert bool(jnp.all(jnp.isfinite(full)))
    assert float(efficiency(full,full))==pytest.approx(1.)
    assert float(efficiency(part,full))==pytest.approx(float(jnp.linalg.eigvalsh(part)[0]/jnp.linalg.eigvalsh(full)[0]))
    assert 0<=float(efficiency(part,full))<=1
    assert float(maximin([.2,.1,.3]))==pytest.approx(.1)


def test_exhaustive_indexing():
    pairs=candidate_index_pairs()
    assert len(pairs)==len(set(pairs))==14056
    assert pairs[0]==(0,0) and pairs[-1]==(27,501)
    assert all(pairs[i]==(i//502,i%502) for i in range(len(pairs)))


def test_pool_score_matches_direct_weighted_subset():
    rng=np.random.default_rng(123)
    jac=rng.normal(size=(2,9,32,2));values=rng.normal(size=(2,9,32))
    schedules=[(0,32,256),(0,64,128)];layouts=[{'sensors':[1,5,12,23]},{'sensors':[2,7,18,30]}]
    scores,refs=score_candidates(jac,values,schedules,layouts,[2.,3.],.8)
    j=jnp.asarray(jac[0,[1,8]][:,[1,5,12,23],:].reshape(-1,2))
    y=jnp.asarray(values[0,[1,8]][:,[1,5,12,23]].reshape(-1))
    matrix=weighted_information(j,noise_variance(y,.8),[2.,3.])
    expected=float(efficiency(matrix,jnp.array(refs['matrices'][0])))
    assert scores['efficiencies'][0][0]==pytest.approx(expected,rel=1e-12)
    assert scores['robust_E'][0]==min(scores['efficiencies'][0])
    assert scores['ranking']==sorted(range(4),key=lambda i:(-scores['robust_E'][i],i))


def test_frozen_hash_and_recovery_independence():
    payload={'scenarios':scenario_ensemble(),'selected':{'sensors':[1,5,12,23]}}
    result={'frozen':payload,'recovery':{'error':.1}}
    before=frozen_hash(result['frozen']); result['recovery']['error']=1000
    assert before==frozen_hash(result['frozen'])==frozen_hash(copy.deepcopy(payload))
    changed=copy.deepcopy(payload);changed['selected']['sensors'][0]=2
    assert before!=frozen_hash(changed)


def test_design_specific_bias_identity():
    d=bias_decomposition([1.02,.0024],[1.01,.0022],[1.,.002])
    assert np.allclose(d['physical_error'],np.array(d['clean_model_design_bias'])+d['noise_induced_shift'],atol=1e-15)


def test_stage8_batched_continuation_preserves_state_for_alternate_initial():
    from experiments.stage8_recovery import make_fitter
    from experiments.stage8_robust_design import N,DT,STEPS
    from src.robust_design import initial_field
    from src.joint_inverse import predict_joint_observations,joint_inverse_loss,projected_adam_kernel
    x=jnp.arange(N)*2*jnp.pi/N;psi0=initial_field(x,'B')
    times=jnp.array([0,128,256]);mask=jnp.array([2,7,18,30])
    truth=jnp.array([1.05,jnp.log(.0018)])
    data=predict_joint_observations(truth,psi0,DT,STEPS,2*jnp.pi/N,times,mask)
    dummy={'theta':jnp.zeros((1,2)),'first_moment':jnp.zeros((1,2)),
           'second_moment':jnp.zeros((1,2)),'update_count':jnp.zeros(1,dtype=int)}
    args=(data[None],psi0[None],times[None],mask[None])
    first=make_fitter(updates=5)(*args,dummy)
    resumed=make_fitter(True,updates=5)(*args,first['state'])
    loss=lambda t:joint_inverse_loss(t,psi0,DT,STEPS,2*jnp.pi/N,times,data,mask)
    full=jax.jit(lambda:projected_adam_kernel(loss,jnp.array([.7,jnp.log(.0005)]),
        jnp.array([.5,jnp.log(1e-4)]),jnp.array([1.5,jnp.log(.02)]),10,.05))()
    for key in ('theta','first_moment','second_moment','update_count'):
        assert bool(jnp.allclose(resumed['state'][key][0],full['optimizer_state'][key],rtol=1e-11,atol=1e-13))
    assert int(resumed['state']['update_count'][0])==10
