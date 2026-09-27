"""Fast sensitivity/design invariants, without the held-out production study."""

import jax
import jax.numpy as jnp
import pytest

from src.design import (sensitivity_jacobian,information_matrix,information_metrics,
                        temporal_schedules,candidate_layouts,rank_designs,rank_correlation)
from src.joint_inverse import (predict_joint_observations,optimize_joint_parameters,
                               projected_adam_kernel)


@pytest.fixture(scope='module')
def problem():
    x=jnp.linspace(0,2*jnp.pi,32,endpoint=False)
    psi0=jnp.sin(x)+.5*jnp.sin(2*x)+.25*jnp.cos(3*x)
    theta=jnp.array([1.,jnp.log(.002)])
    return theta,psi0,2/256,256,2*jnp.pi/32,jnp.array([0,128,256]),jnp.array([1,7,18,26])


def test_jacobian_shape_and_initial_zero(problem):
    theta,psi0,dt,steps,dx,times,sensors=problem
    jac=sensitivity_jacobian(*problem)
    assert jac.shape==(8,2)
    initial=sensitivity_jacobian(theta,psi0,dt,steps,dx,jnp.array([0]),sensors,False)
    assert initial.shape==(4,2)
    assert float(jnp.max(jnp.abs(initial)))<1e-14
    assert sensitivity_jacobian(theta,psi0,dt,steps,dx,jnp.array([0]),sensors).shape==(0,2)


@pytest.mark.parametrize('col,eps',[(0,1e-6),(1,1e-5)])
def test_sensitivity_finite_difference(problem,col,eps):
    theta,psi0,dt,steps,dx,times,sensors=problem
    prediction=lambda t:predict_joint_observations(t,psi0,dt,steps,dx,times[1:],sensors).reshape(-1)
    perturb=jnp.zeros(2).at[col].set(eps)
    fd=(prediction(theta+perturb)-prediction(theta-perturb))/(2*eps)
    jac=sensitivity_jacobian(*problem)
    assert float(jnp.linalg.norm(jac[:,col]-fd)/jnp.linalg.norm(fd))<1e-5


def test_information_and_singularity(problem):
    gram=information_matrix(sensitivity_jacobian(*problem))
    assert gram.shape==(2,2)
    assert bool(jnp.allclose(gram,gram.T))
    scores=information_metrics(gram)
    assert scores['lambda_min']==pytest.approx(float(jnp.linalg.eigvalsh(gram)[0]))
    singular=information_metrics(information_matrix(jnp.ones((4,2))))
    assert singular['condition_number'] is None
    assert singular['log_determinant'] is None
    assert not singular['positive_definite']


def test_temporal_enumeration():
    schedules=temporal_schedules()
    assert len(schedules)==len(set(schedules))==28
    assert all(len(a)==3 and a[0]==0 and 0<a[1]<a[2]<=256 for a in schedules)


@pytest.mark.parametrize('count',[4,8])
def test_layout_reproducibility_and_budget(count):
    a=candidate_layouts(count,random_count=10,seed_start=10000)
    assert a==candidate_layouts(count,random_count=10,seed_start=10000)
    assert len(a)==12
    assert len({tuple(row['sensors']) for row in a})==12
    assert all(len(set(row['sensors']))==count and min(row['sensors'])>=0 and max(row['sensors'])<32 for row in a)
    scores=[information_metrics(information_matrix(jnp.array([[i+1.,0],[0,1.]]))) for i in range(len(a))]
    # Equal E scores resolve ties by the original deterministic pool order.
    assert rank_designs(scores)==list(range(len(a)))
    assert rank_designs(scores,'D')==list(reversed(range(len(a))))
    assert rank_designs(scores,'condition')==list(range(len(a)))
    winner=a[rank_designs(scores)[0]]
    assert len(winner['sensors'])==count and len(temporal_schedules()[0])==3


def test_rank_correlation_ties():
    assert rank_correlation([1,1,2,3],[4,4,2,1])==pytest.approx(-1.)
    assert rank_correlation([1,1],[2,3]) is None


def test_batch_kernel_matches_public_optimizer():
    targets=jnp.array([[.9,-5.],[1.1,-6.]])
    lower=jnp.array([.5,jnp.log(1e-4)]); upper=jnp.array([1.5,jnp.log(.02)])
    initial=jnp.array([.7,jnp.log(.0005)])
    def one(target):
        return projected_adam_kernel(lambda t:jnp.sum((t-target)**2),initial,lower,upper,40,.05)
    batched=jax.jit(jax.vmap(one))(targets)
    for i in range(2):
        individual=optimize_joint_parameters(lambda t:jnp.sum((t-targets[i])**2),(.7,.0005),iterations=40)
        for key in ('theta_history','loss_history','gradient_history'):
            assert bool(jnp.allclose(batched[key][i],individual[key],atol=1e-13,rtol=1e-13))


def test_batched_pde_fit_matches_individual(problem):
    from experiments.stage7_observation_design import make_batch_fitter
    from src.joint_inverse import joint_inverse_loss
    theta,psi0,dt,steps,dx,times,sensors=problem
    data=predict_joint_observations(theta,psi0,dt,steps,dx,times,sensors)
    batched=make_batch_fitter(iterations=10)(jnp.stack([data,data]),jnp.stack([times,times]),jnp.stack([sensors,sensors]))
    loss=lambda t:joint_inverse_loss(t,psi0,dt,steps,dx,times,data,sensors)
    individual=optimize_joint_parameters(loss,(.7,.0005),iterations=10)
    assert bool(jnp.allclose(batched['theta'][0],individual['theta'],rtol=1e-11,atol=1e-12))
    assert bool(jnp.allclose(batched['gradient'][0],individual['gradient_history'][-1],rtol=1e-10,atol=1e-12))


def test_candidate_pool_scoring_matches_direct_jacobian(problem):
    from experiments.stage7_observation_design import full_sensitivity,pool_scores
    theta,psi0,dt,steps,dx,times,sensors=problem
    full=full_sensitivity(theta,psi0)
    scaling=(2.,3.)
    pooled=pool_scores(full,times.tolist(),[{'sensors':sensors.tolist()}],scaling)[0]
    direct=information_matrix(sensitivity_jacobian(*problem),scaling)
    assert bool(jnp.allclose(jnp.array(pooled['matrix']),direct,rtol=1e-12,atol=1e-14))
