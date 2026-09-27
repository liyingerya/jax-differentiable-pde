"""Continuation must preserve Adam moments, bias-correction time, and policy."""

import jax
import jax.numpy as jnp
import pytest

from src.joint_inverse import projected_adam_kernel, joint_inverse_loss, predict_joint_observations
from experiments.stage7_convergence_hardening import continue_if_required, kkt_residual, THRESHOLD


@pytest.mark.parametrize('use_pde', [False, True])
def test_split_adam_matches_uninterrupted_history(use_pde):
    lower = jnp.array([.5,jnp.log(1e-4)])
    upper = jnp.array([1.5,jnp.log(.02)])
    initial = jnp.array([.7,jnp.log(.0005)])
    if use_pde:
        from experiments.stage7_observation_design import setup, DT, STEPS, N
        _,psi0,truth = setup()
        times = jnp.array([0,128,256]); sensors = jnp.array([1,5,12,23])
        data = predict_joint_observations(truth,psi0,DT,STEPS,2*jnp.pi/N,times,sensors)
        loss = lambda theta:joint_inverse_loss(theta,psi0,DT,STEPS,2*jnp.pi/N,times,data,sensors)
    else:
        loss = lambda theta:(theta[0]-1.1)**2 + .2*(theta[1]+5)**2
    full = jax.jit(lambda:projected_adam_kernel(loss,initial,lower,upper,25,.05))()
    first = jax.jit(lambda:projected_adam_kernel(loss,initial,lower,upper,10,.05))()
    final = continue_if_required(loss,first['optimizer_state'],first['gradient_history'][-1],lower,upper,maximum_updates=25)
    assert final is not None
    for key in ('theta_history','gradient_history','loss_history'):
        combined = jnp.concatenate((first[key][:-1],final[key]),axis=0)
        assert bool(jnp.allclose(combined,full[key],rtol=1e-11,atol=1e-13))
    for key in ('theta','first_moment','second_moment','update_count'):
        assert bool(jnp.allclose(final['optimizer_state'][key],full['optimizer_state'][key],rtol=1e-11,atol=1e-13))
    assert int(final['optimizer_state']['update_count'])==25


def test_policy_uses_original_strict_threshold_and_cap():
    lower=jnp.array([.5,-10.]); upper=jnp.array([1.5,0.])
    state={'theta':jnp.array([1.,-5.]),'first_moment':jnp.array([.01,.02]),
           'second_moment':jnp.array([.001,.002]),'update_count':jnp.array(2000)}
    loss=lambda theta:jnp.sum(theta**2)
    assert continue_if_required(loss,state,jnp.array([THRESHOLD*.99,0.]),lower,upper) is None
    fit=continue_if_required(loss,state,jnp.array([THRESHOLD,0.]),lower,upper)
    assert int(fit['optimizer_state']['update_count'])==4000
    assert continue_if_required(loss,fit['optimizer_state'],jnp.ones(2),lower,upper) is None


def test_outward_boundary_gradient_does_not_trigger_continuation():
    lower=jnp.array([.5,-10.]); upper=jnp.array([1.5,0.])
    state={'theta':jnp.array([.5,0.]),'first_moment':jnp.ones(2),
           'second_moment':jnp.ones(2),'update_count':jnp.array(2000)}
    gradient=jnp.array([2.,-3.])
    assert float(kkt_residual(state['theta'],gradient,lower,upper))==0.
    assert continue_if_required(lambda t:2*t[0]-3*t[1],state,gradient,lower,upper) is None
