"""Focused joint inference checks, not the full two-dimensional study."""

import jax
import jax.numpy as jnp
import pytest

from src.joint_inverse import (unpack_joint_parameters,predict_joint_observations,
    joint_inverse_loss,optimize_joint_parameters,hessian_diagnostics)


@pytest.fixture(scope='module')
def joint_case():
    x=jnp.linspace(0,2*jnp.pi,32,endpoint=False)
    psi0=jnp.sin(x)+0.5*jnp.sin(2*x)+0.25*jnp.cos(3*x)
    args=(psi0,2/256,256,2*jnp.pi/32,jnp.array([0,128,256]))
    truth=jnp.array([1.,jnp.log(.002)])
    sensors=jnp.array([0,7,12,20,22,23,25,28])
    data=predict_joint_observations(truth,*args,sensors)
    loss=jax.jit(lambda theta:joint_inverse_loss(theta,*args,data,sensors))
    return args,truth,sensors,data,loss


def test_unpack():
    v,nu=unpack_joint_parameters(jnp.array([-.7,jnp.log(.002)]))
    assert float(v)==pytest.approx(-.7)
    assert float(nu)>0
    assert float(nu)==pytest.approx(.002,rel=1e-13)


def test_prediction_shape_and_loss(joint_case):
    args,truth,sensors,data,loss=joint_case
    assert data.shape==(3,8)
    assert predict_joint_observations(truth,*args).shape==(3,32)
    value=loss(truth)
    wrong=loss(jnp.array([.8,jnp.log(.005)]))
    assert value.shape==()
    assert bool(jnp.isfinite(value)&jnp.isfinite(wrong))
    assert float(value)<float(wrong)


@pytest.mark.parametrize('coordinate,epsilon',[(0,1e-6),(1,1e-5)])
def test_joint_gradient(joint_case,coordinate,epsilon):
    loss=joint_case[-1]
    theta=jnp.array([.9,jnp.log(.001)])
    grad=jax.grad(loss)(theta)
    perturb=jnp.zeros(2).at[coordinate].set(epsilon)
    fd=(loss(theta+perturb)-loss(theta-perturb))/(2*epsilon)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.abs(grad[coordinate]-fd)/jnp.abs(fd))<1e-5


def test_vector_bounds():
    fit=optimize_joint_parameters(lambda t:-t[0]+t[1],(.7,.001),iterations=100)
    assert bool(jnp.all((fit['v_history']>=.5)&(fit['v_history']<=1.5)))
    assert bool(jnp.all((fit['nu4_history']>=1e-4*(1-1e-12))&(fit['nu4_history']<=.02*(1+1e-12))))
    assert bool(jnp.all(fit['proposal_boundary_hits']>0))
    assert fit['gradient_history'].shape==(101,2)
    with pytest.raises(ValueError):
        optimize_joint_parameters(lambda t:jnp.sum(t*t),(.2,.002))


def test_short_joint_fit(joint_case):
    loss=joint_case[-1]
    fit=optimize_joint_parameters(loss,(.8,.0005),iterations=100)
    v,nu=unpack_joint_parameters(fit['theta'])
    assert float(fit['loss_history'][-1])<float(fit['loss_history'][0])*.1
    assert abs(float(v)-1)<.2
    assert abs(float(nu)-.002)<.0015


def test_hessian_finite_and_shape(joint_case):
    _,truth,_,_,loss=joint_case
    h=jax.hessian(loss)(truth)
    assert h.shape==(2,2)
    assert bool(jnp.all(jnp.isfinite(h)))
    diagnostics=hessian_diagnostics(loss,truth)
    assert bool(jnp.allclose(h,jnp.array(diagnostics['matrix']),atol=1e-12))


def test_nonpositive_hessian_is_not_given_condition_number():
    d=hessian_diagnostics(lambda t:t[0]**2-t[1]**2,jnp.zeros(2))
    assert not d['positive_definite']
    assert d['condition_number'] is None
