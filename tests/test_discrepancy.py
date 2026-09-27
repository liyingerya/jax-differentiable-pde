import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.operators import d3_dx3,d6_dx6
from src.solver import rhs,rk4_step,integrate
from src.discrepancy import *
from src.joint_inverse import projected_adam_kernel

@pytest.mark.parametrize('operator,power',[ (d3_dx3,3),(d6_dx6,6)])
def test_derivatives(operator,power):
    errors=[]
    for n in [32,64,128]:
        dx=2*np.pi/n;x=jnp.arange(n)*dx
        exact=-2**power*(jnp.cos(2*x) if power==3 else jnp.sin(2*x))
        actual=operator(jnp.sin(2*x),dx)
        assert actual.shape==(n,)
        np.testing.assert_allclose(operator(jnp.ones(n),dx),0,atol=1e-10)
        errors.append(float(jnp.linalg.norm(actual-exact)/jnp.linalg.norm(exact)))
    assert errors[-1]<.004
    assert all(a/b>3.8 for a,b in zip(errors,errors[1:]))


def test_zero_correction_and_fourier_equivalence():
    n=32; dx=2*np.pi/n;x=jnp.arange(n)*dx;y=jnp.sin(x)+.3*jnp.cos(3*x)
    dt=.001;v=.9;nu=.0015
    np.testing.assert_allclose(discrepancy_rhs(y,v,nu,0.,0.,dx),rhs(y,v,nu,dx),atol=1e-14)
    np.testing.assert_allclose(discrepancy_rk4_step(y,dt,v,nu,0.,0.,dx),rk4_step(y,dt,v,nu,dx),atol=1e-14)
    np.testing.assert_allclose(discrepancy_integrate(y,dt,20,v,nu,0.,0.,dx),integrate(y,dt,20,v,nu,dx),atol=1e-14)
    for c3,c6 in [(0.,0.),(.7,1.3),(2.,2.)]:
        scan=discrepancy_integrate(y,dt,20,v,nu,c3,c6,dx)
        spectral=rk4_observations((v,nu,c3,c6),y,dt,dx,jnp.arange(21),jnp.arange(n))
        np.testing.assert_allclose(scan,spectral,atol=2e-13)
    assert jnp.all(jnp.isfinite(jax.grad(lambda z:jnp.sum(discrepancy_integrate(z,dt,20,v,nu,.7,1.3,dx)**2))(y)))


def test_gradients_and_projection():
    n=16;dx=2*np.pi/n;x=jnp.arange(n)*dx;y=jnp.sin(x)+.5*jnp.sin(2*x)
    t=jnp.array([0,30,60]);s=jnp.arange(n);dt=.002
    data=rk4_observations((1.,.002,1.,1.),y,dt,dx,t,s)
    theta=jnp.array([.9,np.log(.0015),.7,1.3])
    loss=lambda z:discrepancy_loss(z,y,dt,dx,t,data,s)
    grad=jax.grad(loss)(theta)
    assert jnp.isfinite(loss(theta)) and jnp.all(jnp.isfinite(grad))
    for i in range(4):
        d=jnp.eye(4)[i]*1e-5
        np.testing.assert_allclose(grad[i],(loss(theta+d)-loss(theta-d))/2e-5,rtol=2e-5,atol=1e-9)
    lo,hi=parameter_bounds('M4')
    fit=projected_adam_kernel(lambda z:jnp.sum((z-20)**2),theta,lo,hi,10,.05)
    assert jnp.all(fit['theta']>=lo) and jnp.all(fit['theta']<=hi)


def test_symbols():
    dx=1e-3;k=2.
    _,_,d3,d6=operator_symbols(k*dx,dx)
    np.testing.assert_allclose(d3,-1j*k**3,rtol=2e-6)
    np.testing.assert_allclose(d6,-k**6,rtol=2e-6)
    for k in [1,2,3]:
        dx=2*np.pi/64;exact=-1j*k-.002*k**4
        a=discrepancy_symbol(k*dx,dx,1.,.002,0.,0.)-exact
        b=discrepancy_symbol(k*dx,dx,1.,.002,1.,1.)-exact
        assert abs(b.real)<abs(a.real) and abs(b.imag)<abs(a.imag)


def test_continuation_state_four_dimensions():
    lo,hi=parameter_bounds('M4');theta=jnp.array([.8,-7.,.2,1.8])
    loss=lambda z:jnp.sum((z-jnp.array([1.,-6.,1.,1.]))**2)
    first=projected_adam_kernel(loss,theta,lo,hi,7,.05)
    second=projected_adam_kernel(loss,theta,lo,hi,13,.05,first['optimizer_state'])
    full=projected_adam_kernel(loss,theta,lo,hi,20,.05)
    for key in full['optimizer_state']:
        np.testing.assert_allclose(second['optimizer_state'][key],full['optimizer_state'][key],atol=1e-14)
