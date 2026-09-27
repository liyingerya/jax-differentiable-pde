import jax
import jax.numpy as jnp
import numpy as np
import pytest
from src.integrators import *
from src.discrepancy import discrepancy_rhs, discrepancy_integrate

N=16
DX=2*np.pi/N
X=jnp.arange(N)*DX
Y=.3+jnp.sin(X)+.2*jnp.cos(2*X)
P=(1.,.002,1.,1.)

@pytest.mark.parametrize('c',[(0.,0.),(1.,1.),(1.0562544646126,1.0618708313445455)])
def test_symbol_stencil(c):
    np.testing.assert_allclose(fft_rhs(Y,1.,.002,*c,DX),discrepancy_rhs(Y,1.,.002,*c,DX),atol=2e-14)

def test_zero_exact():
    np.testing.assert_array_equal(exponential_propagate(Y,jnp.array([0.]),*P,DX)[0],Y)

def test_shape():
    assert exponential_propagate(Y,jnp.array([0.,.13,.91]),*P,DX).shape==(3,N)

def test_semigroup():
    a=exponential_final(Y,.73,*P,DX)
    b=exponential_final(exponential_final(Y,.21,*P,DX),.52,*P,DX)
    np.testing.assert_allclose(a,b,atol=1e-14)

def test_mean():
    np.testing.assert_allclose(exponential_propagate(Y,jnp.array([0.,.3,2.]),*P,DX).mean(1),Y.mean(),atol=1e-14)

def test_diffusion():
    z=exponential_propagate(Y,jnp.linspace(0,2,9),0.,.002,0.,1.,DX)
    assert np.all(np.diff(np.sum(np.asarray(z)**2,axis=1))<=0)

def test_imaginary():
    assert float(jnp.max(jnp.abs(exponential_complex(Y,jnp.array([0.,2.]),*P,DX).imag)))<1e-14

@pytest.mark.parametrize('i',range(4))
def test_parameter_gradient(i):
    theta=jnp.array([.9,np.log(.003),.7,1.2])
    f=lambda z:jnp.mean((exponential_final(Y,.7,z[0],jnp.exp(z[1]),z[2],z[3],DX)-Y)**2)
    d=jnp.eye(4)[i]*1e-5
    np.testing.assert_allclose(jax.grad(f)(theta)[i],(f(theta+d)-f(theta-d))/2e-5,rtol=2e-6,atol=1e-10)

@pytest.mark.parametrize('method',['exact','imex'])
def test_initial_derivative(method):
    f=(lambda y:exponential_final(y,.5,*P,DX)) if method=='exact' else (lambda y:imex_cnab2_integrate(y,.01,50,*P,DX)[-1])
    d=jnp.cos(3*X)
    _,ad=jax.jvp(f,(Y,),(d,))
    np.testing.assert_allclose(ad,(f(Y+1e-5*d)-f(Y-1e-5*d))/2e-5,atol=1e-9)

def test_rk4_scan():
    scan=discrepancy_integrate(Y,.005,100,*P,DX)
    power=rk4_power_propagate(Y,.005,jnp.arange(101),*P,DX)
    np.testing.assert_allclose(scan,power,atol=3e-14)

@pytest.mark.parametrize('method',['rk4','cn','imex'])
def test_convergence(method):
    ref=exponential_final(Y,1.,*P,DX)
    errors=[]
    for steps in [32,64,128]:
        if method=='rk4':z=rk4_power_propagate(Y,1/steps,jnp.array([steps]),*P,DX)[0]
        elif method=='cn':z=crank_nicolson_integrate(Y,1/steps,steps,*P,DX)[-1]
        else:z=imex_cnab2_integrate(Y,1/steps,steps,*P,DX)[-1]
        errors.append(float(jnp.linalg.norm(z-ref)))
    order=np.log2(errors[-2]/errors[-1])
    assert (3.8<order<4.2) if method=='rk4' else (1.9<order<2.1)

def test_startup_shape_and_selected():
    out=imex_cnab2_integrate(Y,.01,20,*P,DX)
    assert out.shape==(21,N)
    np.testing.assert_allclose(out[1],crank_nicolson_step_fft(Y,.01,*P,DX),atol=1e-14)
    np.testing.assert_allclose(out[[0,3,20],:],imex_cnab2_integrate(Y,.01,20,*P,DX,jnp.array([0,3,20])),atol=1e-14)

def test_cn_power_iterative():
    out=crank_nicolson_integrate(Y,.02,20,*P,DX)
    np.testing.assert_allclose(out,cn_power_propagate(Y,.02,jnp.arange(21),*P,DX),atol=1e-14)

def test_roots_recurrence():
    e=jnp.array(2j);s=jnp.array(-3.);dt=.1
    for z in cnab2_roots(e,s,dt):
        prev=1.+0j;cur=z
        for _ in range(5):
            nxt=((1+dt*s/2)*cur+dt*e*(1.5*cur-.5*prev))/(1-dt*s/2)
            np.testing.assert_allclose(nxt,z*cur,atol=1e-14)
            prev,cur=cur,nxt

@pytest.mark.parametrize('c6',[0.,1.,2.])
def test_split_dissipation(c6):
    for nu in [1e-4,.002,.02]:
        e,s=split_symbol(N,DX,1.5,nu,2.,c6)
        assert float(jnp.max(s.real))<=0
        np.testing.assert_array_equal(e.real,jnp.zeros(N))

def test_original_fd_symbol():
    t=.8;k=2
    expected=jnp.exp(-.002*16*jnp.sin(k*DX/2)**4/DX**4*t)*jnp.sin(k*X-jnp.sin(k*DX)/DX*t)
    np.testing.assert_allclose(exponential_final(jnp.sin(k*X),t,1.,.002,0.,0.,DX),expected,atol=1e-14)

def test_differs_continuum():
    exact=exponential_final(jnp.sin(2*X),1.,1.,.002,0.,0.,DX)
    continuum=jnp.exp(-.002*16)*jnp.sin(2*(X-1.))
    assert float(jnp.linalg.norm(exact-continuum))>1e-2

def test_invalid_steps():
    with pytest.raises(ValueError):imex_cnab2_integrate(Y,.1,-1,*P,DX)
