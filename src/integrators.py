"""Differentiable time integration of the existing periodic FD operator.

All Fourier methods use the archived spatial symbols. Exponential propagation
is exact for this semi-discrete system, not for the continuum PDE. Step counts
and requested integer indices are static scheduling metadata under JIT.
"""
from functools import partial
import jax
import jax.numpy as jnp
from src.discrepancy import operator_symbols, discrepancy_symbol, rk4_observations


def fourier_angles(n):
    return 2*jnp.pi*jnp.fft.fftfreq(n)


def fourier_symbol(n, dx, v, nu4, c3, c6):
    return discrepancy_symbol(fourier_angles(n), dx, v, nu4, c3, c6)


def split_symbol(n, dx, v, nu4, c3, c6):
    d1,d4,d3,d6 = operator_symbols(fourier_angles(n), dx)
    return -v*d1+c3*v*dx**2*d3/6, -nu4*d4+c6*nu4*dx**2*d6/6


def fft_rhs(psi, v, nu4, c3, c6, dx):
    return jnp.fft.ifft(fourier_symbol(len(psi),dx,v,nu4,c3,c6)*jnp.fft.fft(psi)).real


def exponential_complex(psi0, times, v, nu4, c3, c6, dx):
    """Complex output for auditing discarded roundoff; times must be >= 0."""
    times=jnp.atleast_1d(times)
    lam=fourier_symbol(len(psi0),dx,v,nu4,c3,c6)
    return jnp.fft.ifft(jnp.exp(times[:,None]*lam)*jnp.fft.fft(psi0),axis=-1)


def exponential_propagate(psi0, times, v, nu4, c3, c6, dx):
    """Requested nonnegative times only; zero returns the input bit-for-bit."""
    times=jnp.atleast_1d(times)
    out=exponential_complex(psi0,times,v,nu4,c3,c6,dx).real
    return jnp.where(times[:,None]==0,psi0[None,:],out)


def exponential_final(psi0, time, v, nu4, c3, c6, dx):
    return exponential_propagate(psi0,jnp.atleast_1d(time),v,nu4,c3,c6,dx)[0]


def rk4_power_propagate(psi0, dt, time_indices, v, nu4, c3, c6, dx):
    """Unmodified Stage 9 rule, including its floating-point t=0 convention."""
    return rk4_observations((v,nu4,c3,c6),psi0,dt,dx,jnp.asarray(time_indices),jnp.arange(len(psi0)))


def cn_amplification(lam, dt):
    return (1+dt*lam/2)/(1-dt*lam/2)


def crank_nicolson_step_fft(psi, dt, v, nu4, c3, c6, dx):
    lam=fourier_symbol(len(psi),dx,v,nu4,c3,c6)
    return jnp.fft.ifft(cn_amplification(lam,dt)*jnp.fft.fft(psi)).real


def cn_power_propagate(psi0, dt, time_indices, v, nu4, c3, c6, dx):
    lam=fourier_symbol(len(psi0),dx,v,nu4,c3,c6)
    return jnp.fft.ifft(cn_amplification(lam,dt)[None,:]**jnp.asarray(time_indices)[:,None]*jnp.fft.fft(psi0),axis=-1).real


def cnab2_roots(explicit, stiff, dt):
    """Roots of A z² - B z + C, derived by dividing recurrence by u[n-1]."""
    explicit,stiff=jnp.asarray(explicit),jnp.asarray(stiff)
    a=1-dt*stiff/2
    b=1+dt*stiff/2+1.5*dt*explicit
    c=.5*dt*explicit
    disc=jnp.sqrt((b*b-4*a*c).astype(jnp.complex128))
    return jnp.stack(((b+disc)/(2*a),(b-disc)/(2*a)),axis=-1)


def _iterate(psi0, dt, num_steps, v, nu4, c3, c6, dx, method, time_indices):
    if num_steps < 0:
        raise ValueError('num_steps must be nonnegative')
    e,s=split_symbol(len(psi0),dx,v,nu4,c3,c6)
    cn=cn_amplification(e+s,dt)
    h0=jnp.fft.fft(psi0)
    def step(carry, index):
        prev,cur=carry
        if method=='cn':
            nxt=cn*cur
        else:
            ab=((1+dt*s/2)*cur+dt*e*(1.5*cur-.5*prev))/(1-dt*s/2)
            nxt=jnp.where(index==0,cn*cur,ab)
        return (cur,nxt),nxt
    if time_indices is None:
        _,hs=jax.lax.scan(step,(h0,h0),jnp.arange(num_steps))
        return jnp.concatenate((psi0[None],jnp.fft.ifft(hs,axis=-1).real))
    # Observation-only forward storage: two states plus requested outputs.
    # Reverse-mode scan may still save intermediate carry/tangents.
    indices=jnp.asarray(time_indices)
    out=jnp.where((indices==0)[:,None],h0[None],jnp.zeros((len(indices),len(psi0)),dtype=h0.dtype))
    def selected(carry,index):
        states,values=carry
        states,h=step(states,index)
        values=jnp.where((indices==index+1)[:,None],h[None],values)
        return (states,values),None
    (_,out),_=jax.lax.scan(selected,((h0,h0),out),jnp.arange(num_steps))
    fields=jnp.fft.ifft(out,axis=-1).real
    return jnp.where((indices==0)[:,None],psi0[None],fields)


@partial(jax.jit,static_argnames=('num_steps',))
def crank_nicolson_integrate(psi0,dt,num_steps,v,nu4,c3,c6,dx,time_indices=None):
    """Actual CN recurrence; full trajectory or selected integer step times."""
    return _iterate(psi0,dt,num_steps,v,nu4,c3,c6,dx,'cn',time_indices)


@partial(jax.jit,static_argnames=('num_steps',))
def imex_cnab2_integrate(psi0,dt,num_steps,v,nu4,c3,c6,dx,time_indices=None):
    """Actual CNAB2 recurrence, with a full-operator CN startup step."""
    return _iterate(psi0,dt,num_steps,v,nu4,c3,c6,dx,'cnab2',time_indices)
