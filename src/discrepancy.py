"""Dimensionless modified-equation discrepancy, separate from physical v, nu4.

The production observation evaluator diagonalizes the *same* periodic linear
RK4 update. It is not an exact-continuum solver: its amplification is R(dt*lambda)
raised to the integer step index. The stencil/scan integrator remains the public
trajectory implementation and provides an independent equivalence check.
"""
from functools import partial
import itertools
import jax
import jax.numpy as jnp
import numpy as np
from src.operators import d_dx, d4_dx4, d3_dx3, d6_dx6


def discrepancy_rhs(psi, v, nu4, c3, c6, dx):
    return (-v*d_dx(psi, dx)-nu4*d4_dx4(psi, dx)
            + c3*v*dx**2/6*d3_dx3(psi, dx)
            + c6*nu4*dx**2/6*d6_dx6(psi, dx))


def discrepancy_rk4_step(psi, dt, v, nu4, c3, c6, dx):
    f = lambda y: discrepancy_rhs(y, v, nu4, c3, c6, dx)
    k1 = f(psi); k2 = f(psi+dt*k1/2)
    k3 = f(psi+dt*k2/2); k4 = f(psi+dt*k3)
    return psi+dt*(k1+2*k2+2*k3+k4)/6


@partial(jax.jit, static_argnames=('num_steps',))
def discrepancy_integrate(psi0, dt, num_steps, v, nu4, c3, c6, dx):
    if num_steps < 0:
        raise ValueError('num_steps must be nonnegative')
    def step(y, _):
        z = discrepancy_rk4_step(y, dt, v, nu4, c3, c6, dx)
        return z, z
    _, ys = jax.lax.scan(step, psi0, None, length=num_steps)
    return jnp.concatenate((psi0[None], ys), axis=0)


def operator_symbols(angle, dx):
    s = jnp.sin(angle/2)
    return (1j*jnp.sin(angle)/dx, 16*s**4/dx**4,
            -4j*jnp.sin(angle)*s**2/dx**3, -64*s**6/dx**6)


def discrepancy_symbol(angle, dx, v, nu4, c3, c6):
    d1,d4,d3,d6 = operator_symbols(angle, dx)
    return -v*d1-nu4*d4+c3*v*dx**2/6*d3+c6*nu4*dx**2/6*d6


def rk4_amplification(z):
    return 1+z+z*z/2+z*z*z/6+z*z*z*z/24


def unpack_parameters(theta, model='M4', calibrated=(1.,1.)):
    v,nu = theta[0],jnp.exp(theta[1])
    c3 = theta[2] if model in ('M2','M4') else (1. if model=='M1' else calibrated[0] if model=='M5' else 0.)
    c6 = theta[3] if model=='M4' else theta[2] if model=='M3' else (1. if model=='M1' else calibrated[1] if model=='M5' else 0.)
    return v,nu,c3,c6


def parameter_bounds(model):
    n = {'M0':0,'M1':0,'M2':1,'M3':1,'M4':2,'M5':0}[model]
    return jnp.array([.5,np.log(1e-4)]+[0.]*n), jnp.array([1.5,np.log(.02)]+[2.]*n)


def rk4_observations(parameters, psi0, dt, dx, time_indices, sensors):
    """Algebraic diagonalization of fixed-step stencil RK4; integer powers only."""
    n = psi0.shape[0]
    angles = 2*jnp.pi*jnp.fft.fftfreq(n)
    lam = discrepancy_symbol(angles, dx, *parameters)
    amp = rk4_amplification(dt*lam)
    spectrum = jnp.fft.fft(psi0)
    fields = jnp.fft.ifft(amp[None,:]**time_indices[:,None]*spectrum[None,:], axis=1).real
    return fields[:,sensors]


def discrepancy_loss(theta, psi0, dt, dx, time_indices, observations,
                     sensors, model='M4', calibrated=(1.,1.)):
    pred = rk4_observations(unpack_parameters(theta,model,calibrated), psi0,dt,dx,time_indices,sensors)
    return jnp.mean((pred-observations)**2)/(jnp.mean(observations**2)+1e-12)


def projected_residual(theta, gradient, lower, upper):
    inactive = ((theta<=lower+1e-8)&(gradient>=0))|((theta>=upper-1e-8)&(gradient<=0))
    return jnp.linalg.norm(jnp.where(inactive,0.,gradient))


def stability_policy(n):
    """Dense box/angle numerical estimate, including every actual grid mode.

    No claim of an analytic enclosure: 5 samples per coordinate plus 8193 angles.
    dt is at most half the sampled limit and fixed independently of fit values.
    """
    dx=2*np.pi/n
    angles=np.unique(np.r_[np.linspace(0,np.pi,8193),2*np.pi*np.arange(n//2+1)/n])
    s=np.sin(angles/2)
    # Real/imaginary extrema are monotone in positive bounds, but sample interiors
    # as RK4's stability region need not be a rectangular set.
    worst=np.inf; limiting=None
    for v,nu,c3,c6 in itertools.product(np.linspace(.5,1.5,5),np.geomspace(1e-4,.02,5),np.linspace(0,2,5),np.linspace(0,2,5)):
        lam=-1j*v*np.sin(angles)/dx*(1+2*c3*s*s/3)-nu*16*s**4/dx**4*(1+2*c6*s*s/3)
        lo,hi=0.,1.
        for _ in range(45):
            mid=(lo+hi)/2
            z=mid*lam
            if np.max(np.abs(1+z+z*z/2+z**3/6+z**4/24))<=1+1e-12: lo=mid
            else: hi=mid
        if lo<worst: worst=lo; limiting=[v,nu,c3,c6]
    steps=256
    while 2/steps > .5*worst: steps*=2
    return dict(N=n,dx=dx,dt_max=worst,dt=2/steps,num_steps=steps,safety_fraction=(2/steps)/worst,limiting_parameters=limiting,box_samples=625,angle_samples=len(angles))
