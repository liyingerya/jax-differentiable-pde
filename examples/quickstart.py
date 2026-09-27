"""Run from the repository root: python -m examples.quickstart."""
import time
import sys
from pathlib import Path

# Support direct-file execution as well as python -m examples.quickstart.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
from src.integrators import exponential_propagate


def main():
    n=32;dx=2*jnp.pi/n;x=jnp.arange(n)*dx
    psi0=jnp.sin(x)+.5*jnp.sin(2*x)+.25*jnp.cos(3*x)
    times=jnp.array([0.,.5,1.,2.]);sensors=jnp.arange(0,n,4)
    truth=jnp.array([1.,jnp.log(.002)])
    def predict(theta):
        return exponential_propagate(psi0,times,theta[0],jnp.exp(theta[1]),1.,1.,dx)[:,sensors]
    observations=predict(truth)  # Clean matched synthetic observations.
    def loss(theta):
        return jnp.mean((predict(theta)-observations)**2)/(jnp.mean(observations**2)+1e-12)
    evaluate=jax.jit(jax.value_and_grad(loss));guess=jnp.array([.9,jnp.log(.0015)])
    value,gradient=evaluate(guess);jax.block_until_ready((value,gradient))
    start=time.perf_counter();jax.block_until_ready(evaluate(guess));elapsed=time.perf_counter()-start
    print('Observation shape:',observations.shape)
    print('NMSE:',float(value));print('Gradient in (v, log nu4):',gradient)
    print(f'Warmed loss + gradient: {elapsed:.6f} s')
    assert jnp.all(jnp.isfinite(gradient)) and value>0

if __name__=='__main__':main()
