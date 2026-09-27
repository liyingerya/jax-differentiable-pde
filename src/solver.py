"""Differentiable explicit integration of the periodic hyperdiffusion PDE."""

from functools import partial

import jax
import jax.numpy as jnp

from src.operators import d_dx, d4_dx4


def rhs(psi, v, nu4, dx):
    """Evaluate psi_t = -v psi_x - nu4 psi_xxxx on a uniform periodic grid."""
    return -v * d_dx(psi, dx) - nu4 * d4_dx4(psi, dx)


def rk4_step(psi, dt, v, nu4, dx):
    """Advance one classical RK4 step; the caller chooses a stable dt."""
    k1 = rhs(psi, v, nu4, dx)
    k2 = rhs(psi + dt * k1 / 2, v, nu4, dx)
    k3 = rhs(psi + dt * k2 / 2, v, nu4, dx)
    k4 = rhs(psi + dt * k3, v, nu4, dx)
    return psi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


@partial(jax.jit, static_argnames=("num_steps",))
def integrate(psi0, dt, num_steps, v, nu4, dx):
    """Return (num_steps + 1, N) states, including psi0, using lax.scan.

    num_steps must be a nonnegative static integer. dt and dx are positive;
    stability is the caller's responsibility. Gradients through psi0, v and
    nu4 keep dt and num_steps fixed. The full trajectory requires O(steps*N)
    storage.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")

    def step(psi, _):
        next_psi = rk4_step(psi, dt, v, nu4, dx)
        return next_psi, next_psi

    _, states = jax.lax.scan(step, psi0, xs=None, length=num_steps)
    return jnp.concatenate((psi0[jnp.newaxis, :], states), axis=0)
