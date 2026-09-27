"""Centered, second-order finite differences on a periodic 1D grid."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def d_dx(f, dx):
    """Return the first derivative of a 1D array with uniform spacing dx."""
    return (jnp.roll(f, -1) - jnp.roll(f, 1)) / (2 * dx)


def d4_dx4(f, dx):
    """Return the fourth derivative of a periodic 1D array."""
    return (
        jnp.roll(f, 2)
        - 4 * jnp.roll(f, 1)
        + 6 * f
        - 4 * jnp.roll(f, -1)
        + jnp.roll(f, -2)
    ) / dx**4


def d3_dx3(f, dx):
    """Centered second-order periodic third derivative (negative on sin)."""
    return (jnp.roll(f, -2) - 2*jnp.roll(f, -1)
            + 2*jnp.roll(f, 1) - jnp.roll(f, 2))/(2*dx**3)


def d6_dx6(f, dx):
    """Centered second-order periodic sixth derivative."""
    return (jnp.roll(f, 3) - 6*jnp.roll(f, 2) + 15*jnp.roll(f, 1)
            - 20*f + 15*jnp.roll(f, -1) - 6*jnp.roll(f, -2)
            + jnp.roll(f, -3))/dx**6
