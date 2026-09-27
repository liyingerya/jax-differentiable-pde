"""Measure max errors and convergence for sin(3x) on [0, 2 pi)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from src.operators import d_dx, d4_dx4


def main():
    length = 2 * jnp.pi
    previous = None
    print(f"{'N':>5} {'dx':>13} {'d1 max error':>16} {'d4 max error':>16} {'p(d1)':>9} {'p(d4)':>9}")
    print("-" * 74)
    for n in (32, 64, 128, 256, 512):
        dx = length / n
        x = jnp.linspace(0.0, length, n, endpoint=False)
        f = jnp.sin(3 * x)
        error1 = jnp.max(jnp.abs(d_dx(f, dx) - 3 * jnp.cos(3 * x)))
        error4 = jnp.max(jnp.abs(d4_dx4(f, dx) - 81 * f))
        if previous is None:
            orders = f"{'—':>9} {'—':>9}"
        else:
            p1 = jnp.log(previous[0] / error1) / jnp.log(2.0)
            p4 = jnp.log(previous[1] / error4) / jnp.log(2.0)
            orders = f"{float(p1):9.5f} {float(p4):9.5f}"
        print(f"{n:5d} {float(dx):13.6e} {float(error1):16.8e} {float(error4):16.8e} {orders}")
        previous = error1, error4


if __name__ == "__main__":
    main()
