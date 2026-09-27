"""Numerical verification of the periodic spatial stencils."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from src.operators import d_dx, d4_dx4


@pytest.mark.parametrize("operator", [d_dx, d4_dx4])
def test_output_shape(operator):
    f = jnp.arange(32, dtype=jnp.float64)
    assert operator(f, 0.1).shape == f.shape


@pytest.mark.parametrize("operator", [d_dx, d4_dx4])
def test_constant_derivative(operator):
    f = jnp.full(128, 1.3, dtype=jnp.float64)
    result = operator(f, 2 * jnp.pi / f.size)
    # The fourth stencil amplifies cancellation roundoff by dx**-4.
    assert float(jnp.max(jnp.abs(result))) < 1e-8


@pytest.mark.parametrize("operator,tolerance", [(d_dx, 0.003), (d4_dx4, 0.1)])
def test_sine_derivative(operator, tolerance):
    n = 256
    x = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    exact = 3 * jnp.cos(3 * x) if operator is d_dx else 81 * jnp.sin(3 * x)
    error = jnp.max(jnp.abs(operator(jnp.sin(3 * x), 2 * jnp.pi / n) - exact))
    # Tolerances allow the expected O(dx**2) truncation error, including seams.
    assert float(error) < tolerance


@pytest.mark.parametrize("operator", [d_dx, d4_dx4])
def test_refinement_decreases_error(operator):
    errors = []
    for n in (32, 64, 128, 256, 512):
        x = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
        f = jnp.sin(3 * x)
        exact = 3 * jnp.cos(3 * x) if operator is d_dx else 81 * f
        errors.append(float(jnp.max(jnp.abs(operator(f, 2 * jnp.pi / n) - exact))))
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
