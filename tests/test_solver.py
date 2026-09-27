"""Focused checks of RK4, trajectory diagnostics, and differentiation."""

import jax
import jax.numpy as jnp
import pytest

from src.solver import integrate, rhs, rk4_step
from experiments.stage2_time_integration_validation import fourier_exact, stability_limit


@pytest.fixture
def mode():
    n = 64
    dx = 2 * jnp.pi / n
    x = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    return x, jnp.sin(3 * x), dx


def test_rhs_shape(mode):
    _, f, dx = mode
    assert rhs(f, 1.0, 0.0005, dx).shape == f.shape


def test_rhs_constant(mode):
    _, f, dx = mode
    assert float(jnp.max(jnp.abs(rhs(jnp.full_like(f, 1.3), 1.0, 0.0005, dx)))) < 1e-10


def test_step_shape(mode):
    _, f, dx = mode
    assert rk4_step(f, 0.02, 1.0, 0.0005, dx).shape == f.shape


@pytest.mark.parametrize('steps', [0, 10])
def test_trajectory_shape_and_initial(mode, steps):
    _, f, dx = mode
    trajectory = integrate(f, 0.02, steps, 1.0, 0.0005, dx)
    assert trajectory.shape == (steps + 1, f.size)
    assert bool(jnp.array_equal(trajectory[0], f))


def test_advection_direction_and_amplitude(mode):
    x, f, dx = mode
    final = integrate(f, 0.02, 25, 1.0, 0.0, dx)[-1]
    right = fourier_exact(x, 0.5, 3, 1.0, 0.0, dx)
    wrong = fourier_exact(x, 0.5, 3, -1.0, 0.0, dx)
    assert float(jnp.max(jnp.abs(final - right))) < 2e-6
    assert float(jnp.max(jnp.abs(final - wrong))) > 1.0
    assert abs(float(jnp.mean(final**2)) - 0.5) < 1e-6
    # Roll equivariance checks periodic wraparound, not just an interior peak.
    shifted = integrate(jnp.roll(f, 9), 0.02, 25, 1.0, 0.0, dx)[-1]
    assert float(jnp.max(jnp.abs(shifted - jnp.roll(final, 9)))) < 1e-12


def test_hyperdiffusion_amplitude_and_energy(mode):
    x, f, dx = mode
    trajectory = integrate(f, 0.02, 100, 0.0, 0.0005, dx)
    expected = fourier_exact(x, 2.0, 3, 0.0, 0.0005, dx)
    assert float(jnp.max(jnp.abs(trajectory[-1] - expected))) < 1e-10
    amplitude = float(2 * jnp.mean(trajectory[-1] * f))
    assert 0 < amplitude < 0.95
    energy = jnp.mean(trajectory**2, axis=1)
    assert bool(jnp.all(jnp.diff(energy) <= 1e-12))
    assert float(energy[-1]) < float(energy[0])


def test_mean_conservation(mode):
    _, f, dx = mode
    trajectory = integrate(f + 0.7, 0.02, 100, 1.0, 0.0005, dx)
    assert float(jnp.max(jnp.abs(jnp.mean(trajectory, axis=1) - 0.7))) < 1e-11


def test_temporal_convergence(mode):
    x, f, dx = mode
    expected = fourier_exact(x, 2.0, 3, 1.0, 0.0005, dx)
    errors = [float(jnp.max(jnp.abs(integrate(f, 2.0 / steps, steps, 1.0, 0.0005, dx)[-1] - expected))) for steps in (100, 200, 400, 800)]
    orders = jnp.log2(jnp.array(errors[:-1]) / jnp.array(errors[1:]))
    assert bool(jnp.all((orders > 3.8) & (orders < 4.2)))


def test_stability_diagnostic(mode):
    _, _, dx = mode
    limit, amplification = stability_limit(1.0, 0.0005, dx)
    assert 0.02 < 0.7 * limit
    assert float(amplification(0.02)) <= 1 + 1e-12
    assert float(amplification(1.01 * limit)) > 1.001


def test_diffusion_gradient(mode):
    _, f, dx = mode

    def objective(nu4):
        return jnp.mean(integrate(f, 0.02, 100, 1.0, nu4, dx)[-1]**2)

    nu4, eps = 0.0005, 1e-7
    grad = jax.grad(objective)(nu4)
    fd = (objective(nu4 + eps) - objective(nu4 - eps)) / (2 * eps)
    assert bool(jnp.isfinite(grad))
    assert float(grad) < 0
    assert float(jnp.abs(grad - fd) / jnp.abs(fd)) < 1e-6


def test_initial_state_and_velocity_gradients(mode):
    x, f, dx = mode
    weights = jnp.cos(3 * x)

    def objective(initial, speed):
        return jnp.mean(integrate(initial, 0.02, 25, speed, 0.0005, dx)[-1] * weights)

    grad_initial, grad_v = jax.grad(objective, argnums=(0, 1))(f, 1.0)
    direction = jnp.sin(2 * x) + weights
    eps = 1e-5
    fd_initial = (objective(f + eps * direction, 1.0) - objective(f - eps * direction, 1.0)) / (2 * eps)
    fd_v = (objective(f, 1.0 + eps) - objective(f, 1.0 - eps)) / (2 * eps)
    assert bool(jnp.all(jnp.isfinite(grad_initial)))
    assert bool(jnp.isfinite(grad_v))
    assert float(jnp.abs(jnp.vdot(grad_initial, direction) - fd_initial)) < 1e-8
    assert float(jnp.abs(grad_v - fd_v)) < 1e-8
