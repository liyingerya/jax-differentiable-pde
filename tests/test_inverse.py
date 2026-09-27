"""Fast inverse-loss and short-optimization checks; full recovery is an experiment."""

import jax
import jax.numpy as jnp
import pytest

from src.inverse import inverse_loss, nu4_from_log_parameter, optimize_log_parameter, predict_observations


@pytest.fixture(scope="module")
def inverse_case():
    x = jnp.linspace(0.0, 2*jnp.pi, 32, endpoint=False)
    psi0 = jnp.sin(x) + 0.5*jnp.sin(2*x) + 0.25*jnp.cos(3*x)
    args = (psi0, 1/128, 128, 1.0, 2*jnp.pi/32, jnp.array([0, 32, 64, 96, 128]))
    observations = predict_observations(jnp.log(0.002), *args)
    loss = jax.jit(lambda q: inverse_loss(q, *args, observations))
    return args, observations, loss


def test_positive_transform_and_roundtrip():
    coefficients = jnp.array([1e-4, 0.002, 0.02])
    recovered = nu4_from_log_parameter(jnp.log(coefficients))
    assert bool(jnp.all(recovered > 0))
    assert bool(jnp.allclose(recovered, coefficients, rtol=1e-13, atol=0))


def test_observation_shape_and_selection(inverse_case):
    args, observations, _ = inverse_case
    assert observations.shape == (5, 32)
    assert bool(jnp.array_equal(observations[0], args[0]))
    assert float(jnp.linalg.norm(observations[-1] - observations[0])) > 0.1


def test_loss_scalar_finite_and_truth_minimum(inverse_case):
    _, _, loss = inverse_case
    true_loss = loss(jnp.log(0.002))
    wrong_loss = loss(jnp.log(0.008))
    assert true_loss.shape == ()
    assert bool(jnp.isfinite(true_loss) & jnp.isfinite(wrong_loss))
    assert float(true_loss) < 1e-20
    assert float(wrong_loss) > float(true_loss) + 1e-5


@pytest.mark.parametrize("candidate", [0.001, 0.005])
def test_inverse_gradient(inverse_case, candidate):
    _, _, loss = inverse_case
    q, eps = jnp.log(candidate), 1e-5
    gradient = jax.grad(loss)(q)
    finite_difference = (loss(q + eps) - loss(q - eps)) / (2*eps)
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient - finite_difference)/jnp.abs(finite_difference)) < 1e-5


def test_short_recovery(inverse_case):
    _, _, loss = inverse_case
    initial, truth = 0.0005, 0.002
    result = optimize_log_parameter(loss, initial, (1e-4, 0.02), iterations=60)
    assert result["loss_history"].shape == (61,)
    assert result["nu4_history"].shape == (61,)
    assert float(result["loss_history"][-1]) < float(result["loss_history"][0]) * 0.1
    assert abs(float(result["nu4"])-truth) < abs(initial-truth) * 0.2
    assert bool(jnp.all((result["nu4_history"] >= 1e-4) & (result["nu4_history"] <= 0.02)))


def test_optimizer_bounds_are_enforced():
    result = optimize_log_parameter(lambda q: -q, 0.01, (1e-4, 0.02), iterations=30)
    assert int(result["boundary_hits"]) > 0
    assert float(result["nu4"]) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        optimize_log_parameter(lambda q: q*q, 0.03, (1e-4, 0.02))
