"""Scalar hyperdiffusion inference through the validated forward solver."""

import math

import jax
import jax.numpy as jnp

from src.solver import integrate


def nu4_from_log_parameter(q):
    """Map a finite log coefficient to a positive coefficient (within range)."""
    return jnp.exp(q)


def predict_observations(q, psi0, dt, num_steps, v, dx, observation_indices, sensor_indices=None):
    """Select full spatial fields at fixed integer trajectory indices.

    Indices must lie in [0, num_steps]. Keep indices, dt and num_steps fixed
    throughout fitting. Only selected states are returned, though integrate
    and reverse-mode differentiation can still require full-trajectory memory.
    sensor_indices optionally selects spatial columns; None preserves full fields.
    """
    trajectory = integrate(psi0, dt, num_steps, v, nu4_from_log_parameter(q), dx)
    selected = trajectory[observation_indices]
    return selected if sensor_indices is None else selected[:, sensor_indices]


def inverse_loss(q, psi0, dt, num_steps, v, dx, observation_indices, observations, sensor_indices=None):
    """Normalized field MSE, with a fixed 1e-12 denominator safeguard."""
    prediction = predict_observations(q, psi0, dt, num_steps, v, dx, observation_indices, sensor_indices)
    return jnp.mean((prediction - observations)**2) / (jnp.mean(observations**2) + 1e-12)


def optimize_log_parameter(loss_fn, nu4_init, nu4_bounds, iterations=800, learning_rate=0.05):
    """Projected scalar Adam; bounds must be chosen for stability beforehand.

    No truth is supplied. Bounds constrain the search interval, not the answer.
    Return initial and post-update loss/nu4 histories plus boundary-hit count.
    Adam uses beta1=.9, beta2=.999, epsilon=1e-8 and a fixed learning rate.
    """
    lower, upper = nu4_bounds
    if not (0 < lower < upper and lower <= nu4_init <= upper):
        raise ValueError("Require positive ordered bounds containing nu4_init")
    if iterations < 1 or learning_rate <= 0:
        raise ValueError("Require positive iterations and learning_rate")
    q0 = jnp.asarray(math.log(nu4_init))
    q_lower, q_upper = math.log(lower), math.log(upper)
    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def run(q):
        def update(state, iteration):
            q, m, second_moment = state
            loss, gradient = value_and_grad(q)
            m = 0.9 * m + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            m_hat = m / (1 - 0.9**iteration)
            second_hat = second_moment / (1 - 0.999**iteration)
            proposal = q - learning_rate * m_hat / (jnp.sqrt(second_hat) + 1e-8)
            next_q = jnp.clip(proposal, q_lower, q_upper)
            hit = (proposal < q_lower) | (proposal > q_upper)
            return (next_q, m, second_moment), (loss, jnp.exp(q), hit)

        (q_final, _, _), (losses, coefficients, hits) = jax.lax.scan(
            update, (q, jnp.zeros_like(q), jnp.zeros_like(q)),
            jnp.arange(1, iterations + 1),
        )
        return {
            "q": q_final,
            "nu4": jnp.exp(q_final),
            "loss_history": jnp.concatenate((losses, loss_fn(q_final)[None])),
            "nu4_history": jnp.concatenate((coefficients, jnp.exp(q_final)[None])),
            "boundary_hits": jnp.sum(hits),
        }

    result = run(q0)
    if not bool(jnp.all(jnp.isfinite(result["loss_history"]))) or not bool(jnp.all(jnp.isfinite(result["nu4_history"]))):
        raise FloatingPointError("Nonfinite optimizer history; check stability and inputs")
    return result
