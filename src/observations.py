"""Reproducible observation degradation with an exactly known initial field."""

import jax
import jax.numpy as jnp


def sensor_layout(num_points, count, seed):
    """Choose sorted distinct sensors without replacement using a fixed seed."""
    if not 1 <= count <= num_points:
        raise ValueError("sensor count must be between 1 and num_points")
    return jnp.sort(jax.random.choice(jax.random.PRNGKey(seed), num_points,
                                      shape=(count,), replace=False))


def add_measurement_noise(clean, observation_time_indices, sigma_rel, seed):
    """Add Gaussian noise only at positive times, scaled by their clean RMS.

    clean contains only observed entries. At least one time must be positive.
    Return noisy observations and reproducible realized-noise diagnostics.
    """
    if sigma_rel < 0:
        raise ValueError("sigma_rel must be nonnegative")
    positive = jnp.asarray(observation_time_indices) > 0
    count = jnp.sum(positive) * clean.shape[1]
    if int(count) == 0:
        raise ValueError("Require at least one positive-time observation")
    mask = positive[:, None]
    rms = jnp.sqrt(jnp.sum(jnp.where(mask, clean**2, 0.0)) / count)
    sigma_abs = sigma_rel * rms
    noise = jnp.where(mask, sigma_abs * jax.random.normal(
        jax.random.PRNGKey(seed), clean.shape, dtype=clean.dtype), 0.0)
    noisy = clean if sigma_rel == 0 else clean + noise
    return noisy, {"clean_positive_time_rms": float(rms),
                   "sigma_abs": float(sigma_abs),
                   "realized_noise_rms": float(jnp.sqrt(jnp.sum(noise**2)/count))}
