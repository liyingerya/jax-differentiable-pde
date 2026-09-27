"""Fast invariants; the full refinement experiment is not run by pytest."""

import jax.numpy as jnp
import pytest

from experiments.stage3_parameter_inference import continuum_observations
from experiments.stage4_refinement_study import bias_metrics, modified_wavenumbers, run_fit, select_timestep


def test_continuum_shape_and_initial():
    x = jnp.linspace(0, 2*jnp.pi, 24, endpoint=False)
    fields = continuum_observations(x, jnp.array([0., 0.25, 2.]), 1., 0.002)
    assert fields.shape == (3,24)
    initial = jnp.sin(x)+0.5*jnp.sin(2*x)+0.25*jnp.cos(3*x)
    assert bool(jnp.allclose(fields[0], initial, atol=1e-14, rtol=0))


def test_timestep_and_observation_alignment():
    dt, steps, indices = select_timestep(0.0129372079)
    assert dt <= 0.5*0.0129372079
    assert steps % 8 == 0
    assert steps*dt == 2.0
    assert bool(jnp.allclose(indices*dt, jnp.arange(9)/4, atol=1e-15, rtol=0))
    assert int(indices[-1]) == steps
    with pytest.raises(ValueError):
        select_timestep(0.01, safety=1.1)


@pytest.mark.parametrize('coefficient,sign', [(0.003,1), (0.001,-1)])
def test_bias_metrics(coefficient, sign):
    metrics = bias_metrics(coefficient)
    assert metrics['signed_bias'] == pytest.approx(sign*0.001)
    assert metrics['absolute_error'] == pytest.approx(0.001)
    assert metrics['relative_bias'] == pytest.approx(sign*0.5)
    assert metrics['relative_error'] == pytest.approx(0.5)


def test_pure_diffusion_has_no_phase_translation():
    x = jnp.linspace(0, 2*jnp.pi, 32, endpoint=False)
    final = continuum_observations(x, jnp.array([2.]), 0., 0.002)[0]
    for k, basis, coefficient in ((1,jnp.sin,1.), (2,jnp.sin,0.5), (3,jnp.cos,0.25)):
        amplitude = 2*jnp.mean(final*basis(k*x))
        assert float(amplitude) == pytest.approx(coefficient*float(jnp.exp(-0.002*k**4*2)), abs=1e-13)
        opposite = jnp.cos if basis is jnp.sin else jnp.sin
        assert abs(float(2*jnp.mean(final*opposite(k*x)))) < 1e-13


def test_modified_wavenumbers_converge():
    coarse, fine = modified_wavenumbers(16), modified_wavenumbers(48)
    exact = jnp.array(coarse['k4_continuum'])
    assert bool(jnp.all(jnp.abs(jnp.array(fine['k4_discrete'])-exact) < jnp.abs(jnp.array(coarse['k4_discrete'])-exact)))
    assert bool(jnp.all(jnp.array(fine['effective_nu4']) > 0.002))


def test_two_grid_recovery_smoke():
    for n in (16,24):
        result = run_fit(n, iterations=20)
        assert bool(jnp.isfinite(result['recovered_nu4']))
        assert bool(jnp.isfinite(result['final_loss']))
        assert result['range_max_amplification'] <= 1+1e-12
        assert result['safety_fraction'] <= 0.5
        assert 1e-4 <= result['recovered_nu4'] <= 0.02
