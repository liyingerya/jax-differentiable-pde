"""Fast observation-selection and fixed-noise inverse checks."""

import jax
import jax.numpy as jnp
import pytest

from src.inverse import inverse_loss, optimize_log_parameter, predict_observations
from src.observations import add_measurement_noise, sensor_layout


@pytest.fixture(scope='module')
def setup():
    x = jnp.linspace(0,2*jnp.pi,32,endpoint=False)
    psi0 = jnp.sin(x)+0.5*jnp.sin(2*x)+0.25*jnp.cos(3*x)
    args = (psi0,2/256,256,1.0,2*jnp.pi/32,jnp.array([0,128,256]))
    clean = predict_observations(jnp.log(0.002),*args)
    return args, clean


def test_zero_noise_exact(setup):
    args, clean = setup
    noisy, diagnostics = add_measurement_noise(clean,args[-1],0.,100)
    assert bool(jnp.array_equal(noisy,clean))
    assert diagnostics['realized_noise_rms'] == 0


def test_noise_reproducibility_and_initial_state(setup):
    args, clean = setup
    a,_ = add_measurement_noise(clean,args[-1],0.02,100)
    b,_ = add_measurement_noise(clean,args[-1],0.02,100)
    c,_ = add_measurement_noise(clean,args[-1],0.02,101)
    assert bool(jnp.array_equal(a,b))
    assert not bool(jnp.array_equal(a,c))
    assert bool(jnp.array_equal(a[0],clean[0]))
    assert bool(jnp.array_equal(c[0],clean[0]))


def test_noise_scale_uses_positive_observed_entries(setup):
    args, clean = setup
    sensors = jnp.array([1,4,7,12])
    data = clean[:,sensors]
    _, diagnostics = add_measurement_noise(data,args[-1],0.02,100)
    assert diagnostics['sigma_abs'] == pytest.approx(0.02*float(jnp.sqrt(jnp.mean(data[1:]**2))))
    with pytest.raises(ValueError):
        add_measurement_noise(data[:1],jnp.array([0]),0.02,100)


def test_time_and_sensor_shapes(setup):
    args, clean = setup
    assert clean.shape == (3,32)
    selected = predict_observations(jnp.log(0.002),*args,jnp.array([1,3,7,18]))
    assert selected.shape == (3,4)
    assert bool(jnp.array_equal(selected,clean[:,jnp.array([1,3,7,18])]))


def test_full_selection_matches_old_loss(setup):
    args, _ = setup
    args = (*args[:-1], jnp.arange(0,257,32))
    clean = predict_observations(jnp.log(0.002),*args)
    q = jnp.log(0.003)
    full_loss = inverse_loss(q,*args,clean)
    selected_loss = inverse_loss(q,*args,clean,jnp.arange(32))
    assert float(selected_loss) == pytest.approx(float(full_loss),rel=1e-14)


def test_masked_loss_ignores_unobserved_entries(setup):
    args, clean = setup
    sensors = jnp.array([2,6,15,23])
    candidate = jnp.log(0.003)
    perturbed = clean.at[:,0].add(100.)
    original_loss = inverse_loss(candidate,*args,clean[:,sensors],sensors)
    changed_loss = inverse_loss(candidate,*args,perturbed[:,sensors],sensors)
    assert float(original_loss) == float(changed_loss)
    prediction = predict_observations(candidate,*args,sensors)
    expected = jnp.mean((prediction-clean[:,sensors])**2)/(jnp.mean(clean[:,sensors]**2)+1e-12)
    assert float(original_loss) == pytest.approx(float(expected),rel=1e-14)


@pytest.mark.parametrize('count',[4,8,16,32])
def test_sensor_layout_reproducibility(count):
    a = sensor_layout(32,count,200)
    assert bool(jnp.array_equal(a,sensor_layout(32,count,200)))
    assert len(set(a.tolist())) == count
    assert bool(jnp.all((a>=0)&(a<32)))
    assert bool(jnp.all(jnp.diff(a)>0))
    if count < 32:
        assert not bool(jnp.array_equal(a,sensor_layout(32,count,201)))


def test_noisy_masked_gradient_and_short_fit(setup):
    args, clean = setup
    sensors = sensor_layout(32,8,200)
    data,_ = add_measurement_noise(clean[:,sensors],args[-1],0.01,100)
    loss = jax.jit(lambda q: inverse_loss(q,*args,data,sensors))
    q,eps = jnp.log(0.001),1e-5
    value,grad = jax.value_and_grad(loss)(q)
    fd = (loss(q+eps)-loss(q-eps))/(2*eps)
    assert bool(jnp.isfinite(value)&jnp.isfinite(grad))
    assert float(jnp.abs(grad-fd)/jnp.abs(fd)) < 1e-5
    fit = optimize_log_parameter(loss,0.0005,(1e-4,0.02),iterations=60)
    assert float(fit['loss_history'][-1]) < float(fit['loss_history'][0])
