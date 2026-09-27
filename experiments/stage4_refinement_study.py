"""Separate temporal and spatial studies of continuum-data parameter bias."""

import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp

from src.inverse import inverse_loss, optimize_log_parameter, predict_observations
from experiments.stage2_time_integration_validation import stability_limit
from experiments.stage3_parameter_inference import continuum_observations, sampled_range_stability

TRUTH = 0.002
BOUNDS = (1e-4, 0.02)
SPATIAL_GRIDS = (16, 24, 32, 40, 48)
TEMPORAL_STEPS = (256, 512, 1024, 2048)
SAFETY = 0.5
# Set before execution; no prescribed empirical convergence order.
BASELINE_ATOL = 1e-10
GRADIENT_TOL = 1e-10
MULTISTART_ATOL = 1e-10


def select_timestep(dt_max, safety=SAFETY, final_time=2.0, intervals=8):
    """Round steps up to align a uniform observation schedule with a safe dt."""
    if not (dt_max > 0 and 0 < safety < 1 and final_time > 0 and intervals > 0):
        raise ValueError("Require positive times/intervals and safety in (0,1)")
    steps_per_interval = math.ceil(final_time / (intervals * safety * dt_max))
    steps = intervals * steps_per_interval
    return final_time / steps, steps, jnp.arange(intervals + 1) * steps_per_interval


def bias_metrics(coefficient, truth=TRUTH):
    bias = float(coefficient) - truth
    return {"signed_bias": bias, "absolute_error": abs(bias),
            "relative_bias": bias / truth, "relative_error": abs(bias) / truth}


def modified_wavenumbers(n):
    dx = 2 * jnp.pi / n
    k = jnp.arange(1, 4, dtype=jnp.float64)
    discrete = 16 * jnp.sin(k * dx / 2)**4 / dx**4
    return {"N": n, "k": k.tolist(), "k4_continuum": (k**4).tolist(),
            "k4_discrete": discrete.tolist(),
            "effective_nu4": (TRUTH * k**4 / discrete).tolist()}


def run_fit(n, v=1.0, num_steps=None, initial=0.0005, iterations=800):
    """Fit using Stage 3 loss and optimizer, recording local convergence evidence."""
    started = time.perf_counter()
    dx = 2 * jnp.pi / n
    limit, _ = stability_limit(v, BOUNDS[1], dx)
    if num_steps is None:
        dt, num_steps, indices = select_timestep(limit)
    else:
        if num_steps <= 0 or num_steps % 8:
            raise ValueError("num_steps must be positive and divisible by 8")
        dt = 2.0 / num_steps
        indices = jnp.arange(9) * (num_steps // 8)
    amplification = sampled_range_stability(v, dx, dt, BOUNDS)
    if dt / limit > 0.65 or amplification > 1 + 1e-12:
        raise ValueError("Requested fixed step is not comfortably stable")
    x = jnp.linspace(0.0, 2*jnp.pi, n, endpoint=False)
    times = jnp.arange(9, dtype=jnp.float64) / 4
    observations = continuum_observations(x, times, v, TRUTH)
    psi0 = observations[0]

    @jax.jit
    def loss(q):
        return inverse_loss(q, psi0, dt, num_steps, v, dx, indices, observations)

    fit = optimize_log_parameter(loss, initial, BOUNDS, iterations=iterations, learning_rate=0.05)
    q = fit["q"]
    gradient_fn = jax.jit(jax.grad(loss))
    h = 1e-4
    gradient = float(gradient_fn(q))
    left, right = float(gradient_fn(q-h)), float(gradient_fn(q+h))
    curvature = (right-left)/(2*h)
    recovered = float(fit["nu4"])
    estimated_parameter_correction = recovered * abs(gradient/curvature) if curvature > 0 else None
    prediction = predict_observations(q, psi0, dt, num_steps, v, dx, indices)
    record = {"N": n, "v": v, "dx": float(dx), "num_steps": num_steps,
              "dt": dt, "dt_max": limit, "safety_fraction": dt/limit,
              "range_max_amplification": amplification,
              "observation_indices": indices.tolist(), "initial_nu4": initial,
              "recovered_nu4": recovered, **bias_metrics(recovered),
              "initial_loss": float(fit["loss_history"][0]),
              "final_loss": float(fit["loss_history"][-1]),
              "final_relative_l2": float(jnp.linalg.norm(prediction[-1]-observations[-1])/jnp.linalg.norm(observations[-1])),
              "final_gradient": gradient, "probe_log_offset": h,
              "left_gradient": left, "right_gradient": right,
              "local_curvature": curvature,
              "estimated_parameter_correction": estimated_parameter_correction,
              "boundary_hits": int(fit["boundary_hits"]), "iterations": iterations,
              "history_min_nu4": float(jnp.min(fit["nu4_history"])),
              "history_max_nu4": float(jnp.max(fit["nu4_history"])),
              "loss_history": fit["loss_history"].tolist(),
              "nu4_history": fit["nu4_history"].tolist(),
              "wall_seconds_including_compilation": time.perf_counter()-started}
    return record


def main():
    destination = Path(__file__).resolve().parents[1] / 'docs/stage4_results.json'
    results = {"configuration": {"truth": TRUTH, "v": 1.0, "T": 2.0,
               "bounds": BOUNDS, "spatial_grids": SPATIAL_GRIDS,
               "temporal_steps": TEMPORAL_STEPS, "spatial_safety": SAFETY,
               "iterations": 800, "learning_rate": 0.05,
               "baseline_atol": BASELINE_ATOL, "gradient_tol": GRADIENT_TOL,
               "multistart_atol": MULTISTART_ATOL},
               "temporal_refinement": [], "spatial_refinement": [],
               "pure_hyperdiffusion_refinement": [], "multi_start_checks": [],
               "temporal_control_checks": [],
               "modified_wavenumber_diagnostics": [modified_wavenumbers(n) for n in (16,32,48)],
               "experiment_conclusion": "IN PROGRESS"}

    def save():
        destination.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')

    def append_fit(section, **kwargs):
        print('Starting', section, kwargs, flush=True)
        row = run_fit(**kwargs)
        results[section].append(row)
        save()
        print(section, {key: row[key] for key in ('N','num_steps','v','initial_nu4','recovered_nu4','relative_bias','final_loss','final_gradient','wall_seconds_including_compilation')}, flush=True)
        # Different shapes are intentionally compiled independently; release caches
        # after a completed run to avoid accumulating large reverse-mode executables.
        jax.clear_caches()
        return row

    save()
    for steps in TEMPORAL_STEPS:
        append_fit('temporal_refinement', n=32, num_steps=steps)
    for n in SPATIAL_GRIDS:
        append_fit('spatial_refinement', n=n)
    for n in (16,32,48):
        append_fit('pure_hyperdiffusion_refinement', n=n, v=0.0)
    for n in (16,32,48):
        append_fit('multi_start_checks', n=n, initial=0.01)
    # Direct dt-halving checks at the coarsest and finest grids establish that
    # the spatial trend is not a hidden time-stepping trend.
    for n in (16,48):
        base = next(row for row in results['spatial_refinement'] if row['N']==n)
        append_fit('temporal_control_checks', n=n, num_steps=2*base['num_steps'])

    temporal = results['temporal_refinement']
    for row in temporal:
        row['difference_from_finest'] = abs(row['recovered_nu4']-temporal[-1]['recovered_nu4'])
    spatial = results['spatial_refinement']
    for coarse, fine in zip(spatial, spatial[1:]):
        fine['bias_order_from_previous'] = math.log(coarse['absolute_error']/fine['absolute_error'])/math.log(coarse['dx']/fine['dx'])
        fine['field_order_from_previous'] = math.log(coarse['final_relative_l2']/fine['final_relative_l2'])/math.log(coarse['dx']/fine['dx'])
    prior = json.loads((destination.parent/'stage3_results.json').read_text())
    baseline = prior['continuum']['runs'][0]['recovered_nu4']
    results['baseline_comparison'] = {"stage3_nu4": baseline,
        "stage4_nu4": temporal[0]['recovered_nu4'],
        "absolute_difference": abs(baseline-temporal[0]['recovered_nu4'])}
    all_rows = sum((results[key] for key in ('temporal_refinement','spatial_refinement','pure_hyperdiffusion_refinement','multi_start_checks','temporal_control_checks')), [])
    trends = [abs(a['recovered_nu4']-b['recovered_nu4']) for a,b in zip(spatial,spatial[1:])]
    checks = {
        "baseline_reproduced": abs(baseline-temporal[0]['recovered_nu4']) < BASELINE_ATOL,
        "stable": all(row['range_max_amplification'] <= 1+1e-12 and row['history_min_nu4'] >= BOUNDS[0]*(1-1e-12) and row['history_max_nu4'] <= BOUNDS[1]*(1+1e-12) for row in all_rows),
        "stationary_interior_minima": all(abs(row['final_gradient']) < GRADIENT_TOL and row['left_gradient'] < 0 < row['right_gradient'] for row in all_rows),
        "optimizer_error_below_spatial_trend": all(row['estimated_parameter_correction'] is not None and row['estimated_parameter_correction'] < min(trends)*1e-3 for row in spatial),
        "multistart_agreement": all(abs(row['recovered_nu4']-next(base['recovered_nu4'] for base in spatial if base['N']==row['N'])) < MULTISTART_ATOL for row in results['multi_start_checks']),
        "spatial_bias_decreases": all(b['absolute_error'] < a['absolute_error'] for a,b in zip(spatial,spatial[1:])),
        "spatial_time_error_controlled": all(abs(row['recovered_nu4']-next(base['recovered_nu4'] for base in spatial if base['N']==row['N'])) < row['absolute_error']*1e-3 for row in results['temporal_control_checks']),
    }
    results['checks'] = checks
    results['experiment_conclusion'] = 'PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    save()
    print('Checks:', checks, flush=True)
    print(results['experiment_conclusion'], '(full pytest must also pass)', flush=True)
    if not all(checks.values()):
        raise RuntimeError('Refinement validation needs attention')


if __name__ == '__main__':
    main()
