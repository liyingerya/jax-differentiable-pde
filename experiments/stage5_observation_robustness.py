"""Matched-model observation robustness, separated from continuum model bias."""

import json
from pathlib import Path
import statistics

import jax
import jax.numpy as jnp

from src.inverse import inverse_loss, optimize_log_parameter, predict_observations
from src.observations import add_measurement_noise, sensor_layout
from experiments.stage2_time_integration_validation import stability_limit
from experiments.stage3_parameter_inference import continuum_observations, sampled_range_stability

N, STEPS, DT, V, TRUTH = 32, 256, 2/256, 1.0, 0.002
BOUNDS = (1e-4, 0.02)
NOISE_SEEDS = (100,101,102,103,104)
LAYOUT_SEEDS = (200,201,202,203,204)
SCHEDULES = {9: tuple(range(0,257,32)), 5: (0,64,128,192,256),
             3: (0,128,256), 2: (0,256)}
# Predeclared, not selected for favorable outcomes. All use layout seed 200.
COMBINED = ((0.01,5,16), (0.02,3,8), (0.05,3,4))


def prepare_case(times, sensors, sigma_rel, noise_seed, model):
    dx = 2*jnp.pi/N
    x = jnp.linspace(0.0, 2*jnp.pi, N, endpoint=False)
    psi0 = jnp.sin(x)+0.5*jnp.sin(2*x)+0.25*jnp.cos(3*x)
    indices, sensors = jnp.asarray(times), jnp.asarray(sensors)
    if model == 'matched':
        full = predict_observations(jnp.log(TRUTH), psi0, DT, STEPS, V, dx, indices)
        final_truth = predict_observations(jnp.log(TRUTH), psi0, DT, STEPS, V, dx, jnp.array([STEPS]))[0]
    elif model == 'continuum':
        full = continuum_observations(x, indices*DT, V, TRUTH)
        final_truth = continuum_observations(x, jnp.array([2.0]), V, TRUTH)[0]
    else:
        raise ValueError('Unknown truth model')
    clean = full[:, sensors]
    data, diagnostics = add_measurement_noise(clean, indices, sigma_rel, noise_seed)

    @jax.jit
    def loss(q):
        return inverse_loss(q, psi0, DT, STEPS, V, dx, indices, data, sensors)

    return psi0, clean, data, final_truth, loss, diagnostics


def run_case(identifier, times, sensors, sigma_rel=0.0, noise_seed=100,
             layout_seed=None, model='matched', initial=0.0005):
    psi0, clean, data, final_truth, loss, diagnostics = prepare_case(
        times, sensors, sigma_rel, noise_seed, model)
    fit = optimize_log_parameter(loss, initial, BOUNDS, iterations=800, learning_rate=0.05)
    q, coefficient = fit['q'], float(fit['nu4'])
    grad_fn = jax.jit(jax.grad(loss))
    h = 1e-3
    gradient = float(grad_fn(q))
    left, right = float(grad_fn(q-h)), float(grad_fn(q+h))
    curvature = (right-left)/(2*h)
    observed = predict_observations(q, psi0, DT, STEPS, V, 2*jnp.pi/N,
                                    jnp.array(times), jnp.array(sensors))
    final = predict_observations(q, psi0, DT, STEPS, V, 2*jnp.pi/N, jnp.array([STEPS]))[0]
    # Box-constrained minima are judged by the appropriate one-sided gradient.
    at_lower = coefficient <= BOUNDS[0]*(1+1e-8)
    at_upper = coefficient >= BOUNDS[1]*(1-1e-8)
    stationary = ((at_lower and gradient >= -1e-8) or
                  (at_upper and gradient <= 1e-8) or
                  (not at_lower and not at_upper and abs(gradient) < 1e-8 and left < 0 < right))
    return {"id": identifier, "truth_model": model, "time_indices": list(times),
            "times": [i*DT for i in times], "informative_snapshots": sum(i>0 for i in times),
            "sensor_indices": list(sensors), "sensor_layout_seed": layout_seed,
            "sigma_rel": sigma_rel, "noise_seed": noise_seed, **diagnostics,
            "initial_nu4": initial, "recovered_nu4": coefficient,
            "signed_parameter_error": coefficient-TRUTH,
            "absolute_relative_error": abs(coefficient-TRUTH)/TRUTH,
            "initial_loss": float(fit['loss_history'][0]), "final_loss": float(fit['loss_history'][-1]),
            "final_gradient": gradient, "curvature_in_q": curvature,
            "local_probes": [{"q_offset": offset, "loss": float(loss(q+offset)),
                              "gradient": float(grad_fn(q+offset))} for offset in (-h,0.,h)],
            "stationary": bool(stationary), "at_parameter_bound": bool(at_lower or at_upper),
            "boundary_hits": int(fit['boundary_hits']),
            "history_min_nu4": float(jnp.min(fit['nu4_history'])),
            "history_max_nu4": float(jnp.max(fit['nu4_history'])),
            "clean_observation_nmse": float(jnp.mean((observed-clean)**2)/(jnp.mean(clean**2)+1e-12)),
            "final_clean_field_relative_l2": float(jnp.linalg.norm(final-final_truth)/jnp.linalg.norm(final_truth)),
            "loss_history": fit['loss_history'].tolist(), "nu4_history": fit['nu4_history'].tolist()}


def summarize(rows):
    coefficients = [a['recovered_nu4'] for a in rows]
    errors = [a['absolute_relative_error'] for a in rows]
    curvatures = [a['curvature_in_q'] for a in rows]
    return {"count": len(rows), "mean_nu4": statistics.mean(coefficients),
            "median_nu4": statistics.median(coefficients),
            "sample_std_nu4": statistics.stdev(coefficients) if len(rows)>1 else None,
            "min_nu4": min(coefficients), "max_nu4": max(coefficients),
            "mean_absolute_relative_error": statistics.mean(errors),
            "median_absolute_relative_error": statistics.median(errors),
            "mean_curvature": statistics.mean(curvatures),
            "min_curvature": min(curvatures), "max_curvature": max(curvatures),
            "mean_final_clean_field_relative_l2": statistics.mean(a['final_clean_field_relative_l2'] for a in rows),
            "mean_realized_noise_rms": statistics.mean(a['realized_noise_rms'] for a in rows)}


def main():
    destination = Path(__file__).resolve().parents[1]/'docs/stage5_results.json'
    limit, _ = stability_limit(V, BOUNDS[1], 2*jnp.pi/N)
    amplification = sampled_range_stability(V, 2*jnp.pi/N, DT, BOUNDS)
    if not (DT/limit < 0.65 and amplification <= 1+1e-12):
        raise RuntimeError('Stability check failed')
    results = {"configuration": {"N": N, "num_steps": STEPS, "dt": DT, "v": V,
                "nu4_true": TRUTH, "bounds": BOUNDS, "iterations": 800, "learning_rate": 0.05,
                "noise_seeds": NOISE_SEEDS, "layout_seeds": LAYOUT_SEEDS,
                "combined_cases": COMBINED, "combined_layout_seed": 200,
                "baseline_relative_tolerance": 1e-4, "continuum_baseline_atol": 1e-10,
                "gradient_relative_tolerance": 1e-5, "stationarity_tolerance": 1e-8},
               "stability": {"dt_max": limit, "safety_fraction": DT/limit, "range_max_amplification": amplification},
               "runs": {}, "studies": {}, "experiment_conclusion": "IN PROGRESS"}

    def save():
        destination.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')

    def execute(identifier, times=SCHEDULES[9], sensors=tuple(range(N)), **kwargs):
        print('Starting', identifier, flush=True)
        row = run_case(identifier, times, sensors, **kwargs)
        results['runs'][identifier] = row
        save()
        print(identifier, 'nu4=', row['recovered_nu4'], 'rel_error=', row['absolute_relative_error'],
              'gradient=', row['final_gradient'], 'curvature=', row['curvature_in_q'], flush=True)
        jax.clear_caches()
        return identifier

    def group(name, identifiers):
        results['studies'][name] = {"run_ids": identifiers,
                                  "summary": summarize([results['runs'][key] for key in identifiers])}
        save()

    save()
    baseline = execute('matched_clean_full')
    group('noise_0', [baseline])
    for level in (0.005,0.01,0.02,0.05):
        group(f'noise_{level}', [execute(f'noise_{level}_seed_{seed}', sigma_rel=level, noise_seed=seed) for seed in NOISE_SEEDS])
    group('times_9', [baseline])
    for count in (5,3,2):
        group(f'times_{count}', [execute(f'times_{count}', times=SCHEDULES[count])])
    group('sensors_32', [baseline])
    layouts = {}
    for count in (16,8,4):
        identifiers = []
        for seed in LAYOUT_SEEDS:
            sensors = sensor_layout(N,count,seed).tolist()
            layouts[count,seed] = sensors
            identifiers.append(execute(f'sensors_{count}_layout_{seed}', sensors=sensors, layout_seed=seed))
        group(f'sensors_{count}', identifiers)
    for number,(level,count,sensors_count) in enumerate(COMBINED,1):
        group(f'combined_{number}', [execute(f'combined_{number}_seed_{seed}', times=SCHEDULES[count],
              sensors=layouts[sensors_count,200], sigma_rel=level, noise_seed=seed, layout_seed=200) for seed in NOISE_SEEDS])

    # Fixed masked, noisy data at an off-optimum q; no noise resampling in loss.
    *_, loss, _ = prepare_case(SCHEDULES[3],layouts[8,200],0.02,100,'matched')
    q, eps = jnp.log(0.001), 1e-5
    gradient = float(jax.grad(loss)(q))
    fd = float((loss(q+eps)-loss(q-eps))/(2*eps))
    results['gradient_check'] = {"q": float(q), "nu4": 0.001, "epsilon": eps,
        "time_indices": SCHEDULES[3], "sensor_indices": layouts[8,200], "noise_seed": 100,
        "sigma_rel": 0.02, "jax": gradient, "finite_difference": fd,
        "absolute_difference": abs(gradient-fd), "relative_difference": abs(gradient-fd)/max(abs(fd),1e-15)}
    results['multi_start_checks'] = []
    for primary in ('noise_0.05_seed_100','combined_3_seed_100','combined_3_seed_104'):
        source = results['runs'][primary]
        alternate = execute(primary+'_alternate', times=source['time_indices'], sensors=source['sensor_indices'],
            sigma_rel=source['sigma_rel'], noise_seed=source['noise_seed'], layout_seed=source['sensor_layout_seed'], initial=0.01)
        results['multi_start_checks'].append({"primary": primary, "alternate": alternate,
            "absolute_coefficient_difference": abs(source['recovered_nu4']-results['runs'][alternate]['recovered_nu4'])})
    clean_continuum = execute('continuum_clean_full', model='continuum')
    continuum_reference = results['runs'][clean_continuum]['recovered_nu4']
    prior = json.loads((destination.parent/'stage4_results.json').read_text())['temporal_refinement'][0]['recovered_nu4']
    results['continuum_baseline'] = {"reference_nu4": continuum_reference, "stage4_nu4": prior,
        "absolute_difference": abs(continuum_reference-prior)}
    group('continuum_clean_full', [clean_continuum])
    group('continuum_noise_0.01', [execute(f'continuum_noise_seed_{seed}', model='continuum', sigma_rel=0.01, noise_seed=seed) for seed in NOISE_SEEDS])
    # Extra clean masked control distinguishes selection shift from added noise.
    group('continuum_clean_sparse', [execute('continuum_clean_sparse', model='continuum', times=SCHEDULES[3], sensors=layouts[8,200], layout_seed=200)])
    group('continuum_combined', [execute(f'continuum_combined_seed_{seed}', model='continuum', times=SCHEDULES[3],
        sensors=layouts[8,200], layout_seed=200, sigma_rel=0.02, noise_seed=seed) for seed in NOISE_SEEDS])
    for row in results['runs'].values():
        if row['truth_model']=='continuum':
            row['clean_discrete_reference'] = continuum_reference
            row['observation_induced_shift'] = row['recovered_nu4']-continuum_reference
            row['absolute_relative_shift_from_clean_discrete'] = abs(row['observation_induced_shift'])/continuum_reference
    for name, study in results['studies'].items():
        rows = [results['runs'][key] for key in study['run_ids']]
        if rows[0]['truth_model']=='continuum':
            study['summary']['mean_observation_induced_shift'] = statistics.mean(a['observation_induced_shift'] for a in rows)
            study['summary']['mean_physical_error'] = statistics.mean(a['signed_parameter_error'] for a in rows)
    rows = list(results['runs'].values())
    checks = {"matched_baseline": results['runs'][baseline]['absolute_relative_error'] < 1e-4,
        "continuum_baseline": abs(continuum_reference-prior)<1e-10,
        "masked_gradient": results['gradient_check']['relative_difference']<1e-5,
        "stationary": all(a['stationary'] for a in rows),
        "stable_histories": all(a['history_min_nu4'] >= BOUNDS[0]*(1-1e-12) and a['history_max_nu4'] <= BOUNDS[1]*(1+1e-12) for a in rows)}
    results['checks'] = checks
    results['experiment_conclusion'] = 'PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    save()
    print('Gradient:', results['gradient_check'], flush=True)
    print('Checks:', checks, flush=True)
    print(results['experiment_conclusion'], '(full pytest must also pass)', flush=True)
    if not all(checks.values()):
        raise RuntimeError('Stage 5 numerical checks need attention')


if __name__ == '__main__':
    main()
