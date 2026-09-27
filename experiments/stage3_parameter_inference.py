"""Deterministic matched-model and continuum-data scalar inverse experiments."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from src.inverse import inverse_loss, optimize_log_parameter, predict_observations
from experiments.stage2_time_integration_validation import stability_limit

# Declared before recovery: these do not change in response to measured results.
CRITERIA = {"matched_relative_error": 1e-4, "matched_loss_reduction": 1e6,
            "gradient_relative_error": 1e-5, "continuum_final_relative_l2": 0.1}


def continuum_observations(x, times, v, nu4):
    """Exact PDE evolution of sin(x) + .5 sin(2x) + .25 cos(3x)."""
    phase = x[None, :] - v * times[:, None]
    t = times[:, None]
    return (jnp.exp(-nu4 * t) * jnp.sin(phase)
            + 0.5 * jnp.exp(-nu4 * 16 * t) * jnp.sin(2 * phase)
            + 0.25 * jnp.exp(-nu4 * 81 * t) * jnp.cos(3 * phase))


def sampled_range_stability(v, dx, dt, bounds):
    """Check RK4 amplification over the declared parameter interval and angles."""
    coefficients = jnp.geomspace(*bounds, 101)
    theta = jnp.linspace(0.0, jnp.pi, 8193)
    eigenvalues = (-1j * v * jnp.sin(theta)[None, :] / dx
                  - coefficients[:, None] * 16 * jnp.sin(theta / 2)[None, :]**4 / dx**4)
    z = dt * eigenvalues
    return float(jnp.max(jnp.abs(1 + z + z**2/2 + z**3/6 + z**4/24)))


def main():
    n, steps, final_time, speed, truth = 32, 256, 2.0, 1.0, 0.002
    bounds, starts = (1e-4, 0.02), (0.0005, 0.001, 0.005, 0.01)
    dt, dx = final_time / steps, 2 * jnp.pi / n
    indices = jnp.arange(0, steps + 1, 32)
    times = indices * dt
    x = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    psi0 = jnp.sin(x) + 0.5 * jnp.sin(2*x) + 0.25 * jnp.cos(3*x)
    limit, _ = stability_limit(speed, bounds[1], dx)
    range_amplification = sampled_range_stability(speed, dx, dt, bounds)
    if not (dt < 0.65 * limit and range_amplification <= 1 + 1e-12):
        raise RuntimeError("Preselected timestep failed stability checks")

    matched = predict_observations(jnp.log(truth), psi0, dt, steps, speed, dx, indices)
    continuum = continuum_observations(x, times, speed, truth)
    result = {
        "config": {"N": n, "num_steps": steps, "T": final_time, "dt": dt,
                   "v": speed, "nu4_true": truth, "bounds": bounds,
                   "observation_indices": indices.tolist(), "observation_times": times.tolist(),
                   "iterations": 800, "learning_rate": 0.05},
        "criteria": CRITERIA,
        "stability": {"dt_max_at_upper_bound": limit, "safety_fraction": dt/limit,
                      "max_amplification_over_sampled_range": range_amplification},
    }
    print("Configuration:", result["config"], flush=True)
    print("Stability:", result["stability"], flush=True)
    candidates = jnp.geomspace(*bounds, 81)

    for name, observations, initial_guesses in (("matched", matched, starts), ("continuum", continuum, starts)):
        @jax.jit
        def loss(q):
            return inverse_loss(q, psi0, dt, steps, speed, dx, indices, observations)

        checks = []
        for candidate in (0.001, 0.005):
            q, eps = jnp.log(candidate), 1e-5
            grad = float(jax.grad(loss)(q))
            fd = float((loss(q + eps) - loss(q - eps)) / (2 * eps))
            checks.append({"q": float(q), "nu4": candidate, "epsilon": eps,
                           "jax": grad, "finite_difference": fd,
                           "absolute_difference": abs(grad-fd),
                           "relative_difference": abs(grad-fd)/max(abs(fd), 1e-15)})
        landscape = [float(loss(jnp.log(candidate))) for candidate in candidates]
        minimum = min(range(len(landscape)), key=landscape.__getitem__)
        data = {"gradient_checks": checks,
                "landscape": {"nu4": candidates.tolist(), "loss": landscape,
                              "minimum_nu4": float(candidates[minimum]), "minimum_loss": landscape[minimum]},
                "runs": []}
        print(name, "gradient checks:", checks, flush=True)
        print(name, "landscape minimum:", data["landscape"]["minimum_nu4"], landscape[minimum], flush=True)
        for initial in initial_guesses:
            fit = optimize_log_parameter(loss, initial, bounds)
            recovered = float(fit["nu4"])
            prediction = predict_observations(fit["q"], psi0, dt, steps, speed, dx, indices)
            run = {"initial_nu4": initial, "recovered_nu4": recovered,
                   "absolute_parameter_error": abs(recovered-truth),
                   "relative_parameter_error": abs(recovered-truth)/truth,
                   "initial_loss": float(fit["loss_history"][0]),
                   "final_loss": float(fit["loss_history"][-1]),
                   "iterations": 800, "boundary_hits": int(fit["boundary_hits"]),
                   "mse": float(jnp.mean((prediction-observations)**2)),
                   "final_relative_l2": float(jnp.linalg.norm(prediction[-1]-observations[-1])/jnp.linalg.norm(observations[-1])),
                   "loss_history": fit["loss_history"].tolist(), "nu4_history": fit["nu4_history"].tolist()}
            data["runs"].append(run)
            print(name, {key: value for key, value in run.items() if not key.endswith("history")}, flush=True)
        best = min(data["runs"], key=lambda run: run["final_loss"])
        q_best = jnp.log(best["recovered_nu4"])
        offsets = (-0.01, 0.0, 0.01)
        data["local_landscape"] = [
            {"log_offset": offset, "nu4": float(jnp.exp(q_best+offset)),
             "loss": float(loss(q_best+offset)), "gradient": float(jax.grad(loss)(q_best+offset))}
            for offset in offsets]
        data["loss_at_truth"] = float(loss(jnp.log(truth)))
        result[name] = data
        print(name, "local landscape:", data["local_landscape"], flush=True)

    matched_runs = result["matched"]["runs"]
    all_runs = matched_runs + result["continuum"]["runs"]
    checks = {
        "matched_recovery": all(run["relative_parameter_error"] < CRITERIA["matched_relative_error"] for run in matched_runs),
        "matched_loss_reduction": all(run["final_loss"] < run["initial_loss"] / CRITERIA["matched_loss_reduction"] for run in matched_runs),
        "gradient_agreement": all(check["relative_difference"] < CRITERIA["gradient_relative_error"] for name in ("matched", "continuum") for check in result[name]["gradient_checks"]),
        "matched_grid_minimum_near_truth": abs(result["matched"]["landscape"]["minimum_nu4"] / truth - 1) < 0.04,
        "local_minima": all(result[name]["local_landscape"][0]["gradient"] < 0 < result[name]["local_landscape"][2]["gradient"] for name in ("matched", "continuum")),
        "interior_recovery": all(bounds[0] < run["recovered_nu4"] < bounds[1] for run in all_runs),
        "stable_history": all(bounds[0]*(1-1e-12) <= value <= bounds[1]*(1+1e-12) for run in all_runs for value in run["nu4_history"]),
        "continuum_field_error": all(run["final_relative_l2"] < CRITERIA["continuum_final_relative_l2"] and run["final_loss"] < run["initial_loss"] for run in result["continuum"]["runs"]),
    }
    result["checks"] = checks
    result["experiment_conclusion"] = "PASS" if all(checks.values()) else "NEEDS ATTENTION"
    destination = Path(__file__).resolve().parents[1] / "docs" / "stage3_results.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Checks:", checks, flush=True)
    print("Experiment:", result["experiment_conclusion"], "(full pytest must also pass)", flush=True)
    print("Saved", destination, flush=True)
    if not all(checks.values()):
        raise RuntimeError("Stage 3 validation criteria not met")


if __name__ == "__main__":
    main()
