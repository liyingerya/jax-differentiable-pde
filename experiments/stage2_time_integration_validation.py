"""Fourier validation, sampled RK4 stability, and fixed-step gradient checks."""

import math

import jax
import jax.numpy as jnp

from src.solver import integrate


def fourier_exact(x, t, k, v, nu4, dx=None):
    """Continuum solution, or semi-discrete solution when dx is supplied."""
    k1 = k if dx is None else jnp.sin(k * dx) / dx
    k4 = k**4 if dx is None else 16 * jnp.sin(k * dx / 2)**4 / dx**4
    return jnp.exp(-nu4 * k4 * t) * jnp.sin(k * x - v * k1 * t)


def stability_limit(v, nu4, dx):
    """Estimate dt_max on 8193 angles; not an analytic combined CFL bound."""
    theta = jnp.linspace(0.0, jnp.pi, 8193)
    eigenvalues = -1j * v * jnp.sin(theta) / dx - nu4 * 16 * jnp.sin(theta / 2)**4 / dx**4

    @jax.jit
    def amplification(dt):
        z = dt * eigenvalues
        return jnp.max(jnp.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24))

    # The zero mode always has |R|=1; allow only floating-point comparison noise.
    low, high = 0.0, 1.0
    for _ in range(60):
        if float(amplification(high)) > 1 + 1e-12:
            break
        high *= 2
    else:
        raise ValueError("No finite stability boundary found")
    for _ in range(60):
        mid = (low + high) / 2
        if float(amplification(mid)) <= 1 + 1e-12:
            low = mid
        else:
            high = mid
    return low, amplification


def mode_amplitude_phase(f, x, k):
    sine = 2 * jnp.mean(f * jnp.sin(k * x))
    cosine = 2 * jnp.mean(f * jnp.cos(k * x))
    return float(jnp.hypot(sine, cosine)), float(jnp.arctan2(-cosine, sine))


def main():
    n, k, v, nu4, final_time = 64, 3, 1.0, 0.0005, 2.0
    dx = 2 * jnp.pi / n
    x = jnp.linspace(0.0, 2 * jnp.pi, n, endpoint=False)
    psi0 = jnp.sin(k * x)
    dt_max, amplification = stability_limit(v, nu4, dx)
    # Round the step count up so all refinements end at exactly final_time.
    base_steps = math.ceil(final_time / (0.65 * dt_max))
    dt = final_time / base_steps
    print(f"N={n}, k={k}, v={v}, nu4={nu4}, T={final_time}")
    print(f"Combined stability: dt_max={dt_max:.10e}, dt={dt:.10e}, fraction={dt/dt_max:.6f}, max|R|={float(amplification(dt)):.12f}")
    print("Temporal convergence against semi-discrete exact solution")
    print("steps       dt              max error         order")
    previous = None
    for factor in (1, 2, 4, 8):
        steps = base_steps * factor
        step_dt = final_time / steps
        final = integrate(psi0, step_dt, steps, v, nu4, dx)[-1]
        error = float(jnp.max(jnp.abs(final - fourier_exact(x, final_time, k, v, nu4, dx))))
        order = math.log2(previous / error) if previous is not None else float('nan')
        print(f"{steps:5d} {step_dt:16.9e} {error:16.9e} {order:9.5f}")
        previous = error

    print("Physical cases (same dt and T; phase reported modulo 2*pi)")
    for name, speed, diffusion in (("advection", v, 0.0), ("hyperdiffusion", 0.0, nu4), ("combined", v, nu4)):
        trajectory = integrate(psi0, dt, base_steps, speed, diffusion, dx)
        final = trajectory[-1]
        continuum_error = float(jnp.max(jnp.abs(final - fourier_exact(x, final_time, k, speed, diffusion))))
        discrete_error = float(jnp.max(jnp.abs(final - fourier_exact(x, final_time, k, speed, diffusion, dx))))
        amplitude, phase = mode_amplitude_phase(final, x, k)
        mean_drift = float(jnp.max(jnp.abs(jnp.mean(trajectory, axis=1) - jnp.mean(psi0))))
        energy = jnp.mean(trajectory**2, axis=1)
        limit, _ = stability_limit(speed, diffusion, dx)
        early_phase = mode_amplitude_phase(trajectory[1], x, k)[1]
        print(f"{name}: continuum_error={continuum_error:.9e}, semidiscrete_error={discrete_error:.9e}, amplitude={amplitude:.10f}, phase={phase:.10f}, first_step_phase={early_phase:.10f}")
        print(f"  mean_drift={mean_drift:.3e}, energy_initial={float(energy[0]):.10f}, energy_final={float(energy[-1]):.10f}, max_energy_increment={float(jnp.max(jnp.diff(energy))):.3e}, dt_max={limit:.10e}, safety={dt/limit:.6f}")

    # Neither the step count nor the step size depends on the differentiated nu4.
    def objective(diffusion):
        final = integrate(psi0, dt, base_steps, v, diffusion, dx)[-1]
        return jnp.mean(final**2)

    eps = 1e-7
    grad_jax = float(jax.grad(objective)(nu4))
    grad_fd = float((objective(nu4 + eps) - objective(nu4 - eps)) / (2 * eps))
    k4 = 16 * jnp.sin(k * dx / 2)**4 / dx**4
    grad_exact = float(-final_time * k4 * jnp.exp(-2 * nu4 * k4 * final_time))
    print(f"Gradient: eps={eps:.1e}, jax={grad_jax:.12e}, fd={grad_fd:.12e}, abs_difference={abs(grad_jax-grad_fd):.6e}, relative_difference={abs(grad_jax-grad_fd)/abs(grad_fd):.6e}, semidiscrete_analytic={grad_exact:.12e}")


if __name__ == "__main__":
    main()
