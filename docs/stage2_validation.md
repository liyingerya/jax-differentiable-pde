# Stage 2 validation: PASS

## Objective and scope

Implement and validate a differentiable explicit integrator for periodic
advection plus hyperdiffusion on [0, 2π). Stage 1 operators, tests, experiment,
and validation report are unchanged. No inverse problem, parameter optimization,
neural network, GPU benchmark, or Stage 3 implementation was added.

Created `src/solver.py`, `tests/test_solver.py`,
`experiments/stage2_time_integration_validation.py`, and this report at
`docs/stage2_validation.md`. Updated `README.md`. Requirements are unchanged.

## PDE and integrator

The PDE is ψ_t + v ψ_x = -ν4 ψ_xxxx. The implemented RHS is

    F(ψ) = -v d_dx(ψ, dx) - ν4 d4_dx4(ψ, dx).

It reuses both existing periodic spatial stencils. Positive ν4 produces damping.
Classical RK4 uses

    k1 = F(ψ)
    k2 = F(ψ + dt k1/2)
    k3 = F(ψ + dt k2/2)
    k4 = F(ψ + dt k3)
    ψ_next = ψ + dt (k1 + 2k2 + 2k3 + k4)/6.

`integrate(psi0, dt, num_steps, v, nu4, dx)` uses `jax.lax.scan` and returns
all states, including the initial state, with shape `(num_steps + 1, N)`.
It is JIT-compiled with static `num_steps`; changing that length can recompile.
Zero steps returns just the initial state; negative counts are rejected.
The caller is responsible for a stable positive dt and positive uniform dx.
All arithmetic uses JAX arrays with float64 enabled by the spatial module.

## Fourier references

For initial data sin(kx), the continuum solution is

    ψ(x,t) = exp(-ν4 k⁴ t) sin(kx - kv t).

For the spatial finite differences, define

    k1_discrete = sin(k dx)/dx
    k4_discrete = 16 sin⁴(k dx/2)/dx⁴.

The exact semi-discrete ODE solution is

    ψ_sd(x,t) = exp(-ν4 k4_discrete t) sin(kx - v k1_discrete t).

Temporal convergence uses ψ_sd, avoiding spatial truncation error. Comparisons
against the continuum solution measure both spatial and temporal error.

## Temporal convergence

Parameters: N=64, L=2π, k=3, v=1, ν4=0.0005, T=2, dx=2π/64.
The initial condition is sin(3x), with an endpoint-exclusive periodic grid.
The base count is ceil(T/(0.65 dt_max))=96. Each refinement doubles the count
and uses dt=T/num_steps, ending at the same T.

| Steps | dt | Max error vs semi-discrete | Observed order |
|---:|---:|---:|---:|
| 96 | 2.083333333e-02 | 6.553589416e-07 | — |
| 192 | 1.041666667e-02 | 4.095597202e-08 | 4.00014 |
| 384 | 5.208333333e-03 | 2.560203508e-09 | 3.99974 |
| 768 | 2.604166667e-03 | 1.600170829e-10 | 3.99996 |

Orders are log2(error_coarse/error_fine), measured from the runs. All three
are consistent with fourth order; this sequence shows no obvious saturation.

## Physical checks

All cases use the same grid, k=3, T=2, and dt=2/96. Amplitude is measured
using sine/cosine Fourier projections rather than the grid maximum.

| Case | v | ν4 | Final amplitude | Error vs continuum | Error vs semi-discrete |
|---|---:|---:|---:|---:|---:|
| Pure advection | 1 | 0 | 0.9999999636 | 8.629919854e-02 | 7.092771276e-07 |
| Pure hyperdiffusion | 0 | 0.0005 | 0.9232672538 | 1.073562329e-03 | 1.221245327e-15 |
| Combined | 1 | 0.0005 | 0.9232672644 | 7.966505601e-02 | 6.553589416e-07 |

For positive velocity, the first-step phase lag is +0.0616003143 radians in
both advective cases: sin(kx - phase) moves toward increasing x. The final
phase is reported modulo 2π as -0.3695551308 (advection) and -0.3695551325
(combined). These correspond to a positive accumulated phase near 5.913630,
not reversed motion. The exact semi-discrete accumulated phase is
5.913630884977; the continuum phase is 6.0. Pure hyperdiffusion has zero phase
motion to numerical precision.

Pure advection approximately preserves amplitude. Tests compare against both
correct and reversed propagation, include every grid point, and verify that
rolling the initial condition rolls the final state, checking periodic behavior.

Hyperdiffusion damps without reversing the Fourier amplitude or producing growth.
The continuum final amplitude is 0.922193691445, compared with the semi-discrete
value near 0.9232672538. The finite-difference stencil damps this mode slightly
less strongly. The combined solution exhibits both phase motion and damping.
Its much larger continuum error is dominated by spatial discretization.
The pure-diffusion temporal error is already at roundoff level; it is not used
to infer temporal convergence order.

## Mean conservation and dissipation

Diagnostics cover every trajectory entry; energy means mean(ψ²).

| Case | Maximum mean drift | Initial energy | Final energy | Largest per-step energy change |
|---|---:|---:|---:|---:|
| Pure advection | 8.327e-17 | 0.5 | 0.4999999636 | -3.793e-10 |
| Pure hyperdiffusion | 1.110e-16 | 0.5 | 0.4262112109 | -7.095e-04 |
| Combined | 6.245e-17 | 0.5 | 0.4262112207 | -7.095e-04 |

A supplementary combined run with ψ0=0.7+sin(3x), using the same 96 steps,
has maximum mean drift 2.220446049e-16. Tests also check a nonzero mean.
Pure hyperdiffusion energy decreases at every step; its largest increment is
strictly negative. Hyperdiffusion dissipates L2 energy; it does not conserve it.

## Explicit RK4 stability

For Fourier angle θ, the spatial eigenvalue and RK4 polynomial are

    λ(θ) = -i v sin(θ)/dx - ν4 16 sin⁴(θ/2)/dx⁴
    R(z) = 1 + z + z²/2 + z³/6 + z⁴/24.

The experiment samples 8193 angles on [0,π], brackets the boundary, and applies
60 bisection iterations to max|R(dt λ)| <= 1 + 1e-12. The tiny tolerance permits
floating-point comparison noise around the neutral zero mode. The grid includes
all independent resolvable angles for N=64. Dense sampling estimates the boundary;
it is not an exact analytical combined CFL condition or a guarantee for every
possible continuous angle. Tests verify amplification above one just beyond
the estimated boundary.

| Case | Estimated dt_max | dt used | dt/dt_max |
|---|---:|---:|---:|
| Pure advection | 2.7768018363e-01 | 2.0833333333e-02 | 0.075026 |
| Pure hyperdiffusion | 3.2343019757e-02 | 2.0833333333e-02 | 0.644137 |
| Combined | 3.2343019757e-02 | 2.0833333333e-02 | 0.644137 |

For the combined case, max|R| at the selected timestep is 1.000000000000,
including the constant mode. The selected step is comfortably inside the
sampled stability region. The pure-diffusion and combined boundaries coincide
numerically here; this is specific to these parameters, not a universal rule.

## Autodiff verification

Objective: mean(ψ(T;ν4)²), with the combined-case v=1 and initial sin(3x).
The differentiated function fixes dt=2/96 and num_steps=96; neither depends on ν4.
At ν4=0.0005, centered finite differences use epsilon=1e-7.

| Quantity | Value |
|---|---:|
| JAX gradient | -1.361088304003e+02 |
| Centered finite-difference gradient | -1.361088304239e+02 |
| Absolute difference | 2.357524e-08 |
| Relative difference | 1.732087e-10 |
| Analytic semi-discrete gradient | -1.361089090298e+02 |

The analytic reference is -T k4_discrete exp(-2ν4 k4_discrete T). Its small
difference from the numerical gradient reflects RK4 discretization. The JAX/FD
comparison verifies differentiation of the same numerical objective.
Additional tests verify finite gradients with respect to ψ0 and v, comparing
an initial-state directional derivative and a velocity derivative with centered
finite differences of a phase-sensitive scalar objective.

## Tests and execution

Commands executed from the project root:

    .venv/bin/python -m experiments.stage2_time_integration_validation
    .venv/bin/python -m pytest -q

Full pytest summary:

    ....................                                                     [100%]
    20 passed in 3.23s

This includes all 8 unchanged Stage 1 cases and 12 Stage 2 cases covering RHS
shape/constants, RK4 shape, zero/nonzero trajectory lengths, initial-state
identity, propagation direction/amplitude/periodicity, damping and energy,
nonzero mean conservation, measured temporal order, stability, and gradients.
Tests use dt=0.02 (inside the measured bound). No dependencies were added.
The environment remains Python 3.10, JAX/jaxlib 0.6.2, pytest 9.1.1.

## Numerical caveats and conclusion

- Explicit fourth-order diffusion imposes a severe dt proportional to dx⁴
  restriction at fixed ν4. Doubling N can require about 16 times as many steps.
- Spatial finite differences change phase speed and damping. Refining dt alone
  cannot remove the observed continuum error at fixed N.
- Very small dt eventually exposes floating-point effects. Pure-diffusion error
  is already near roundoff here; the combined temporal study remains resolved.
- Centered advection is nondissipative at the semi-discrete level. RK4 introduces
  small amplitude and phase errors, visible in the pure-advection diagnostics.
- Correct gradients through this discretization do not establish inverse-problem
  conditioning, parameter identifiability, or continuum-gradient accuracy.
- Full trajectories consume storage proportional to number of steps times N;
  this is intentionally a small validation solver, not a scalable adjoint system.

**PASS:** all tests pass; temporal order is approximately four, motion and damping
have the correct signs, mean drift is near machine precision, pure-diffusion
energy decreases, chosen timesteps are stable by the sampled diagnostic, and
JAX gradients agree with finite differences.

Recommended Stage 3 direction: formulate a small synthetic parameter-estimation
experiment with fixed discretization, an observation model, and identifiability
checks before optimizing v and ν4. Stage 3 is not implemented.

Git status remains:

    fatal: not a git repository (or any of the parent directories): .git

No repository was initialized, and no commit, push, or history change was made.
