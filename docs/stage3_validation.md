# Stage 3 validation: PASS

## Objective and scope

Recover the positive hyperdiffusion coefficient ν4 from noiseless full-field
observations with known velocity v, by differentiating through the validated
Stage 2 solver for ψ_t = -v ψ_x - ν4 ψ_xxxx. This stage fits one scalar only.
No joint velocity inference, noise, neural networks, GPU benchmarking, or
Stage 4 work is implemented.

Created:
- `src/inverse.py`: parameter transformation, observation extraction, normalized
  loss, and scalar projected Adam.
- `experiments/stage3_parameter_inference.py`: deterministic two-model validation.
- `tests/test_inverse.py`: seven fast test cases.
- `docs/stage3_results.json`: full sampled landscapes, all optimization histories,
  gradient checks, configuration, metrics, and acceptance checks.
- `docs/stage3_validation.md`: this report.

Modified `README.md`; requirements are unchanged. SHA-256 checks confirm that
Stage 1/2 operators, solver, tests, and validation reports remain unchanged.

## Inverse formulation and observations

Optimize q = log(ν4), with ν4 = exp(q), using predictions from the existing
`src.solver.integrate`. The inverse code does not implement another solver.
For the bounded finite q range used here, this transformation ensures positivity.
Arbitrarily extreme floating-point q values can still underflow/overflow exp;
the declared search bounds avoid that issue.

Configuration:

| Quantity | Value |
|---|---|
| Domain | [0, 2π), endpoint excluded |
| Grid points | 32 |
| Known v | 1 |
| True ν4 | 0.002 |
| Final time T | 2 |
| Fixed num_steps | 256 |
| Fixed dt | 0.0078125 |
| Observation indices | 0, 32, 64, 96, 128, 160, 192, 224, 256 |
| Observation times | 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2 |
| Search interval | [0.0001, 0.02] |

Initial condition:

    ψ0(x) = sin(x) + 0.5 sin(2x) + 0.25 cos(3x).

There are nine observations of all 32 spatial points. Eight observations occur
after t=0 and provide the parameter information. The t=0 residual is identically
zero. Selected observations are retained as arrays of shape (9,32), rather
than storing a truth trajectory. The solver and reverse-mode differentiation
can still allocate intermediate trajectory states internally.

The continuum damping factors at T are exp(-0.004), exp(-0.064), and
exp(-0.324), approximately 0.9960, 0.9380, and 0.7233 for modes 1, 2, and 3.
Thus damping is measurable without erasing the field.

Experiment A generates data with the same solver used in fitting. Experiment B
uses the exact continuum field at each observation time:

    exp(-ν4 t) sin(x-vt)
    + 0.5 exp(-16 ν4 t) sin(2(x-vt))
    + 0.25 exp(-81 ν4 t) cos(3(x-vt)).

No random noise is added. Matched-model recovery isolates the optimization and
autodiff pipeline; continuum observations expose numerical-model bias.

## Fixed graph and stability

The timestep, number of steps, grid, velocity, and observation indices are fixed
through every loss and gradient evaluation. No candidate parameter changes dt.
The Stage 2 diagnostic estimates dt_max = 0.012937207902633223 at ν4=0.02
using 8193 angles and bisection. The selected dt/dt_max is 0.6038783684.

A supplementary check samples 101 logarithmically spaced coefficients across
the declared interval and 8193 Fourier angles for each. The largest RK4
amplification is 1.0, including the neutral zero mode. This is numerical
stability evidence, not an exact analytic CFL proof for a continuous interval.
The angular samples include all independent N=32 resolvable angles.

Adam proposals are projected onto [log(0.0001), log(0.02)] before their loss
is evaluated. Bounds are chosen for stability and do not use the true parameter.
All eight runs had zero boundary hits and all histories stayed in range;
observed coefficients ranged from approximately 0.0005 to 0.01.

## Loss and optimizer

The optimized scalar objective is exactly

    L(q) = mean((prediction(q) - observations)^2)
           / (mean(observations^2) + 1e-12).

Means cover observation times and spatial locations. There is no regularizer.
Raw MSE and final-time relative L2 error are also recorded. All loss values
below are normalized MSE unless stated otherwise.

Each run uses 800 projected Adam updates, fixed learning rate α=0.05,
β1=0.9, β2=0.999, and ε=1e-8. Starting from m=s=0, at iteration j:

    g = dL/dq, computed by jax.value_and_grad
    m = β1 m + (1-β1) g
    s = β2 s + (1-β2) g²
    m_hat = m/(1-β1^j), s_hat = s/(1-β2^j)
    q_next = project(q - α m_hat/(sqrt(s_hat)+ε)).

The optimizer receives the loss, initial guess, bounds, and optimization
settings, never the truth. Every run saves 801 losses and coefficients,
including the initial and final states. Adam is not required to decrease loss
at every update; recovery is assessed by final loss and parameter error.

## Acceptance criteria declared before recovery

- Matched-model relative parameter error below 1e-4 for every start: a 0.01%
  target is modest for deterministic noiseless fitting of a single scalar.
- Matched-model loss reduction above 1e6: substantial pipeline convergence,
  not merely a step in the right direction.
- Centered-difference versus JAX gradient relative error below 1e-5, leaving
  room for finite-difference cancellation and truncation error.
- Matched sampled-grid minimum within 4% of truth: the 81-point log grid has
  about 6.85% spacing, so this permits its unavoidable coarse location error.
- Interior optima with negative/positive gradients on either side, and histories
  within the validated stable interval.
- Continuum final relative L2 error below 0.1 and loss decrease: a coarse-grid
  field-fit sanity check, not a claim of 10% parameter accuracy. No criterion
  forces continuum parameter recovery to match the true coefficient.
- All old and new tests must pass.

These thresholds were set in the experiment before observing recovery results
and were not loosened. All numerical experiment checks and all tests passed.

## Inverse-loss gradient verification

Centered finite differences use epsilon=1e-5 in q, holding the graph fixed.
Both sides of the minimum are tested against the actual inverse loss.

| Data | q | ν4 | JAX dL/dq | FD dL/dq | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|---:|---:|
| matched | -6.907755279 | 0.001 | -7.287084052090e-04 | -7.287084048945e-04 | 3.14491e-13 | 4.31574e-10 |
| matched | -5.298317367 | 0.005 | 6.188893611135e-03 | 6.188893611351e-03 | 2.15816e-13 | 3.48715e-11 |
| continuum | -6.907755279 | 0.001 | -9.824000041965e-04 | -9.824000039241e-04 | 2.72368e-13 | 2.77248e-10 |
| continuum | -5.298317367 | 0.005 | 5.386520064462e-03 | 5.386520065273e-03 | 8.11113e-13 | 1.50582e-10 |

## Loss landscapes and empirical identifiability

Each dataset is evaluated at the same 81 log-spaced candidates in [1e-4,0.02].
The complete candidate/loss pairs are saved in `stage3_results.json`.
Each sampled curve has exactly one interior minimum. These grid minima are
coarse diagnostics, distinct from optimized minima.

| Data | Sampled minimum ν4 | Loss at sampled minimum | Loss at true ν4 |
|---|---:|---:|---:|
| matched | 0.00196937931719 | 2.945514907e-07 | 1.714670614e-32 |
| continuum | 0.00240224886796 | 1.887144240e-03 | 1.927097053e-03 |

Local probes around the optimized solution use offsets of ±0.01 in log ν4,
roughly ±1% in the coefficient:

| Data | Log offset | ν4 | Loss | dL/dq |
|---|---:|---:|---:|---:|
| matched | -0.01 | 0.0019800996675 | 1.242771906440e-07 | -2.475571385e-05 |
| matched | +0.00 | 0.002 | 2.266036774241e-32 | 2.291492188e-18 |
| matched | +0.01 | 0.00202010033417 | 1.262874248092e-07 | 2.535878690e-05 |
| continuum | -0.01 | 0.00234078290451 | 1.886886049707e-03 | -3.244219352e-05 |
| continuum | +0.00 | 0.00236430816381 | 1.886723210898e-03 | 1.971021112e-18 |
| continuum | +0.01 | 0.00238806985589 | 1.886888579269e-03 | 3.320106439e-05 |

Finite-difference local curvatures in q are approximately 0.00250565 (matched)
and 0.00328207 (continuum). Both losses rise detectably above their minimum
under ±1% coefficient changes, and gradients change from negative to positive.
The matched loss increase is about 1.25e-7, and the continuum increase is
about 1.64e-7 on a nonzero residual floor. There is a resolved minimum, although
these small changes also show why noise sensitivity cannot be inferred from
noiseless success. All four starts converge to the same region for each dataset.
Under this synthetic experiment and observation design, ν4 is empirically
well identified within the chosen numerical model. This is not a formal
statistical identifiability result or evidence about joint parameter recovery.

## Experiment A: matched-model recovery

| Initial ν4 | Recovered ν4 | Absolute parameter error | Relative parameter error | Initial loss | Final loss | Iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.002 | 4.77049e-18 | 2.38524e-15 | 8.213049366e-04 | 2.634823457e-32 | 800 |
| 0.001 | 0.002 | 3.03577e-18 | 1.51788e-15 | 3.464808533e-04 | 2.266036774e-32 | 800 |
| 0.005 | 0.002 | 3.03577e-18 | 1.51788e-15 | 2.134471601e-03 | 2.266036774e-32 | 800 |
| 0.01 | 0.00199999999999999 | 6.50521e-18 | 3.25261e-15 | 1.027810811e-02 | 3.461898859e-32 | 800 |

All starts recover ν4=0.002 to floating-point precision. Loss reduction exceeds
1e28 in all runs, far beyond the required 1e6. These machine-level residuals
should not be interpreted as physical accuracy or meaningful extra digits.
JIT fusion and evaluation order can change the last roundoff-level digits
between a separately evaluated raw MSE and the optimizer's normalized loss.

| Initial ν4 | Observation raw MSE | Final-time relative L2 error |
|---:|---:|---:|
| 0.0005 | 8.378896019e-33 | 1.392341589e-16 |
| 0.001 | 4.000122093e-33 | 9.416282225e-17 |
| 0.005 | 4.000122093e-33 | 9.416282225e-17 |
| 0.01 | 1.846639426e-32 | 2.332214815e-16 |

## Experiment B: continuum-data recovery

The same four starts were also fitted to continuum observations; this exceeds
the single continuum recovery needed for the basic comparison.

| Initial ν4 | Recovered ν4 | Absolute parameter error | Relative parameter error | Initial loss | Final loss | Iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.00236430816381 | 3.643081638e-04 | 1.821540819e-01 | 3.118426981e-03 | 1.886723211e-03 | 800 |
| 0.001 | 0.00236430816381 | 3.643081638e-04 | 1.821540819e-01 | 2.512925738e-03 | 1.886723211e-03 | 800 |
| 0.005 | 0.00236430816381 | 3.643081638e-04 | 1.821540819e-01 | 3.488757299e-03 | 1.886723211e-03 | 800 |
| 0.01 | 0.00236430816381 | 3.643081638e-04 | 1.821540819e-01 | 1.101191111e-02 | 1.886723211e-03 | 800 |

All continuum fits give approximately:

- Recovered ν4: **0.002364308164**, versus truth 0.002.
- Signed bias: **+0.000364308164**, or **+18.2154%**.
- Observation normalized MSE: **0.001886723211**.
- Observation raw MSE: **0.001204588667**.
- Final-time relative L2 field error: **0.07012214956 (7.0122%)**.

This is a substantial numerical-model bias, not exact continuum recovery.
The continuum loss at the true coefficient is 0.001927097053; the biased
coefficient yields a lower loss. It is therefore not appropriate to force the
optimizer to return the true coefficient or hide the residual floor.

At fixed grid spacing the finite differences replace k by sin(k dx)/dx and
k⁴ by 16 sin⁴(k dx/2)/dx⁴. The numerical model has phase and damping errors
that one fitted scalar cannot eliminate. Increasing ν4 can compensate for
weaker discrete damping and can reduce some phase-mismatched residuals by
attenuating those modes. The recovered coefficient belongs to that numerical
model. This experiment quantifies the total discretization-induced bias; it
does not separately measure spatial versus temporal contributions to the bias.

## Tests and reproducibility

Executed from the project root using Python 3.10, JAX/jaxlib 0.6.2, pytest 9.1.1:

    .venv/bin/python -m experiments.stage3_parameter_inference
    .venv/bin/python -m pytest -q

Full pytest summary:

    ...........................                                              [100%]
    27 passed in 5.79s

The 20 existing Stage 1/2 cases pass unchanged. Seven Stage 3 cases cover the
positive transform and round trip, observation shape and initial selection,
finite scalar loss and matched truth minimum, two gradient checks, a 60-update
loss-decreasing recovery, and enforcement of stability bounds. The full
800-update multi-start claims are validated by the experiment, not slow tests.
The JSON records every run's loss/coefficient histories and all numerical checks;
re-running the experiment deterministically overwrites that results file.

## Caveats, conclusion, and next stage

Matched-model synthetic recovery can be artificially easy because data and
predictions share all discretization errors. It validates the differentiable
optimization pipeline, not accuracy of a recovered physical coefficient.
Continuum-generated data provide a more meaningful test of numerical bias;
here they expose an 18.22% upward shift on the deliberately small N=32 grid.

One scalar recovered under full-field noiseless observations does not establish
identifiability of several parameters or recoverability under sparse, noisy
real observations. Loss curvature is not an uncertainty estimate. Grid/time
refinement and observation robustness remain future validation work.
Explicit hyperdiffusion still has its severe dx⁴ timestep restriction. Reverse
mode through many steps and full trajectory allocation can become memory and
compute bottlenecks on larger problems; Stage 3 does not attempt to solve that.

**PASS for the stated Stage 3 pipeline and bias-characterization scope:** all
27 tests pass, multi-start matched recovery is accurate, both landscapes have
resolved interior minima with gradient sign changes, inverse-loss gradients
agree with finite differences, and all optimization histories stay in the stable
range. Continuum fitting is sensible but biased; PASS does not mean its
18.22% coefficient error is acceptable for physical inference.

Recommended Stage 4: robustness to imperfect observations, including noise,
fewer observation times, partial spatial observations, possible simultaneous
inference of v and ν4, and uncertainty/sensitivity studies. Stage 4 is not
implemented.

Git status:

    fatal: not a git repository (or any of the parent directories): .git

No Git repository was initialized. No commit, push, or history modification was made.
