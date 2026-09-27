# Stage 4 refinement validation

## Objective and relationship to Stage 3

Stage 3 fitted noiseless continuum observations with a coarse N=32 finite-
difference model and recovered ν4≈0.002364308164 instead of 0.002: a +18.22%
effective-coefficient bias. Stage 4 separates temporal from spatial error and
asks whether the inferred coefficient approaches continuum truth under refinement.
The earlier result and its interpretation are preserved.

Created `experiments/stage4_refinement_study.py`, `tests/test_refinement.py`,
`docs/stage4_results.json`, and this report. Updated `README.md` only among
existing project files. Operators, solver, inverse utilities, earlier tests,
and earlier validation reports/results are unchanged. No new dependency was
added. There is no second solver or inverse optimizer.

## Continuum setup and fixed inverse problem

Domain [0,2π), ν4_true=0.002, known v=1, T=2, and

    ψ0(x) = sin(x) + 0.5 sin(2x) + 0.25 cos(3x).

Nine full-field observations occur at t=0,0.25,...,2. The Stage 3 analytical
continuum generator evaluates each mode with exp(-ν4 k⁴ t) damping and phase
k(x-vt). Continuum truth has no finite-difference error. Each spatial grid
samples the same physical field at its own points. Data remain noiseless.

All fits call the Stage 3 normalized MSE and the unchanged Stage 2 RK4/scan
solver. The loss is mean((prediction-observation)²)/(mean(observation²)+1e-12).
Optimize q=log(ν4) using exactly the Stage 3 projected Adam: 800 updates,
learning rate 0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, coefficient bounds
[1e-4,0.02]. Primary starts are 0.0005; selected alternate starts are 0.01.
Within each fit, dt, step count, grid, and observation indices stay fixed.
No optimizer setting was tuned per resolution.

## Declared design and acceptance checks

Before seeing recovered values, we selected temporal steps 256,512,1024,2048
at N=32; spatial grids 16,24,32,40,48; and pure-diffusion grids 16,32,48.
The practical N≤48 list was chosen from the outset because the stability-limited
step count scales approximately as N⁴; N=64 would need about 4952 steps under
this policy. No grid was silently removed or tolerance loosened after execution.

Baseline and multi-start coefficient agreement tolerance is 1e-10 absolute,
far below the Stage 3 bias and prospective spatial changes. Stationarity requires
|dL/dq|<1e-10 and opposite gradient signs at q±1e-4. Local Newton correction
estimates must be below 0.1% of the smallest adjacent spatial coefficient change.
Direct dt-halving controls must change the recovered coefficient by less than
0.1% of the remaining parameter bias. These checks were declared before the
runs. No predetermined bias convergence order is an acceptance requirement.
All old/new tests and stable optimized solutions are required; lack of a trend
toward truth would require NEEDS ATTENTION.

## Study A: temporal refinement at fixed N=32

The first row exactly reproduces the Stage 3 256-step schedule. Each subsequent
row doubles steps, keeping all nine observation times and T exact. The stability
limit is evaluated at the search upper bound ν4=0.02, not at the current candidate.

| Steps | dt | dt/dt_max | Fitted ν4 | Signed bias | Absolute error | Relative error | Final normalized MSE | Final relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.0078125 | 0.603878 | 0.00236430816380558 | +3.6430816381e-04 | 3.6430816381e-04 | 18.215408190% | 1.886723210898e-03 | 7.0122149565e-02 |
| 512 | 0.00390625 | 0.301939 | 0.00236430812816429 | +3.6430812816e-04 | 3.6430812816e-04 | 18.215406408% | 1.886723119601e-03 | 7.0122147916e-02 |
| 1024 | 0.001953125 | 0.150970 | 0.00236430812589246 | +3.6430812589e-04 | 3.6430812589e-04 | 18.215406295% | 1.886723113911e-03 | 7.0122147814e-02 |
| 2048 | 0.0009765625 | 0.075485 | 0.00236430812574912 | +3.6430812575e-04 | 3.6430812575e-04 | 18.215406287% | 1.886723113556e-03 | 7.0122147807e-02 |

From the first to finest timestep, the coefficient changes by 3.805646e-11, only 1.044623e-07 of the coarse-grid bias. The +18.22% bias is essentially unchanged. Temporal integration error is not its dominant cause.

For an estimate of temporal behavior, e_dt below is the difference from the
finest fitted coefficient, not the difference from continuum truth:

| Steps | Difference from finest coefficient | log2 ratio to next difference |
|---:|---:|---:|
| 256 | 3.805645859e-11 | 3.97794 |
| 512 | 2.415176999e-12 | 4.07463 |
| 1024 | 1.433388998e-13 | — |
| 2048 | 0.000000000e+00 | — |

These ratios are diagnostic only: subtracting a finite-resolution reference
changes the ratios, especially for the penultimate row, and coefficient
differences are tiny. The last row is zero by definition. We do not assign an
order to that row or assume an unlimited fourth-order asymptotic regime.
The reliable scientific conclusion is negligible temporal influence on the
observed spatial bias; optimizer precision is assessed separately below.

## Study B: spatial refinement and timestep policy

For each N, estimate dt_max with the Stage 2 8193-angle RK4 diagnostic at
ν4_max=0.02. Set target dt=0.5 dt_max. Round the number of steps upward to a
multiple of eight so that T=2 and all observation times are represented.
Actual dt is 2/num_steps, never adjusted during optimization.

For every run, the Stage 3 supplemental diagnostic samples 101 coefficients
across the entire search interval and 8193 angles. All resolvable angles for
these even grids lie in [0,π]; the dense-angle diagnostic is a numerical estimate,
not an analytical continuous-interval stability proof. Projected Adam keeps
candidates inside the prevalidated coefficient bounds. Direct dt-halving fits
at N=16 and N=48 further check that time error is small relative to spatial bias.

| N | dx | Steps | dt | dt/dt_max | Fitted ν4 | Signed bias | Relative bias | Final normalized MSE | Final relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.39269908 | 24 | 8.333333333e-02 | 0.402586 | 0.00672667321799 | +4.726673218e-03 | +236.333661% | 2.341069755e-02 | 2.365494503e-01 |
| 24 | 0.26179939 | 104 | 1.923076923e-02 | 0.470328 | 0.00301938956468 | +1.019389565e-03 | +50.969478% | 5.669451608e-03 | 1.206383251e-01 |
| 32 | 0.19634954 | 312 | 6.410256410e-03 | 0.495490 | 0.00236430814311 | +3.643081431e-04 | +18.215407% | 1.886723158e-03 | 7.012214860e-02 |
| 40 | 0.15707963 | 760 | 2.631578947e-03 | 0.496611 | 0.00217236641174 | +1.723664117e-04 | +8.618321% | 7.870184503e-04 | 4.538642998e-02 |
| 48 | 0.13089969 | 1568 | 1.275510204e-03 | 0.499124 | 0.00209697912051 | +9.697912051e-05 | +4.848956% | 3.827458279e-04 | 3.167539132e-02 |

Absolute bias equals signed bias for these positive-bias fits; relative error
is the absolute value of the reported relative bias. The recovered coefficient
approaches truth as N increases, while normalized MSE and final-field error
also decrease. This supports convergence on the tested grid range; it is not
a proof of an asymptotic order as dx tends to zero.

Empirical adjacent-grid orders use log(error_coarse/error_fine)/log(dx_coarse/dx_fine).
The parameter error is |ν4_fit-ν4_true|; the field error is final relative L2.

| Grid pair | Parameter-bias order | Field-error order |
|---|---:|---:|
| 16 → 24 | 3.78335 | 1.66071 |
| 24 → 32 | 3.57672 | 1.88596 |
| 32 → 40 | 3.35379 | 1.94953 |
| 40 → 48 | 3.15446 | 1.97274 |

These are measured effective orders, not an assumption that inferred-parameter
bias equals the second-order spatial truncation error. The multimode minimizer
combines damping and phase mismatch nonlinearly. The finite grid range does not
justify assigning a universal parameter-convergence order.

Direct temporal controls for the spatial policy:

| N | Base steps | Doubled steps | Absolute coefficient change | Change / spatial bias |
|---:|---:|---:|---:|---:|
| 16 | 24 | 48 | 7.421588526e-07 | 1.570150544e-04 |
| 48 | 1568 | 3136 | 1.977584763e-14 | 2.039186118e-10 |

## Pure hyperdiffusion and phase-mismatch diagnostic

Set v=0 in both continuum truth and fitting, retaining every other physical
and inverse setting. This changes the truth to stationary-phase diffusion;
it does not fit moving observations using a zero-velocity model. The same
systematic spatial timestep policy is used.

| N | Steps | Fitted ν4 (v=0) | Pure signed bias | Pure relative bias | Combined relative bias | Pure normalized MSE | Pure final relative L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 24 | 0.00245445589697 | +4.544558970e-04 | +22.722795% | +236.333661% | 3.999214859e-06 | 3.316679337e-03 |
| 32 | 312 | 0.00210570992243 | +1.057099224e-04 | +5.285496% | +18.215407% | 2.300891498e-07 | 7.970503873e-04 |
| 48 | 1568 | 0.00204635628007 | +4.635628007e-05 | +2.317814% | +4.848956% | 4.474906257e-08 | 3.516225388e-04 |

Both problems have positive bias that decreases with refinement, but the
combined problem has additional bias. This is consistent with compensation for
advection phase mismatch: reducing mismatched mode amplitudes can reduce
field-space error. It is a controlled comparison, not an additive or formal
decomposition of two independent biases.

## Modified-wavenumber interpretation

For each mode, k4_discrete=16 sin⁴(k dx/2)/dx⁴. A single-mode rate match would
require ν4_effective=ν4_true k⁴/k4_discrete. These numbers are interpretive;
no single scalar matches all three modes' decay rates at finite dx.

| N | k | Continuum k⁴ | Discrete k4 | Single-mode effective ν4 |
|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 0.974593122 | 0.00205213843057 |
| 16 | 2 | 16 | 14.429094648 | 0.00221774136086 |
| 16 | 3 | 81 | 64.096794976 | 0.00252742746437 |
| 32 | 1 | 1 | 0.993593023 | 0.00201289658122 |
| 32 | 2 | 16 | 15.593489953 | 0.00205213843057 |
| 32 | 3 | 81 | 76.435809769 | 0.00211942544324 |
| 48 | 1 | 1 | 0.997147879 | 0.00200572055802 |
| 48 | 2 | 16 | 15.818166175 | 0.00202299050638 |
| 48 | 3 | 81 | 78.942042889 | 0.00205213843057 |

Discrete k4 is below k⁴, so matching a continuum decay rate favors a coefficient
above truth. Values approach truth with refinement. The optimized multimode
coefficient is a loss-weighted compromise and need not equal any single-mode
value. With advection, the modified first-derivative wavenumber sin(k dx)/dx
also changes phase and further affects that compromise.

## Optimizer convergence and Stage 3 reproduction

For each main spatial run, evaluate g=dL/dq and gradients at q±h, h=1e-4.
Estimate curvature H=[g(q+h)-g(q-h)]/(2h) and a local coefficient correction
ν4 |g/H|. This is a local stationarity diagnostic, not a certified error bound.

| N | Final abs(g) | Local curvature | Estimated coefficient correction | Left gradient | Right gradient |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.550878e-18 | 1.208374e-02 | 2.533343e-18 | -1.208272e-06 | 1.208477e-06 |
| 24 | 1.111724e-18 | 4.524759e-03 | 7.418581e-19 | -4.524263e-07 | 4.525254e-07 |
| 32 | 4.998607e-21 | 3.281981e-03 | 3.600949e-21 | -3.281601e-07 | 3.282360e-07 |
| 40 | 2.542161e-19 | 2.949667e-03 | 1.872248e-19 | -2.949320e-07 | 2.950013e-07 |
| 48 | 3.916455e-19 | 2.830525e-03 | 2.901485e-19 | -2.830191e-07 | 2.830859e-07 |

Alternate starts use 0.01 versus the main 0.0005 at the same controlled timestep:

| N | Main fitted ν4 | Alternate fitted ν4 | Absolute difference |
|---:|---:|---:|---:|
| 16 | 0.00672667321799362 | 0.00672667321799362 | 0.000000e+00 |
| 32 | 0.00236430814311219 | 0.00236430814311218 | 8.673617e-18 |
| 48 | 0.00209697912050701 | 0.00209697912050702 | 5.637851e-18 |

The exact N=32, 256-step Stage 3 baseline reproduces 0.002364308163805576; the stored Stage 3 value is 0.002364308163805576. Absolute difference: 0.000000e+00, against a predeclared tolerance of 1e-10.

Across all 17 fits, maximum final gradient magnitude is 1.084831e-17; boundary hits total 0. All local probes bracket minima, and maximum sampled RK4 amplification is 1, including the neutral constant mode.

The optimizer evidence is much more precise than the spatial trends. Very small
temporal coefficient differences should still be interpreted cautiously because
floating-point effects are not represented by a local Newton correction alone.

## Explicit hyperdiffusion cost

| N | Spatial-study steps | Main solve seconds, including compilation |
|---:|---:|---:|
| 16 | 24 | 3.03 |
| 24 | 104 | 2.92 |
| 32 | 312 | 5.46 |
| 40 | 760 | 11.87 |
| 48 | 1568 | 22.98 |

Halving dx can require about 16 times as many timesteps for explicit fourth-
order diffusion. Wall time also depends on N, hardware, JIT compilation,
reverse-mode work, and 800 optimizer updates; these rough timings are not
performance benchmarks. The implementation clears compiled caches after each
completed fit to avoid retaining many shape-specific reverse-mode executables.
Full trajectories and differentiation still carry memory cost. No scalable
adjoint or alternative integrator is introduced in this stage.

## Execution, caveats, and conclusion

Executed:

    .venv/bin/python -m pytest -q
    .venv/bin/python -m experiments.stage4_refinement_study

Full pytest summary:

    34 passed in 10.08s

All 27 previous cases pass unchanged. Seven new cases cover continuum shape
and initial state, observation/step alignment and safety, signed/absolute/relative
bias metrics, pure-diffusion phase, modified wavenumbers, and a short two-grid
finite/stable recovery smoke check. Full refinement is not run inside pytest.
Detailed numerical results, histories, local probes, and acceptance checks are
stored in `docs/stage4_results.json`. Rerunning the experiment overwrites only
that Stage 4 results file and reports IN PROGRESS until the study completes.

Scientific distinctions remain important: forward-solution convergence does not
automatically dictate parameter convergence. A finite-difference model can fit
continuum data best with a biased effective coefficient. The Stage 3 result
came from a deliberately coarse grid; temporal and spatial effects must remain
separate. A multimode coefficient compromises among different damping errors,
and phase mismatch interacts with that fit. Explicit refinement is expensive.
This remains a noiseless full-field experiment, not evidence of robust inference
from real, sparse, or noisy measurements.

**PASS**: the baseline is reproduced, temporal refinement leaves the coarse-grid bias essentially unchanged, spatial refinement reduces parameter and field errors, optimizer accuracy is finer than the reported spatial trend, and pure-diffusion/modified-wavenumber diagnostics support the mechanism interpretation. All numerical checks and all tests pass. Residual fine-grid bias remains; the study supports a trend, not exact continuum recovery at N=48.

Recommended Stage 5: inverse robustness under imperfect observations—additive
noise, fewer observation times, partial spatial sensors, recovery degradation
curves, and possible uncertainty/sensitivity analysis. Joint v and ν4 inference
is not recommended for implementation without a later explicit request.
Stage 5 is not implemented.

Git status:

    fatal: not a git repository (or any of the parent directories): .git

No repository was initialized, and no commit, push, or history change was made.
