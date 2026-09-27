# Stage 11: Exact Semi-Discrete Exponential Propagation, IMEX Integration, and Differentiable Scalability

**Status: PASS**

## Objective and preserved scope

This study isolates time integration of the existing linear periodic finite-difference (FD) model. Spatial stencils, discrepancy definitions, M5 calibration, sensor layouts, parameter bounds, Adam settings and Stage 1–10 results are unchanged. No Stage 10 uncertainty study is repeated. No Stage 12 implementation, Git initialization, commit or push is performed.

Created `src/integrators.py`, `tests/test_integrators.py`, `experiments/stage11_scalable_integration.py`, `experiments/stage11_report.py`, `docs/stage11_results.json` and this report. README has a concise addition. SHA-256 hashes of all earlier source, tests, experiment scripts and documentation are embedded in the results file and checked on every resume.

## Exactness and canonical spatial symbols

Exact means exact time evolution of the semi-discrete FD equations, up to floating-point error. It does not mean exact continuum physics. All Fourier methods reuse the Stage 9 `operator_symbols` / `discrepancy_symbol`, with θ = 2π fftfreq(N), float64 and complex128. No continuum ik or k⁴ symbols replace these operators.

D₁ = i sinθ/dx; D₃ = −4i sinθ sin²(θ/2)/dx³; D₄ = 16 sin⁴(θ/2)/dx⁴; D₆ = −64 sin⁶(θ/2)/dx⁶.

λ = −vD₁ − ν₄D₄ + c₃v dx²D₃/6 + c₆ν₄ dx²D₆/6.

## Fourier versus stencil RHS

Deterministic random smooth fields (seed 1101, modes 1–5), v=.87, ν₄=.003; M0=(0,0), M1=(1,1), M5=(1.0562544646126, 1.0618708313445455).

| N | model | max absolute | relative L2 |
| --- | --- | --- | --- |
| 16 | M0 | 1.776357e-15 | 2.297424e-16 |
| 16 | M1 | 2.664535e-15 | 2.584856e-16 |
| 16 | M5 | 3.552714e-15 | 2.999514e-16 |
| 32 | M0 | 1.24345e-14 | 9.526803e-16 |
| 32 | M1 | 1.687539e-14 | 1.261949e-15 |
| 32 | M5 | 1.865175e-14 | 1.242993e-15 |
| 64 | M0 | 2.779998e-13 | 2.353551e-14 |
| 64 | M1 | 4.893863e-13 | 3.652036e-14 |
| 64 | M5 | 5.018208e-13 | 3.732563e-14 |

## Exact exponential and semigroup

The requested-time solution is IFFT(exp(tλ) FFT(ψ₀)). It accepts arbitrary nonnegative times, with no internal steps or dense internal trajectory. At t=0, the public real-valued API returns ψ₀ bit-for-bit. `exponential_final` requests one time. Complex output is available separately for auditing imaginary roundoff.

Zero-time bitwise equality: True; shape: [4, 32]; maximum discarded imaginary part: 1.387779e-16.

| t₁ | t₂ | semigroup max error |
| --- | --- | --- |
| 0.137 | 0.831 | 4.440892e-16 |
| 0.3 | 1.7 | 2.220446e-16 |
| 0 | 0.61 | 0 |

## Stage 9 RK4-power and independent scan comparison

The archived Fourier evaluator computes R(dtλ)ⁿ, R(z)=1+z+z²/2+z³/6+z⁴/24. This is algebraically the same fixed-step RK4 scheme as the stencil/lax.scan baseline. It avoids scanning steps but retains RK4 stability and temporal error. Only integer step times are used; continuous-time semigroup accuracy is not claimed.

| N | model | scan/power max error | relative L2 |
| --- | --- | --- | --- |
| 16 | M0 | 3.064216e-14 | 5.504316e-15 |
| 16 | M1 | 3.28626e-14 | 4.148185e-15 |
| 16 | M5 | 2.120526e-14 | 3.607893e-15 |
| 32 | M0 | 3.064216e-14 | 5.463922e-15 |
| 32 | M1 | 2.531308e-14 | 4.334064e-15 |
| 32 | M5 | 4.218847e-14 | 7.250345e-15 |
| 64 | M0 | 2.842171e-14 | 3.907806e-15 |
| 64 | M1 | 2.076117e-14 | 3.270044e-15 |
| 64 | M5 | 4.751755e-14 | 6.747367e-15 |

Discrete semigroup max error: 2.220446e-16.

## Crank–Nicolson and IMEX recurrence

Full CN has amplification g=(1+dtλ/2)/(1−dtλ/2). Both actual Fourier-state scan stepping and a direct-power requested-time evaluator are provided. The inverse study uses CN direct powers; runtime tables labeled CN use actual iteration. These are the same linear CN discretization.

Split E=−vD₁+c₃v dx²D₃/6 (purely imaginary) and S=−ν₄D₄+c₆ν₄ dx²D₆/6 (real nonpositive in the positive coefficient box). CNAB2 uses one full-operator CN step for second-order startup, then (1−dtS/2)uₙ₊₁=(1+dtS/2)uₙ+dtE(3uₙ/2−uₙ₋₁/2). The implementation evaluates this actual recurrence and supports full or requested outputs. It is differentiated through JAX scan.

Substitute uₙ=zⁿ and divide by zⁿ⁻¹: (1−dtS/2)z²−(1+dtS/2+3dtE/2)z+dtE/2=0. Both quadratic roots must have magnitude at most 1+10⁻¹². A separate scalar test initializes each root as u₁ and verifies five direct recurrence updates. The zero mode has roots 1 and 0.

Scalar root/recurrence max discrepancy: 5.551115e-17.

## Numerical box stability and spatial scaling

Each grid uses five samples per coordinate: v∈[.5,1.5], ν₄∈[10⁻⁴,.02] (geometric), c₃,c₆∈[0,2], totaling 625 parameter tuples. Angles include 2049 evenly spaced points on [0,π] and every actual nonnegative grid mode. Forty bisections on [0,1] estimate the first stable interval boundary; the criterion includes both roots. This is numerical sampling, not a proof over all continuous parameters. S has maximum real part zero and E has zero real part throughout the sampled box.

| N | dx | RK4 dtmax | CNAB2 dtmax | ratio | RK4 steps T=2 | CNAB2 steps T=2 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.3926991 | 0.08871228 | 0.03353944 | 0.3780699 | 23 | 60 |
| 24 | 0.2617994 | 0.01752341 | 0.03285371 | 1.874846 | 115 | 61 |
| 32 | 0.1963495 | 0.005544518 | 0.03193137 | 5.759088 | 361 | 63 |
| 48 | 0.1308997 | 0.001095213 | 0.02958233 | 27.01056 | 1827 | 68 |
| 64 | 0.09817477 | 0.0003465324 | 0.02702311 | 77.9815 | 5772 | 75 |
| 96 | 0.06544985 | 6.845083e-05 | 0.02303864 | 336.5721 | 29219 | 87 |
| 128 | 0.04908739 | 2.165827e-05 | 0.02127212 | 982.1708 | 92344 | 95 |

All-grid empirical slopes of log(dtmax) versus log(dx): RK4 4, CNAB2 0.2305109. No expected exponent was imposed. The CNAB2 limiting sampled tuple and angle are recorded per grid. A smaller stiff diffusion restriction does not imply a universally larger allowable timestep: weak diffusion and explicit phase motion can control CNAB2.

## Forward temporal convergence

Initial field A, N=32, T=2, v=1, ν₄=.002. All errors are against the exact solution of the same FD model. Here RK4 steps start at 128, stable for these fixed representative parameters; this is not the full-box production policy. Order is log₂(error at previous step count / error at current count).

| model | method | steps | dt | max error | relative L2 | order |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | rk4 | 128 | 0.015625 | 4.663809e-08 | 3.222697e-08 | — |
| M0 | rk4 | 256 | 0.0078125 | 2.912308e-09 | 2.012422e-09 | 4.001264 |
| M0 | rk4 | 512 | 0.00390625 | 1.819274e-10 | 1.25723e-10 | 4.000613 |
| M0 | rk4 | 1024 | 0.001953125 | 1.134004e-11 | 7.85973e-12 | 3.999625 |
| M0 | rk4 | 2048 | 0.0009765625 | 7.556178e-13 | 4.845443e-13 | 4.019779 |
| M0 | cn | 128 | 0.015625 | 0.0003127277 | 0.0002015986 | — |
| M0 | cn | 256 | 0.0078125 | 7.819355e-05 | 5.040638e-05 | 1.999807 |
| M0 | cn | 512 | 0.00390625 | 1.954911e-05 | 1.260202e-05 | 1.999952 |
| M0 | cn | 1024 | 0.001953125 | 4.887323e-06 | 3.15053e-06 | 1.999988 |
| M0 | cn | 2048 | 0.0009765625 | 1.221834e-06 | 7.876342e-07 | 1.999997 |
| M0 | cnab2 | 128 | 0.015625 | 0.001540538 | 0.001000113 | — |
| M0 | cnab2 | 256 | 0.0078125 | 0.0003870531 | 0.0002508584 | 1.995219 |
| M0 | cnab2 | 512 | 0.00390625 | 9.702268e-05 | 6.283161e-05 | 1.997311 |
| M0 | cnab2 | 1024 | 0.001953125 | 2.428927e-05 | 1.572334e-05 | 1.998583 |
| M0 | cnab2 | 2048 | 0.0009765625 | 6.076588e-06 | 3.932814e-06 | 1.999274 |
| M1 | rk4 | 128 | 0.015625 | 5.588963e-08 | 4.101269e-08 | — |
| M1 | rk4 | 256 | 0.0078125 | 3.504939e-09 | 2.56087e-09 | 4.001364 |
| M1 | rk4 | 512 | 0.00390625 | 2.194964e-10 | 1.599793e-10 | 4.000677 |
| M1 | rk4 | 1024 | 0.001953125 | 1.366085e-11 | 9.992058e-12 | 4.00096 |
| M1 | rk4 | 2048 | 0.0009765625 | 8.872902e-13 | 6.328682e-13 | 3.980805 |
| M1 | cn | 128 | 0.015625 | 0.0003443801 | 0.0002264231 | — |
| M1 | cn | 256 | 0.0078125 | 8.61019e-05 | 5.661414e-05 | 1.999787 |
| M1 | cn | 512 | 0.00390625 | 2.15259e-05 | 1.415406e-05 | 1.999947 |
| M1 | cn | 1024 | 0.001953125 | 5.381503e-06 | 3.538547e-06 | 1.999987 |
| M1 | cn | 2048 | 0.0009765625 | 1.345377e-06 | 8.846387e-07 | 1.999997 |
| M1 | cnab2 | 128 | 0.015625 | 0.001726911 | 0.001123519 | — |
| M1 | cnab2 | 256 | 0.0078125 | 0.0004327153 | 0.000281766 | 1.995454 |
| M1 | cnab2 | 512 | 0.00390625 | 0.0001083172 | 7.056912e-05 | 1.997389 |
| M1 | cnab2 | 1024 | 0.001953125 | 2.709744e-05 | 1.765925e-05 | 1.998613 |
| M1 | cnab2 | 2048 | 0.0009765625 | 6.776674e-06 | 4.416998e-06 | 1.999286 |
| M5 | rk4 | 128 | 0.015625 | 5.675689e-08 | 4.155724e-08 | — |
| M5 | rk4 | 256 | 0.0078125 | 3.560772e-09 | 2.59486e-09 | 4.001371 |
| M5 | rk4 | 512 | 0.00390625 | 2.228444e-10 | 1.621006e-10 | 4.000695 |
| M5 | rk4 | 1024 | 0.001953125 | 1.394818e-11 | 1.012386e-11 | 4.001058 |
| M5 | rk4 | 2048 | 0.0009765625 | 9.897638e-13 | 6.584455e-13 | 3.942552 |
| M5 | cn | 128 | 0.015625 | 0.0003455388 | 0.0002278734 | — |
| M5 | cn | 256 | 0.0078125 | 8.639118e-05 | 5.69768e-05 | 1.999786 |
| M5 | cn | 512 | 0.00390625 | 2.15982e-05 | 1.424473e-05 | 1.999947 |
| M5 | cn | 1024 | 0.001953125 | 5.399575e-06 | 3.561215e-06 | 1.999987 |
| M5 | cn | 2048 | 0.0009765625 | 1.349895e-06 | 8.903058e-07 | 1.999997 |
| M5 | cnab2 | 128 | 0.015625 | 0.001734552 | 0.001130729 | — |
| M5 | cnab2 | 256 | 0.0078125 | 0.0004345607 | 0.0002835717 | 1.995468 |
| M5 | cnab2 | 512 | 0.00390625 | 0.00010877 | 7.102114e-05 | 1.997393 |
| M5 | cnab2 | 1024 | 0.001953125 | 2.720953e-05 | 1.777235e-05 | 1.998614 |
| M5 | cnab2 | 2048 | 0.0009765625 | 6.804558e-06 | 4.445283e-06 | 1.999287 |

## Stiff damping: stability is not fidelity

For negative-real λ and a=dt|λ|, CN and the E=0 CNAB2 recurrence share (1−a/2)/(1+a/2). Negative signs imply alternating states. Their magnitudes approach one for very stiff modes. CN is A-stable, not L-stable. Backward Euler, included only as a scalar diagnostic, approaches zero and is L-stable; its generic accuracy is first order.

| a | exact | CN signed | CNAB2 E=0 signed | BE |
| --- | --- | --- | --- | --- |
| 1 | 0.3678794 | 0.3333333 | 0.3333333 | 0.5 |
| 10 | 4.539993e-05 | -0.6666667 | -0.6666667 | 0.09090909 |
| 100 | 3.720076e-44 | -0.9607843 | -0.9607843 | 0.00990099 |
| 1000 | 0 | -0.996008 | -0.996008 | 0.000999001 |

## Parameter gradients

The clean continuum-observation NMSE objective is differentiated at (v,q,c₃,c₆)=(.9,log(.0015),.7,1.3), N=32, nine times. Centered finite differences use h=10⁻⁵ in each coordinate.

| coordinate | exact AD | centered FD | absolute error | relative error |
| --- | --- | --- | --- | --- |
| v | -0.5223437 | -0.5223437 | 8.296774e-11 | 1.588374e-10 |
| q | -0.002024088 | -0.002024088 | 8.136976e-13 | 4.02007e-10 |
| c3 | -0.01028747 | -0.01028747 | 1.634075e-13 | 1.588413e-11 |
| c6 | -8.968694e-05 | -8.968694e-05 | 6.046542e-13 | 6.741831e-09 |

| method | steps | gradient relative error | observed order |
| --- | --- | --- | --- |
| rk4 | 128 | 1.552414e-08 | — |
| rk4 | 256 | 9.773605e-10 | 3.989479 |
| rk4 | 512 | 6.130514e-11 | 3.994811 |
| rk4 | 1024 | 3.801211e-12 | 4.011477 |
| cn | 128 | 0.0002712664 | — |
| cn | 256 | 6.78483e-05 | 1.999326 |
| cn | 512 | 1.696406e-05 | 1.999831 |
| cn | 1024 | 4.241138e-06 | 1.999958 |
| cnab2 | 128 | 0.001309871 | — |
| cnab2 | 256 | 0.0003302739 | 1.98769 |
| cnab2 | 512 | 8.296541e-05 | 1.993081 |
| cnab2 | 1024 | 2.079392e-05 | 1.996348 |

## Initial-state derivatives

JVP of the scalar objective along cos(3x)+.2sin(5x), compared with centered directional differences h=10⁻⁵. CNAB2 uses 512 updates.

| method | JVP | FD | absolute error |
| --- | --- | --- | --- |
| exact | 0.03216006 | 0.03216006 | 1.86224e-12 |
| cnab2 | 0.03214313 | 0.03214313 | 1.013585e-12 |

## Mean conservation and dissipation

N=32, 33 output times, ψ₀=A+.23; approximate methods use 1024 updates. Negative maximum energy increment indicates every sampled interval dissipates. This is a sampled trajectory check, not a theorem about all CNAB2 states.

| case | method | max mean drift | max energy increment | final/initial energy |
| --- | --- | --- | --- | --- |
| diffusion | exact | 1.665335e-16 | -0.02820207 | 0.9535864 |
| diffusion | rk4 | 1.665335e-16 | -0.02820207 | 0.9535864 |
| diffusion | cn | 1.665335e-16 | -0.02820207 | 0.9535864 |
| diffusion | cnab2 | 1.665335e-16 | -0.02820207 | 0.9535864 |
| corrected_diffusion | exact | 1.665335e-16 | -0.02875353 | 0.9522517 |
| corrected_diffusion | rk4 | 1.665335e-16 | -0.02875353 | 0.9522517 |
| corrected_diffusion | cn | 1.665335e-16 | -0.02875353 | 0.9522517 |
| corrected_diffusion | cnab2 | 1.665335e-16 | -0.02875353 | 0.9522517 |
| combined | exact | 1.665335e-16 | -0.0287867 | 0.9521701 |
| combined | rk4 | 1.665335e-16 | -0.0287867 | 0.9521701 |
| combined | cn | 1.665335e-16 | -0.02878662 | 0.9521703 |
| combined | cnab2 | 1.665335e-16 | -0.02878694 | 0.9521695 |

## Focused clean inverse equivalence

All methods use the identical continuum A observations, physical truth (1,.002), N=32, fixed M1 or M5, and nine full-grid output times. M0 is additionally fit with RK4 and exact. RK4 uses the archived 1024-step schedule; CN/CNAB2 use 512 steps (dt=.00390625), declared before production. Every fit starts at (.9,log(.0015)), with Adam lr=.05, β₁=.9, β₂=.999, ε=10⁻⁸ and original bounds. After exactly 2000 updates, only fits whose projected NMSE-gradient/KKT residual is ≥10⁻⁷ continue the same state to 4000. No restart or optimizer retuning.

| model | method | v | nu4 | NMSE | final-field rel L2 | Δv vs exact | Δnu vs exact | KKT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | rk4 | 1.021861 | 0.002201215 | 0.0007666544 | 0.04499283 | 1.74305e-12 | 1.199479e-13 | 6.112537e-16 |
| M0 | exact | 1.021861 | 0.002201215 | 0.0007666544 | 0.04499283 | 0 | 0 | 4.250112e-16 |
| M1 | rk4 | 1.00092 | 0.002006258 | 4.098049e-06 | 0.003258612 | 2.242428e-12 | 8.379972e-14 | 4.503791e-16 |
| M1 | exact | 1.00092 | 0.002006258 | 4.098049e-06 | 0.003258612 | 0 | 0 | 7.954927e-16 |
| M1 | cn | 1.000924 | 0.002006322 | 4.119061e-06 | 0.003267031 | 4.313483e-06 | 6.416934e-08 | 1.579497e-15 |
| M1 | cnab2 | 1.000898 | 0.002006075 | 3.993263e-06 | 0.00321626 | -2.157236e-05 | -1.821992e-07 | 2.126796e-15 |
| M5 | rk4 | 0.9997471 | 0.001999859 | 6.077613e-07 | 0.001252235 | 2.26974e-12 | 8.289116e-14 | 1.199401e-15 |
| M5 | exact | 0.9997471 | 0.001999859 | 6.077613e-07 | 0.001252235 | 0 | 0 | 5.873714e-16 |
| M5 | cn | 0.9997515 | 0.001999921 | 6.135286e-07 | 0.001257999 | 4.330107e-06 | 6.224877e-08 | 6.373458e-16 |
| M5 | cnab2 | 0.9997255 | 0.001999686 | 5.795503e-07 | 0.00122366 | -2.165588e-05 | -1.732816e-07 | 7.572686e-17 |

Optimizer summary: 136 total fits; 0 continuations; maximum final KKT 1.744659e-13.

## Sparse noisy paired comparisons

Frozen sensors [9,11,13,19,22,27,29,31], times [0,1.75,2], 2% noise, fresh seeds 1000–1019. Each noisy array is generated once per design/seed and reused across every model and method. SHA-256 hashes are stored. Each model/integrator also has its own zero-noise sparse pseudo-optimum. All 20 paired results and physical errors, shifts relative to the same-seed exact result, and deviations from the integrator’s own clean pseudo-optimum are retained.

| model | method | clean sparse v | clean sparse nu4 | mean Δv vs exact | std Δv | max |Δv| | mean Δnu | std Δnu | max |Δnu| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | rk4 | 1.000376 | 0.002009357 | 8.431145e-13 | 5.356277e-14 | 9.636736e-13 | 9.699151e-14 | 9.95716e-15 | 1.111585e-13 |
| M1 | exact | 1.000376 | 0.002009357 | 0 | 0 | 0 | 0 | 0 | 0 |
| M1 | cnab2 | 1.000362 | 0.002009155 | -1.473967e-05 | 2.660083e-07 | 1.536432e-05 | -2.085116e-07 | 3.219479e-08 | 2.653204e-07 |
| M5 | rk4 | 0.9995511 | 0.002002532 | 8.771039e-13 | 5.388759e-14 | 9.836576e-13 | 9.226279e-14 | 9.83305e-15 | 1.072428e-13 |
| M5 | exact | 0.9995511 | 0.002002532 | 0 | 0 | 0 | 0 | 0 | 0 |
| M5 | cnab2 | 0.9995361 | 0.002002353 | -1.487955e-05 | 2.671829e-07 | 1.550465e-05 | -1.847961e-07 | 3.214327e-08 | 2.423547e-07 |

## Pseudo-truth decomposition

Physical error = clean integrator/model/design bias + noisy shift about that integrator’s own pseudo-optimum. Paired differences relative to exact isolate the numerical time-integrator contribution, rather than attributing existing spatial/model bias to integration.

| model | method | mean physical error (v,nu) | std physical error | mean own-clean shift | std own-clean shift |
| --- | --- | --- | --- | --- | --- |
| M1 | rk4 | 0.0001019455, 4.543862e-05 | 0.001561825, 0.0001544483 | -0.0002744832, 3.608157e-05 | 0.001561825, 0.0001544483 |
| M1 | exact | 0.0001019455, 4.543862e-05 | 0.001561825, 0.0001544483 | -0.0002744832, 3.608157e-05 | 0.001561825, 0.0001544483 |
| M1 | cnab2 | 8.720578e-05, 4.523011e-05 | 0.001561671, 0.0001544293 | -0.0002744126, 3.607542e-05 | 0.001561671, 0.0001544293 |
| M5 | rk4 | -0.0007206549, 3.836044e-05 | 0.001557078, 0.0001537462 | -0.000271719, 3.582839e-05 | 0.001557078, 0.0001537462 |
| M5 | exact | -0.0007206549, 3.836044e-05 | 0.001557078, 0.0001537462 | -0.000271719, 3.582839e-05 | 0.001557078, 0.0001537462 |
| M5 | cnab2 | -0.0007355344, 3.817564e-05 | 0.001556925, 0.0001537274 | -0.0002716479, 3.582225e-05 | 0.001556925, 0.0001537274 |

## Stage 9 temporal-bias contribution

| model | exact physical bias (v,nu) | RK4 minus exact (v,nu) | |time shift|/|exact bias| per coordinate |
| --- | --- | --- | --- |
| M0 | 0.02186117, 0.0002012151 | 1.74305e-12, 1.199479e-13 | 7.97327e-11, 5.961178e-10 |
| M1 | 0.0009196993, 6.257532e-06 | 2.242428e-12, 8.379972e-14 | 2.438219e-09, 1.339182e-08 |
| M5 | -0.0002528619, -1.410738e-07 | 2.26974e-12, 8.289116e-14 | 8.976203e-09, 5.875731e-07 |

The RK4 temporal contribution is tiny relative to the clean spatial/model bias: the largest coordinate-wise ratio in this table is below 10⁻⁶. Stage 9 conclusions about spatial/model discrepancy therefore do not arise from its RK4 time error.

## Regression and preservation

Archived Stage 9/10 RK4 fits are checked with parameter absolute tolerance 2×10⁻⁸ and Stage 9 NMSE tolerance 10⁻¹². Reproduction differences do not overwrite historical values.

| stage | model | parameter difference | passed |
| --- | --- | --- | --- |
| 9 | M0 | 0, 0 | True |
| 9 | M1 | 0, 0 | True |
| 10 | M1 | -2.220446e-16, 1.851817e-16 | True |
| 9 | M5 | 0, 0 | True |
| 10 | M5 | -2.220446e-16, 2.042637e-16 | True |

| N | RK4 dtmax difference from Stage 9 | passed |
| --- | --- | --- |
| 16 | -1.421085e-13 | True |
| 24 | -8.526513e-13 | True |
| 32 | -3.410605e-13 | True |
| 48 | -3.126388e-13 | True |
| 64 | -1.705303e-13 | True |

Byte preservation verified: True (56 files).

## Runtime methodology

Every timing uses JIT and blocks all outputs until ready. First-call time includes compilation and first execution, not compilation alone. Ten subsequent calls report median/min/max and retain raw repetitions. Inputs are identical parameter vectors; the gradient task is scalar loss plus gradient in all four physical/discrepancy coordinates. Forward tasks are nine observations, final T=2, and full trajectory where practical. The full requested grid for exact and direct-power methods is a sampling choice, not an internal stepping requirement. Benchmarks include input dispatch overhead and are specific to this hardware/JAX version.

{'platform': 'macOS-14.7.4-arm64-arm-64bit', 'jax': '0.6.2', 'devices': ['TFRT_CPU_0']}

Actual iterative CN is timed; clean CN fits use its direct-power equivalent. CNAB2 always uses actual recurrence. The archived scan RK4 allocates its full trajectory before selecting outputs. Step counts are stability-safe estimates from the full-box stiff endpoint with half-limit safety and power-of-two rounding; CN/CNAB2 schedules are 512·N/32. Their accuracy is not presumed equal: consult the separate accuracy-at-cost table. Scan workloads above 16384 steps and full outputs above 600000 elements are omitted under predeclared caps.

## First call and warmed timings

| N | method | task | steps | first-call ms | warm median ms | min ms | max ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | scan_rk4 | observations | 1024 | 194.6638 | 4.779688 | 4.730792 | 4.917708 |
| 32 | scan_rk4 | final | 1024 | 329.7338 | 4.945 | 4.823125 | 5.356459 |
| 32 | scan_rk4 | full | 1024 | 175.042 | 4.808 | 4.759458 | 4.954125 |
| 32 | scan_rk4 | gradient | 1024 | 1486.208 | 55.8664 | 55.25654 | 56.364 |
| 32 | rk4 | observations | 1024 | 45.004 | 0.05160406 | 0.03966608 | 0.06925 |
| 32 | rk4 | final | 1024 | 38.818 | 0.03312447 | 0.01383305 | 0.05212508 |
| 32 | rk4 | full | 1024 | 43.34929 | 1.240979 | 1.165292 | 1.593583 |
| 32 | rk4 | gradient | 1024 | 109.103 | 0.07574999 | 0.07200008 | 0.1074171 |
| 32 | exact | observations | 0 | 35.96154 | 0.04431297 | 0.02312497 | 0.060083 |
| 32 | exact | final | 0 | 34.289 | 0.00999996 | 0.008999952 | 0.02337503 |
| 32 | exact | full | 0 | 38.49621 | 0.2708751 | 0.226333 | 0.410666 |
| 32 | exact | gradient | 0 | 66.87029 | 0.06441696 | 0.03733404 | 0.09562494 |
| 32 | cn | observations | 512 | 80.509 | 0.09610446 | 0.09420805 | 0.1179579 |
| 32 | cn | final | 512 | 53.42642 | 0.075271 | 0.07154199 | 0.09304099 |
| 32 | cn | full | 512 | 49.85596 | 0.1640835 | 0.1244589 | 0.919958 |
| 32 | cn | gradient | 512 | 139.6509 | 0.6037285 | 0.597959 | 0.662917 |
| 32 | cnab2 | observations | 512 | 107.463 | 0.2108545 | 0.208042 | 0.2247919 |
| 32 | cnab2 | final | 512 | 60.02821 | 0.182854 | 0.179667 | 0.199625 |
| 32 | cnab2 | full | 512 | 61.47875 | 0.260271 | 0.239584 | 0.456667 |
| 32 | cnab2 | gradient | 512 | 189.1056 | 1.168063 | 1.144375 | 1.384583 |
| 64 | scan_rk4 | observations | 16384 | 325.342 | 123.0548 | 122.2229 | 125.0704 |
| 64 | scan_rk4 | final | 16384 | 293.7583 | 123.845 | 122.6853 | 129.0481 |
| 64 | scan_rk4 | gradient | 16384 | 2652.991 | 1182.775 | 1175.744 | 1334.448 |
| 64 | rk4 | observations | 16384 | 44.6325 | 0.07447903 | 0.07020799 | 0.09183295 |
| 64 | rk4 | final | 16384 | 38.36808 | 0.03820856 | 0.01504098 | 0.05720905 |
| 64 | rk4 | gradient | 16384 | 119.6132 | 0.1203335 | 0.11725 | 0.147167 |
| 64 | exact | observations | 0 | 36.90121 | 0.05658303 | 0.04287495 | 0.08004205 |
| 64 | exact | final | 0 | 34.52983 | 0.04025002 | 0.02495805 | 0.05412498 |
| 64 | exact | full | 0 | 38.43592 | 0.7847705 | 0.6985 | 0.946583 |
| 64 | exact | gradient | 0 | 75.71654 | 0.0682915 | 0.05008408 | 0.09033293 |
| 64 | cn | observations | 1024 | 52.43012 | 0.334979 | 0.331916 | 0.353083 |
| 64 | cn | final | 1024 | 47.87975 | 0.1403955 | 0.137792 | 0.167708 |
| 64 | cn | full | 1024 | 77.62946 | 0.418375 | 0.295834 | 0.564417 |
| 64 | cn | gradient | 1024 | 174.0779 | 1.955313 | 1.879958 | 2.105208 |
| 64 | cnab2 | observations | 1024 | 63.78425 | 0.7211455 | 0.715083 | 0.7391659 |
| 64 | cnab2 | final | 1024 | 71.49342 | 0.5351035 | 0.530333 | 0.558333 |
| 64 | cnab2 | full | 1024 | 95.91546 | 0.875229 | 0.81375 | 1.043417 |
| 64 | cnab2 | gradient | 1024 | 221.5797 | 3.657 | 3.501834 | 4.009416 |
| 128 | rk4 | observations | 262144 | 45.84229 | 0.109917 | 0.106375 | 0.125375 |
| 128 | rk4 | final | 262144 | 38.20954 | 0.0464585 | 0.03624998 | 0.06854103 |
| 128 | rk4 | gradient | 262144 | 125.3385 | 0.1895205 | 0.1871251 | 0.222875 |
| 128 | exact | observations | 0 | 37.41617 | 0.0747295 | 0.05250005 | 0.09041699 |
| 128 | exact | final | 0 | 66.13525 | 0.03162504 | 0.01575006 | 0.04441699 |
| 128 | exact | full | 0 | 40.55658 | 2.519625 | 2.263542 | 3.069 |
| 128 | exact | gradient | 0 | 81.04092 | 0.1057085 | 0.07720804 | 0.1228331 |
| 128 | cn | observations | 2048 | 50.50279 | 1.208292 | 1.198209 | 1.226375 |
| 128 | cn | final | 2048 | 44.58104 | 0.395479 | 0.390375 | 0.411834 |
| 128 | cn | full | 2048 | 47.40508 | 0.990667 | 0.741542 | 1.227083 |
| 128 | cn | gradient | 2048 | 149.6713 | 6.507292 | 6.251083 | 6.777208 |
| 128 | cnab2 | observations | 2048 | 63.7775 | 2.731688 | 2.709792 | 2.772625 |
| 128 | cnab2 | final | 2048 | 58.67396 | 1.91725 | 1.894875 | 1.99825 |
| 128 | cnab2 | full | 2048 | 63.54221 | 2.342583 | 2.231084 | 2.675 |
| 128 | cnab2 | gradient | 2048 | 204.3944 | 12.1491 | 11.76971 | 12.47954 |
| 256 | rk4 | observations | 4194304 | 47.64754 | 0.180646 | 0.1768749 | 0.200042 |
| 256 | rk4 | final | 4194304 | 39.232 | 0.05106255 | 0.04266598 | 0.07120799 |
| 256 | rk4 | gradient | 4194304 | 122.3792 | 0.33025 | 0.327375 | 0.363625 |
| 256 | exact | observations | 0 | 36.83238 | 0.1221255 | 0.09608304 | 0.129791 |
| 256 | exact | final | 0 | 34.43887 | 0.04556245 | 0.03358303 | 0.06995793 |
| 256 | exact | gradient | 0 | 78.51363 | 0.121896 | 0.105708 | 0.1565419 |
| 256 | cn | observations | 4096 | 54.87833 | 4.268833 | 4.227834 | 4.377125 |
| 256 | cn | final | 4096 | 48.79117 | 1.242375 | 1.235792 | 1.3015 |
| 256 | cn | gradient | 4096 | 169.0145 | 25.01512 | 24.766 | 43.00692 |
| 256 | cnab2 | observations | 4096 | 73.11638 | 10.29529 | 10.21721 | 10.37792 |
| 256 | cnab2 | final | 4096 | 67.97929 | 7.253958 | 7.221208 | 7.322958 |
| 256 | cnab2 | gradient | 4096 | 291.234 | 50.23313 | 49.76221 | 59.93029 |
| 512 | exact | observations | 0 | 40.82679 | 0.1520205 | 0.10925 | 0.322041 |
| 512 | exact | final | 0 | 34.56275 | 0.05383347 | 0.04291604 | 0.07591711 |
| 512 | exact | gradient | 0 | 87.33838 | 0.180875 | 0.1505 | 0.390333 |
| 1024 | exact | observations | 0 | 41.11371 | 0.2286875 | 0.1977921 | 0.4235 |
| 1024 | exact | final | 0 | 39.80271 | 0.06287551 | 0.04541606 | 0.09791704 |
| 1024 | exact | gradient | 0 | 89.85829 | 0.311021 | 0.2885 | 0.529042 |

## Estimated workloads and resource omissions

| N | method | task | estimated steps | status | reason |
| --- | --- | --- | --- | --- | --- |
| 64 | scan_rk4 | full | 16384 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 64 | rk4 | full | 16384 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 128 | scan_rk4 | observations | 262144 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 128 | scan_rk4 | final | 262144 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 128 | scan_rk4 | full | 262144 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 128 | scan_rk4 | gradient | 262144 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 128 | rk4 | full | 262144 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | scan_rk4 | observations | 4194304 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | scan_rk4 | final | 4194304 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | scan_rk4 | full | 4194304 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | scan_rk4 | gradient | 4194304 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | rk4 | full | 4194304 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | exact | full | 0 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | cn | full | 4096 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 256 | cnab2 | full | 4096 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 512 | exact | full | 0 | omitted_resource_cap | Predeclared scan step or full-output element cap |
| 1024 | exact | full | 0 | omitted_resource_cap | Predeclared scan step or full-output element cap |

Observed resource failures: 0. Intentional workload omissions are not measured timings.

## Algorithmic memory/storage

| method | storage interpretation |
| --- | --- |
| scan_rk4 | One persistent real state plus RK4 stage temporaries; archived solver materializes (steps+1)*N output even for selected observations. Reverse-mode can retain step history. |
| rk4 | One initial spectrum plus requested_times*N complex amplification/output workspace; no internal step history. |
| exact | One initial spectrum plus requested_times*N complex amplification/output workspace; no internal step history. |
| cn | Iterative implementation carries two complex arrays (one redundant previous state), plus requested_times*N complex spectral output followed by real output; full output (steps+1)*N. Reverse scan can retain/reconstruct history. |
| cnab2 | Two persistent complex Fourier states, coefficient arrays and requested_times*N complex spectral output followed by real output; full output (steps+1)*N. Reverse scan can retain/reconstruct history. |
| peak_memory | Not measured; algorithmic storage only. No claim of precise peak device memory. |

Stored output counts vary for the full-trajectory task because each method uses its declared schedule; those timings are not equal-output workload comparisons. Final and nine-observation tasks have identical output shapes across methods. Full-trajectory runtime slopes are deliberately not fitted.

Output elements are N for final-only, 9N for observations, and (steps+1)N for full output. Selected scan RK4 still internally returns its full trajectory. FFT outputs/workspaces are complex before returning real fields. Reverse-mode storage can differ substantially from forward persistent state storage; no reliable peak device memory measurement was attempted.

## Accuracy at computational cost

Targets 10⁻²,10⁻⁴,10⁻⁶ were fixed before measurements. N=32, M5, initial A, T=2. Smallest tested power-of-two count satisfying both full-box sampled stability and target error is shown. Work counts, rather than incomparable runtime promises, are reported. RK4 direct powers do not actually perform the listed explicit updates; those counts define the numerical scheme. CN also has an equivalent direct-power shortcut. Stability is a distinct constraint from accuracy.

| method | target | minimum stable steps (sampled box) | selected tested steps | achieved relative L2 |
| --- | --- | --- | --- | --- |
| rk4 | 0.01 | 361 | 512 | 1.621006e-10 |
| rk4 | 0.0001 | 361 | 512 | 1.621006e-10 |
| rk4 | 1e-06 | 361 | 512 | 1.621006e-10 |
| cn | 0.01 | 1 | 32 | 0.003635168 |
| cn | 0.0001 | 1 | 256 | 5.69768e-05 |
| cn | 1e-06 | 1 | 2048 | 8.903058e-07 |
| cnab2 | 0.01 | 63 | 64 | 0.004503719 |
| cnab2 | 0.0001 | 63 | 512 | 7.102114e-05 |
| cnab2 | 1e-06 | 63 | 8192 | 2.779349e-07 |

## Large-grid exact scaling

| N | task | first-call ms | warm median ms |
| --- | --- | --- | --- |
| 128 | observations | 37.41617 | 0.0747295 |
| 128 | final | 66.13525 | 0.03162504 |
| 128 | gradient | 81.04092 | 0.1057085 |
| 256 | observations | 36.83238 | 0.1221255 |
| 256 | final | 34.43887 | 0.04556245 |
| 256 | gradient | 78.51363 | 0.121896 |
| 512 | observations | 40.82679 | 0.1520205 |
| 512 | final | 34.56275 | 0.05383347 |
| 512 | gradient | 87.33838 | 0.180875 |
| 1024 | observations | 41.11371 | 0.2286875 |
| 1024 | final | 39.80271 | 0.06287551 |
| 1024 | gradient | 89.85829 | 0.311021 |

| task | empirical time exponent, N128–1024 |
| --- | --- |
| observations | 0.5156785 |
| final | 0.3214951 |
| gradient | 0.5240105 |

The same output task is used within each slope. Small CPU arrays and dispatch overhead can dominate, so these empirical slopes do not establish asymptotic complexity. Fourier direct methods require FFT work plus requested-time spectral factors, approximately O(N log N + mN) for fixed m requested times. Actual stepping adds step-count work. Large-grid timing is computational scaling only, not a new spatial-accuracy study.

## Semi-discrete versus continuum truth

At N=32, true physical parameters (1,.002), initial A, T=2, exponential propagation has zero time-discretization error but retains these finite-grid spatial/model errors:

| model | max error vs continuum | relative L2 |
| --- | --- | --- |
| M0 | 0.1102571 | 0.07114654 |
| M1 | 0.005599193 | 0.003973284 |
| M5 | 0.002291258 | 0.001442731 |

## Recommendations by use case

For this linear, constant-coefficient periodic project, use exact semi-discrete exponential propagation as the time-exact reference and preferred new production propagator. Historical RK4 results remain archived. It removes timestep selection and temporal bias, but is not promised to be faster than the already efficient FFT RK4-power evaluator; the measured task-specific timing tables decide that comparison.

For future semilinear extensions, CNAB2 supplies a transferable differentiable stepping structure, subject to an explicit phase/advection stability check and an accuracy requirement. Its sampled stability advantage varies with resolution and parameter regime. Keep stencil/scan RK4 as an independent validation baseline. Full CN offers second-order A-stability, but poor very-stiff damping can make it unsuitable when high-frequency transients matter. Backward Euler is only a first-order L-stable diagnostic here.

## Caveats

Exact exponential and direct powers rely on linearity, constant coefficients and periodic diagonalizability; they are not drop-in nonlinear PDE solvers. CN is not L-stable; CNAB2 retains explicit phase restrictions. Dense box stability sampling is not an analytic enclosure. Timing is hardware/JAX dependent; first call and warmed execution are separated and synchronized. Reverse-mode memory is not inferred from forward state counts. Eliminating temporal error does not eliminate spatial discretization error, model discrepancy or identifiability limits. No universal speed winner or generic solution of PDE time integration is claimed.

## Tests and acceptance

124 passed in 35.67s

| check | status |
| --- | --- |
| all_tests_pass | PASS |
| symbol_equivalence | PASS |
| exact_gradients | PASS |
| rk4_equivalence | PASS |
| temporal_convergence_all_methods | PASS |
| box_stability_and_scaling | PASS |
| continuum_distinction | PASS |
| inverse_stationarity | PASS |
| paired_noise_complete | PASS |
| timing_synchronized_separated | PASS |
| large_grid_exact_complete | PASS |
| archived_regression | PASS |
| archives_preserved | PASS |
| use_case_recommendations_and_caveats | PASS |
| exact_zero_semigroup_imaginary | PASS |
| initial_state_derivatives | PASS |
| conservation_and_dissipation | PASS |

Overall: **PASS**.

Git status: fatal: not a git repository (or any of the parent directories): .git; no initialization, commit or push.

## Reproduction and recommended Stage 12

Run `.venv/bin/python -m pytest -q > docs/stage11_pytest.txt`, then `.venv/bin/python -m experiments.stage11_scalable_integration --phase PHASE` for symbols, temporal, stability, gradients, inverse, benchmark, accuracy, finalize in that order. Results checkpoint by inverse group and benchmark task, with resume validation of the protocol and archived hashes. The report is generated from the recorded results.

Stage 12 should be final synthesis/release only: final plots, consolidated technical report, concise GitHub README, architecture diagram, reproduction guide, test matrix, benchmark summary, limitations and interview explanation. Repository initialization, release tagging and versioning should occur only if requested. No new core physics is proposed or implemented here.
