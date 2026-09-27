# Stage 6: joint parameter inference and two-dimensional identifiability

## Objective and motivation

Infer both advection velocity v and positive hyperdiffusion ν4, and examine
the two-dimensional loss geometry rather than judging identifiability from
optimizer success alone. Stage 5 showed that sparse/noisy data can support
converged fits with large coefficient errors. Stage 6 asks how freeing velocity
changes parameter variability, local coupling, and continuum discretization bias.

Created `src/joint_inverse.py`, `experiments/stage6_joint_inference.py`,
`tests/test_joint_inverse.py`, `docs/stage6_results.json`,
`docs/stage6_optimizer_calibration.json`, and this report.
Updated `README.md`. All existing numerical modules, earlier tests, validation
reports, and result files are preserved. Joint prediction and loss wrap the
existing inverse functions, which call the same validated RK4/scan solver.
No second solver, scalar-pipeline rewrite, or new dependency was added.

## Setup, parameters, optimizer, and reproducibility

Retain [0,2π), N=32, T=2, truth (v,ν4)=(1,0.002), and

    ψ0(x)=sin(x)+0.5 sin(2x)+0.25 cos(3x).

Use 256 steps, dt=2/256. Initial state is exact. Full observations occur at
0,.25,...,2; sparse observations use 0,1,2. The objective is normalized MSE
over observed entries only, with denominator mean(data_observed²)+1e-12.

The optimization coordinates are theta=[v,q], q=log(ν4). Velocity is direct
and can be signed in the prediction API; this experiment searches v∈[0.5,1.5]
and ν4∈[1e-4,0.02]. Bounds are fixed and not tight around truth. Projected vector
Adam uses learning rate .05, beta1=.9, beta2=.999, epsilon=1e-8 and 2000 updates
for every final production fit. All six clean pilot starts converged at 800
updates, but severe seed 101 had not reached stationarity. At 1200 it was still
moving toward smaller diffusion; at 2000 it reached the declared lower bound
and satisfied the constrained-gradient condition. The update count was therefore
increased globally, with no learning-rate/bounds change or case-specific extension.
Calibration evidence is preserved in the JSON and a separate calibration file.

Updates are componentwise Adam with bias correction followed by projection in
(v,q). The optimizer receives no truth. Every fit saves 2001 states for v, q,
ν4, loss, gradient_v, and gradient_q, including the initial and final state.
Proposal boundary hits and actual bound contacts are recorded separately by
coordinate. All graph-defining quantities and data realizations stay fixed
within each fit.

Noise convention is unchanged: sigma_abs=sigma_rel*RMS(clean observed t>0
entries), with iid Gaussian noise only at positive times. Noise seeds are
100–104. Sparse layouts are generated without replacement using Stage 5 seed
200, then sorted. Five draws describe observed variability, not confidence
intervals. Matched clean data come from the solver; continuum truth is analytic.

Declared criteria: clean relative errors <1e-4 in both parameters; gradient
relative error <1e-5; final projected/KKT gradient norm <1e-7; finite histories
inside stable bounds; complete numerical loss grids and finite Hessians.
No accuracy or low-condition-number requirement is imposed on degraded cases.

Optimizer-count calibration for severe seed 101 (same initial pair and fixed hyperparameters):

| Updates | v | ν4 | Loss | Gradient v | Gradient q | Proposal hits (v,q) |
|---:|---:|---:|---:|---:|---:|---|
| 800 | 0.996019443168 | 0.000181891783186 | 2.246519329e-04 | 1.769214e-06 | 1.255866e-05 | [0, 0] |
| 1200 | 0.995811348147 | 0.000135498537429 | 2.220048090e-04 | 6.405069e-07 | 6.098852e-06 | [0, 0] |
| 2000 | 0.995652407329 | 0.0001 | 2.207365254e-04 | 2.245081e-10 | 2.641206e-06 | [0, 172] |

At the lower diffusion bound, a positive q-gradient points toward decreasing q outside the allowed box. It satisfies the one-sided optimality condition; only the remaining feasible velocity gradient contributes to the projected/KKT residual. The calibration is reported instead of silently continuing just this one production run.

Stability is checked over 21 velocities in [0.5,1.5], 101 coefficients across [1e-4,.02], and 8193 Fourier angles. Maximum sampled RK4 amplification is 1, including the constant mode. At the upper corner dt_max=0.0129372079026; dt/dt_max=0.603878368. This numerical box diagnostic covers the declared search range without adapting timesteps, but is not an analytical continuous-box CFL proof.

## Two-coordinate gradient verification

At (v,ν4)=(.9,.001), compare both components on clean full data and fixed masked/noisy data (3 times, 8 sensors, 2% noise, seed 100). Both perturbation scales are reported to check finite-difference stability.

| Data | Coordinate | JAX derivative | Epsilon | FD derivative | Absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|---:|
| clean | v | -4.838567378752e-01 | 1.0e-06 | -4.838567378793e-01 | 4.08351e-12 | 8.43950e-12 |
| clean | v | -4.838567378752e-01 | 5.0e-07 | -4.838567379192e-01 | 4.39822e-11 | 9.08991e-11 |
| clean | q | -1.433933853417e-03 | 1.0e-05 | -1.433933851692e-03 | 1.72497e-12 | 1.20296e-09 |
| clean | q | -1.433933853417e-03 | 5.0e-06 | -1.433933850131e-03 | 3.28622e-12 | 2.29175e-09 |
| masked_noisy | v | -6.826211678224e-01 | 1.0e-06 | -6.826211678482e-01 | 2.58056e-11 | 3.78037e-11 |
| masked_noisy | v | -6.826211678224e-01 | 5.0e-07 | -6.826211678204e-01 | 1.95000e-12 | 2.85663e-12 |
| masked_noisy | q | -9.333882258422e-04 | 1.0e-05 | -9.333882246970e-04 | 1.14517e-12 | 1.22690e-09 |
| masked_noisy | q | -9.333882258422e-04 | 5.0e-06 | -9.333882146356e-04 | 1.12066e-11 | 1.20063e-08 |

## Clean matched-model multi-start recovery

| Initial v | Initial ν4 | Fitted v | Fitted ν4 | Relative v error | Relative ν4 error | Initial loss | Final loss | Gradient norm | Updates | Proposal hits (v,q) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.7 | 0.0005 | 1 | 0.002 | 0.0000e+00 | 2.3852e-15 | 2.11132e-01 | 3.11884e-32 | 3.62008e-17 | 2000 | [0, 0] |
| 0.7 | 0.01 | 1 | 0.002 | 0.0000e+00 | 6.5052e-16 | 1.81570e-01 | 2.10024e-32 | 4.73938e-17 | 2000 | [0, 0] |
| 1.0 | 0.0005 | 0.99999999756687 | 0.0020000000133845 | 2.4331e-09 | 6.6922e-09 | 8.21305e-04 | 1.41762e-17 | 1.16066e-08 | 2000 | [0, 0] |
| 1.0 | 0.01 | 0.99999999999968 | 0.0020000000000004 | 3.2219e-13 | 1.7889e-13 | 1.02781e-02 | 2.47503e-25 | 1.53652e-12 | 2000 | [0, 0] |
| 1.3 | 0.0005 | 1 | 0.002 | 0.0000e+00 | 2.3852e-15 | 2.11132e-01 | 3.11884e-32 | 3.62008e-17 | 2000 | [0, 0] |
| 1.3 | 0.01 | 1 | 0.002 | 0.0000e+00 | 6.5052e-16 | 1.81570e-01 | 2.10024e-32 | 4.73938e-17 | 2000 | [0, 0] |

All six starts recover the same pair to high accuracy. Final residuals need not
be monotone with update count under fixed-rate Adam; the table reports the actual
last iterate, not a selected best iterate. This verifies
the matched-model pipeline and local consistency, not continuum physical accuracy.
The following surfaces are evaluated independently of optimization histories.

## Global and local two-dimensional landscapes

Each global grid contains 31 linear velocities from .7 to 1.3 and 31 logarithmic
coefficients from .0005 to .005 (961 pairs). This is a predeclared window inside
the broader optimizer bounds; statements about visible basins apply to that
sampled window, not a proof of global uniqueness over all possible parameters.
Each local 31×31 grid spans ±2% in velocity and ±.05 in log coefficient around
the corresponding fitted optimum. Loss-only evaluations use a JIT-vmapped row
of 31 points, reusing compilation across rows. No optimizer trajectory is used
to construct either grid.

All axes and every grid value are stored in the JSON with rows indexed by v
and columns by q. The coarse grid generally does not contain the exact optimum.

| Dataset/grid | Sampled minimum v | Sampled minimum ν4 | Minimum loss | Maximum loss | Interior sampled minima |
|---|---:|---:|---:|---:|---:|
| matched/global | 1 | 0.00199053585277 | 2.807924538e-08 | 2.111319073e-01 | 1 |
| matched/local | 1 | 0.002 | 2.634823457e-32 | 9.593594332e-04 | 1 |
| continuum/global | 1.02 | 0.00214933117354 | 7.759696275e-04 | 2.394517005e-01 | 1 |
| continuum/local | 1.02186117041 | 0.00220121509265 | 7.666544082e-04 | 1.765811413e-03 | 1 |

The numerical minima count compares each interior point with its eight neighbors; it is a finite-grid basin diagnostic, not a proof about unsampled minima. Loss dynamic ranges (log10(max/min)) are:
- matched/global: 6.876168 decades.
- matched/local: 28.561230 decades.
- continuum/global: 2.489373 decades.
- continuum/local: 0.362345 decades.

## Hessian, eigenvectors, and coordinate-dependent geometry

Compute H=jax.hessian(L), symmetrized as (H+Hᵀ)/2, in coordinates **(v,q=log ν4)**.
Eigenvectors are columns, ordered by increasing eigenvalue. Positive definiteness
requires eigenvalues above 100*machine_epsilon*max(1,max_abs_eigenvalue); otherwise
the condition number is null rather than forced positive. The normalized Hessian
coupling c=H_vq/sqrt(H_vv H_qq) is computed only for positive diagonal entries.
It is not a statistical parameter correlation.

Matched optimum:

```text
H = [[4.770220529564e+00, 3.399444891342e-17],
     [3.399444891342e-17, 2.505567271350e-03]]
eigenvalues = [0.0025055672713499362, 4.770220529564051]
eigenvectors (columns) = [[7.130134494674039e-18, -1.0], [-1.0, -7.130134494674039e-18]]
condition number = 1903.8485153080633
normalized Hessian coupling = 3.109465501809048e-16
positive-definite threshold = 1.059202e-13
```
The shallow eigenvector has (delta v,delta q)=(7.1301345e-18,-1). Eigenvector signs are arbitrary. Its relative components indicate whether the weak direction is predominantly log diffusion or a velocity/diffusion trade-off; interpretation depends on these coordinates.

Continuum optimum:

```text
H = [[4.725756225718e+00, 6.847794552709e-03],
     [6.847794552709e-03, 2.930307444119e-03]]
eigenvalues = [0.0029203786022634444, 4.725766154560055]
eigenvectors (columns) = [[0.0014499312650946804, -0.9999989488491108], [-0.9999989488491108, -0.0014499312650946804]]
condition number = 1618.203253132091
normalized Hessian coupling = 0.05819135506088154
positive-definite threshold = 1.049331e-13
```
The shallow eigenvector has (delta v,delta q)=(0.0014499313,-0.99999895). Eigenvector signs are arbitrary. Its relative components indicate whether the weak direction is predominantly log diffusion or a velocity/diffusion trade-off; interpretation depends on these coordinates.

Hessian conditioning changes if velocity is rescaled or ν4 replaces log ν4.
Neither an inverse Hessian nor a narrow-looking valley is a calibrated posterior
covariance or an uncertainty interval in this study.

Clean matched one-dimensional slices hold the other parameter fixed at its optimum:

| Relative v / log ν4 offset | Loss varying v only | Loss varying q only |
|---:|---:|---:|
| -0.050 | 5.952628044e-03 | 3.008763416e-06 |
| -0.020 | 9.537840487e-04 | 4.931354558e-07 |
| -0.010 | 2.384947704e-04 | 1.242771906e-07 |
| -0.005 | 5.962674058e-05 | 3.119419842e-08 |
| +0.000 | 2.634823457e-32 | 2.634823457e-32 |
| +0.005 | 5.962674057e-05 | 3.144547638e-08 |
| +0.010 | 2.384947704e-04 | 1.262874248e-07 |
| +0.020 | 9.537840484e-04 | 5.092176647e-07 |
| +0.050 | 5.952628040e-03 | 3.260084623e-06 |

These slices illustrate relative sharpness but do not replace the full 2D grids.
The local grid and Hessian assess the center more precisely than the global grid.

## Noise-only and sparse/noisy matched studies

Full-data noise studies use 1% and 5%. Moderate stress uses 2% noise, t=0,1,2,
and eight sensors; severe stress uses 5% noise, the same three times, and four
sensors. Masks are fixed at Stage 5 layout seed 200, not selected after results.
All four designs use seeds 100–104 and the same production optimizer settings.

| Design | Mean v | SD(v) | Mean abs. relative v error | Mean ν4 | SD(ν4) | Mean abs. relative ν4 error | Mean clean full-field L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| noise_0.01 | 0.9998079642 | 4.227317e-04 | 0.03120% | 0.002009658683 | 3.314265e-05 | 1.53410% | 0.13000% |
| noise_0.05 | 0.9990339986 | 2.111405e-03 | 0.15565% | 0.002048182979 | 1.669175e-04 | 7.72264% | 0.65040% |
| moderate | 1.001162368 | 1.692460e-03 | 0.16288% | 0.001812854472 | 2.119269e-04 | 12.08383% | 0.87183% |
| severe | 0.9994558038 | 7.684517e-03 | 0.56908% | 0.002178691063 | 1.256341e-03 | 46.93455% | 3.19898% |

Representative local geometry is recorded for every fit, not only selected favorable ones:

| Run | Small eigenvalue | Large eigenvalue | Condition number | Normalized Hessian coupling |
|---|---:|---:|---:|---:|
| noise_0.01_seed_100 | 2.5681873e-03 | 4.7643568e+00 | 1855.144 | -0.0006076164 |
| noise_0.05_seed_100 | 2.8192865e-03 | 4.7324910e+00 | 1678.613 | -0.002997948 |
| moderate_seed_100 | 2.9491827e-03 | 6.3603336e+00 | 2156.643 | -0.2695678 |
| moderate_seed_104 | 2.0522310e-03 | 6.4508717e+00 | 3143.346 | -0.2745915 |
| severe_seed_100 | 1.6951453e-03 | 6.5904934e+00 | 3887.863 | -0.4800693 |
| severe_seed_101 | 7.9067384e-06 | 8.2918110e+00 | 1048702 | -0.4163753 |
| severe_seed_104 | 3.7414050e-03 | 6.0761530e+00 | 1624.03 | -0.4392934 |

## Stage 5 fixed-v versus joint recovery

The exact same noise definitions, seeds, schedules, masks, and underlying truth
are reused for comparison. Stage 6 adds a degree of freedom, so equality of
coefficient estimates is not expected.

| Design/model | SD(ν4) | Mean abs. relative ν4 error | Mean clean final field L2 |
|---|---:|---:|---:|
| moderate/Stage 5 fixed v | 2.2996300e-04 | 13.79800% | 0.80045% |
| moderate/Stage 6 joint | 2.1192695e-04 | 12.08383% | 0.87183% |
| severe/Stage 5 fixed v | 1.2633929e-03 | 44.16161% | 2.50707% |
| severe/Stage 6 joint | 1.2563407e-03 | 46.93455% | 3.19898% |

In this five-seed sample, freeing v slightly reduces diffusion SD in both
designs. Moderate diffusion mean absolute relative error falls from 13.80% to
12.08%, but mean clean-field error rises from 0.800% to 0.872%. In severe
stress, diffusion error rises from 44.16% to 46.93%, and field error rises from
2.507% to 3.199%, despite slightly lower diffusion SD. Reduced spread alone is
therefore not evidence of more accurate identification. These are small-sample
comparisons, not universal effects of freeing velocity.

The comparison tests observed changes in coefficient spread and clean-field
prediction, not a general theorem that freeing velocity must help or hurt.
Matched-model noisy data have no discretization phase mismatch at truth;
velocity adjustments there reflect noise/sampling and parameter coupling.
The continuum study below separately addresses compensation for phase error.

## Multi-start and bound diagnostics

Moderate and severe seeds 100 and 104 use alternate (1.3,.01) starts versus
(.7,.0005). Both clean continuum fits also use these two starts.

| Case | Alternate − primary v | Alternate − primary ν4 | Alternate − primary loss |
|---|---:|---:|---:|
| moderate_seed_100 | +0.00000000e+00 | +1.73472348e-18 | +8.13151629e-19 |
| moderate_seed_104 | +0.00000000e+00 | +0.00000000e+00 | +0.00000000e+00 |
| severe_seed_100 | -1.11022302e-16 | -2.16840434e-18 | +9.75781955e-19 |
| severe_seed_104 | -1.11022302e-16 | -8.67361738e-18 | -1.95156391e-18 |
| continuum_clean | +0.00000000e+00 | +3.90312782e-18 | -9.75781955e-19 |

Across 37 fits, maximum final gradient norm is 2.641206e-06; maximum projected/KKT residual norm is 1.160657e-08. Total proposal boundary hits (v,q) are [0, 172]; total history bound contacts are [0, 172]. Individual counts and full histories are retained in JSON. Bound contact, if present, must not be interpreted as a physical optimum without further analysis.

## Clean continuum joint optimum and bias mechanism

Continuum observations evolve each Fourier mode analytically with phase rate
v_true*k and damping rate ν4_true*k⁴. They do not share the numerical solver's
modified wavenumbers. Fitting them produces an effective discrete-model pair.

- Fitted v: **1.02186117041**, physical error +2.186117041e-02 (+2.186117%).
- Fitted ν4: **0.00220121509265**, physical error +2.012150927e-04 (+10.060755%).
- Normalized MSE: 7.666544082e-04.
- Clean final full-field relative L2: 4.499283%.
- Stage 5 fixed-v continuum ν4: 0.00236430816381.

The global/local continuum landscapes and Hessian above quantify the displaced
basin and any change of tilt/conditioning. Single-mode rate-matching diagnostics
at N=32 are

    k1_discrete = sin(k dx)/dx
    k4_discrete = 16 sin⁴(k dx/2)/dx⁴
    v_effective = k/k1_discrete
    ν4_effective = .002 k⁴/k4_discrete.

| k | k1_discrete | k4_discrete | Effective v | Effective ν4 |
|---:|---:|---:|---:|---:|
| 1 | 0.993586851 | 0.993593023 | 1.0064545428 | 0.00201289658122 |
| 2 | 1.94899072 | 15.59349 | 1.02617215298 | 0.00205213843057 |
| 3 | 2.82949596 | 76.4358098 | 1.06025950913 | 0.00211942544324 |

No single pair matches all three modes simultaneously. The joint optimum is a
loss-weighted compromise, not any single-mode effective pair. Freeing velocity
can account for some phase mismatch rather than forcing a fixed-v diffusion
coefficient to absorb all of its effect.

## Degraded continuum cross-check

Use 2% noise, three times, eight sensors, seed-200 layout, and five noise seeds.
For each parameter distinguish physical error from shift relative to the NEW
clean joint discrete optimum, not the old scalar optimum. Shifts include the
change of observation design as well as noise.

| Seed | Fitted v | Fitted ν4 | Physical v error | Physical ν4 error | Shift in v | Shift in ν4 | Clean full-field L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.019541712 | 0.00183901961 | +1.9541712e-02 | -1.6098039e-04 | -2.3194582e-03 | -3.6219548e-04 | 4.65360% |
| 101 | 1.022741188 | 0.001526679787 | +2.2741188e-02 | -4.7332021e-04 | +8.8001711e-04 | -6.7453531e-04 | 4.94412% |
| 102 | 1.021495628 | 0.001246294439 | +2.1495628e-02 | -7.5370556e-04 | -3.6554230e-04 | -9.5492065e-04 | 5.35147% |
| 103 | 1.019614055 | 0.001411279173 | +1.9614055e-02 | -5.8872083e-04 | -2.2471155e-03 | -7.8993592e-04 | 5.12021% |
| 104 | 1.023008892 | 0.001387856288 | +2.3008892e-02 | -6.1214371e-04 | +1.1477219e-03 | -8.1335881e-04 | 5.13600% |

Across the five stressed continuum fits, mean v is 1.021280295 and mean ν4 is
0.001482225859. Mean signed physical errors are +0.021280295 and -0.000517774141;
mean shifts from the clean joint discrete optimum are -0.0005808754 and
-0.0007189892. Mean full-grid final relative L2 error is 5.0411%. The diffusion
shift from the clean joint optimum is larger in magnitude than its clean model
bias, so reporting only total physical error would obscure the mechanism.

These observation-induced shifts must not be confused with the clean numerical-
model bias. Clean-field quality uses all 32 final spatial points for every run,
including sparse/noisy designs. Fitting measured entries well need not identify
either physical coefficient accurately.

## Individual fit diagnostics

Full histories and Hessians for all fits are saved in JSON; the table reports
parameters, errors, loss, and clean-field prediction quality for inspection.

| Fit | v | ν4 | Relative v error | Relative ν4 error | Final loss | Gradient norm | Clean field L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_start_0 | 1 | 0.002 | 0.00000% | 0.00000% | 3.118839e-32 | 3.62008e-17 | 0.00000% |
| clean_start_1 | 1 | 0.002 | 0.00000% | 0.00000% | 2.100243e-32 | 4.73938e-17 | 0.00000% |
| clean_start_2 | 0.9999999976 | 0.002000000013 | 0.00000% | 0.00000% | 1.417622e-17 | 1.16066e-08 | 0.00000% |
| clean_start_3 | 1 | 0.002 | 0.00000% | 0.00000% | 2.475027e-25 | 1.53652e-12 | 0.00000% |
| clean_start_4 | 1 | 0.002 | 0.00000% | 0.00000% | 3.118839e-32 | 3.62008e-17 | 0.00000% |
| clean_start_5 | 1 | 0.002 | 0.00000% | 0.00000% | 2.100243e-32 | 4.73938e-17 | 0.00000% |
| noise_0.01_seed_100 | 1.000163392 | 0.002031592603 | 0.01634% | 1.57963% | 8.970475e-05 | 6.55653e-16 | 0.09924% |
| noise_0.01_seed_101 | 1.000008324 | 0.002028245016 | 0.00083% | 1.41225% | 9.140047e-05 | 2.02915e-16 | 0.08032% |
| noise_0.01_seed_102 | 1.000128291 | 0.001973023201 | 0.01283% | 1.34884% | 7.883588e-05 | 7.47694e-16 | 0.08382% |
| noise_0.01_seed_103 | 0.9991939275 | 0.001974418491 | 0.08061% | 1.27908% | 8.017923e-05 | 4.56290e-16 | 0.22117% |
| noise_0.01_seed_104 | 0.9995458861 | 0.002041014105 | 0.04541% | 2.05071% | 8.555339e-05 | 1.35726e-16 | 0.16546% |
| noise_0.05_seed_100 | 1.000821945 | 0.002159229205 | 0.08219% | 7.96146% | 2.237008e-03 | 2.33468e-16 | 0.49656% |
| noise_0.05_seed_101 | 1.000022335 | 0.002141495428 | 0.00223% | 7.07477% | 2.273674e-03 | 9.24065e-17 | 0.39941% |
| noise_0.05_seed_102 | 1.000632034 | 0.00186528138 | 0.06320% | 6.73593% | 1.962792e-03 | 6.38491e-16 | 0.42026% |
| noise_0.05_seed_103 | 0.9959700515 | 0.001869044117 | 0.40299% | 6.54779% | 2.002581e-03 | 5.58226e-16 | 1.11121% |
| noise_0.05_seed_104 | 0.9977236266 | 0.002205864766 | 0.22764% | 10.29324% | 2.127932e-03 | 4.80787e-16 | 0.82455% |
| moderate_seed_100 | 0.9993824953 | 0.002136327568 | 0.06175% | 6.81638% | 1.632297e-04 | 7.91955e-16 | 0.41665% |
| moderate_seed_101 | 1.002732633 | 0.001894354254 | 0.27326% | 5.28229% | 1.699016e-04 | 5.97868e-16 | 0.77074% |
| moderate_seed_102 | 1.001397125 | 0.001582353914 | 0.13971% | 20.88230% | 2.220009e-04 | 1.14366e-15 | 1.27463% |
| moderate_seed_103 | 0.9994514752 | 0.001722135035 | 0.05485% | 13.89325% | 1.779093e-04 | 1.06230e-15 | 0.81794% |
| moderate_seed_104 | 1.002848114 | 0.001729101591 | 0.28481% | 13.54492% | 2.211937e-04 | 1.25134e-16 | 1.07922% |
| severe_seed_100 | 0.9984866921 | 0.002102609053 | 0.15133% | 5.13045% | 4.786375e-04 | 6.65334e-16 | 0.48710% |
| severe_seed_101 | 0.9956524073 | 0.0001 | 0.43476% | 95.00000% | 2.207365e-04 | 2.64121e-06 | 6.24207% |
| severe_seed_102 | 1.010855728 | 0.002996373299 | 1.08557% | 49.81866% | 4.759026e-04 | 9.60782e-17 | 3.84275% |
| severe_seed_103 | 1.002010768 | 0.002386632308 | 0.20108% | 19.33162% | 6.964103e-05 | 2.02799e-15 | 1.19268% |
| severe_seed_104 | 0.9902734241 | 0.003307840655 | 0.97266% | 65.39203% | 7.002371e-04 | 3.95873e-16 | 4.23029% |
| moderate_seed_100_alternate | 0.9993824953 | 0.002136327568 | 0.06175% | 6.81638% | 1.632297e-04 | 6.41071e-16 | 0.41665% |
| moderate_seed_104_alternate | 1.002848114 | 0.001729101591 | 0.28481% | 13.54492% | 2.211937e-04 | 1.25134e-16 | 1.07922% |
| severe_seed_100_alternate | 0.9984866921 | 0.002102609053 | 0.15133% | 5.13045% | 4.786375e-04 | 1.51513e-16 | 0.48710% |
| severe_seed_104_alternate | 0.9902734241 | 0.003307840655 | 0.97266% | 65.39203% | 7.002371e-04 | 9.98372e-16 | 4.23029% |
| continuum_clean | 1.02186117 | 0.002201215093 | 2.18612% | 10.06075% | 7.666544e-04 | 5.75634e-16 | 4.49928% |
| continuum_clean_alternate | 1.02186117 | 0.002201215093 | 2.18612% | 10.06075% | 7.666544e-04 | 4.81577e-16 | 4.49928% |
| continuum_stressed_seed_100 | 1.019541712 | 0.00183901961 | 1.95417% | 8.04902% | 6.524237e-04 | 7.65376e-16 | 4.65360% |
| continuum_stressed_seed_101 | 1.022741188 | 0.001526679787 | 2.27412% | 23.66601% | 1.229597e-03 | 3.77076e-16 | 4.94412% |
| continuum_stressed_seed_102 | 1.021495628 | 0.001246294439 | 2.14956% | 37.68528% | 1.535000e-03 | 2.88663e-16 | 5.35147% |
| continuum_stressed_seed_103 | 1.019614055 | 0.001411279173 | 1.96141% | 29.43604% | 1.436241e-03 | 1.91663e-15 | 5.12021% |
| continuum_stressed_seed_104 | 1.023008892 | 0.001387856288 | 2.30089% | 30.60719% | 1.052928e-03 | 6.05180e-17 | 5.13600% |

## Execution and caveats

Executed from the project root:

    .venv/bin/python -m pytest -q
    .venv/bin/python -m experiments.stage6_joint_inference

Full pytest summary:

    53 passed in 20.51s

All 45 previous cases pass unchanged. Eight joint tests cover parameter mapping,
prediction shape and loss, both gradient components against finite differences,
bounds and gradient history, a short loss-decreasing fit moving both parameters
toward truth, finite Hessian shape, and correct handling of a nonpositive Hessian.
Full landscapes and stochastic production fits are not executed inside pytest.
Environment: Python 3.10, JAX/jaxlib 0.6.2, pytest 9.1.1, float64.

The results JSON preserves all grid values, axes, masks, times, seeds, noise RMS,
configurations, optimizer histories, bounds, Hessian matrices/eigensystems,
comparisons, and checks. JSON forbids NaN/Infinity; nonfinite numerical output
raises an error. A rerun overwrites only the Stage 6 results file and records
IN PROGRESS until completion.

Scientific caveats: conditioning and eigenvectors depend on coordinates/scaling;
normalized Hessian coupling is not statistical correlation; inverse Hessian is
not automatically covariance; visible valleys do not supply practical uncertainty
without a statistical model. Matched-model recovery is artificially clean;
continuum observations expose discretization mismatch. Sparse/noisy data can
yield modest field errors and poor coefficient estimates. Two scalar parameters
remain much simpler than a coefficient field or neural closure. Bounds can affect
weakly identified fits, and all contacts are reported. Initial state remains
exact, noise iid Gaussian, and five seeds are not a confidence interval.

**PASS** for numerical soundness of the joint-identifiability study. All tests and declared checks pass; poor identifiability in degraded cases is a scientific finding rather than a requirement to recover truth accurately.

Git status:

    fatal: not a git repository (or any of the parent directories): .git

No repository was initialized, no commit or push was made, and Stage 7 is not implemented.

## Interpretation and recommended Stage 7

The clean matched optimum is an isolated positive-curvature minimum with an
approximately axis-aligned valley. Its shallow direction is almost entirely
log diffusion: the normalized mixed curvature is 3.109e-16,
while the eigenvalue ratio is 1903.85. Large coordinate-dependent
conditioning therefore does not imply strong parameter coupling in the clean
full-field problem. The diffusion eigenvalue is well above the declared
roundoff threshold; it is not numerically zero.

Continuum mismatch shifts the fitted velocity upward by
2.1861% and diffusion by
10.0608%. The latter is smaller than the fixed-v
Stage 5 bias of approximately 18.22%; clean final field error falls from about
7.01% to 4.4993%. Allowing velocity to
compensate for modified phase speed improves the fit but does not remove the
multimode model mismatch. Normalized Hessian coupling becomes
0.05819, a small nonzero tilt; conditioning is
1618.20 in the same coordinates. Neither result is a
statistical correlation or uncertainty statement.

Under severe matched stress, mean absolute relative velocity error is
0.5691%, versus
46.9346% for diffusion. Mean full-field error
is 3.1990%. Thus modest prediction
error again coexists with poor coefficient identification. The seed-101
solution reaches the diffusion lower bound; its nonzero outward gradient is
compatible with a constrained minimum, not an unconstrained physical optimum.
The result is sensitive to the declared search interval and must not be
interpreted as establishing a physical lower limit.

The chosen moderate/severe alternate starts agree on each dataset, so the
observed pair variability is principally across noise realizations, rather
than demonstrated multiple local minima. The local Hessians show weaker and
more coupled directions for sparse data. Global uniqueness in all degraded
designs has not been proved. Stage 5/6 tables show that freeing velocity does
not uniformly increase diffusion variance or improve clean-field prediction;
those conclusions must be assessed separately from the continuum phase-bias
mechanism.

Recommend **Stage 7: observation-design and sensor-placement studies**. The
observed bottleneck is weak diffusion sensitivity in sparse designs despite
well-estimated velocity, not failure of clean optimization. Compare predeclared
layouts/times using suitably scaled sensitivity directions, then evaluate
recovery on held-out noise seeds. Avoid optimizing only the normalized
curvature score or selecting layouts by closeness to known truth. No Stage 7
implementation is included.
