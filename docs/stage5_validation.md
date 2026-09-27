# Stage 5 observation robustness validation

## Objective and scope

Quantify recovery of ν4 under additive measurement noise, fewer observation
times, partial spatial sensors, and a small declared set of combined stresses.
Velocity remains known. Primary data are matched-model synthetic observations,
so observation degradation is isolated from numerical-model mismatch.

Stage 4 established that N=32 continuum data fitted with this model yield
ν4≈0.0023643081 rather than 0.002, predominantly from spatial discretization.
A separate continuum cross-check reports both physical error and shift from
the clean discrete optimum. The latter can include mask/time-selection effects
as well as noise; it is not automatically a noise-only effect.

## Files and preservation

Created `src/observations.py`, `experiments/stage5_observation_robustness.py`,
`tests/test_observations.py`, `docs/stage5_results.json`, and this report.
Updated `README.md` and minimally extended `src/inverse.py` with a trailing
optional `sensor_indices=None` argument on prediction and loss functions.
The default still returns and compares full spatial fields. The projected Adam
implementation is byte-for-byte unchanged. No solver or inverse optimizer was
duplicated, no dependency added, and all previous tests remain unchanged.
Prior operators, solver, validation reports, and result files were preserved.

## Clean setup and observation convention

Use N=32 on [0,2π), v=1, ν4_true=0.002, T=2, and

    ψ0 = sin(x) + 0.5 sin(2x) + 0.25 cos(3x).

The initial condition is known exactly. Use 256 RK4 steps with fixed dt=2/256,
with the same validated solver and full trajectory internally. Candidate
coefficients cannot change dt, num_steps, masks, or data within a fit.

Nine candidate observations are at t=0,0.25,...,2. The t=0 observation stays
exact under every degradation; it contains no parameter information. Only
selected time-space entries are passed to the loss.

Noise is iid Gaussian with ensemble mean zero; a finite realization need not
have exactly zero empirical mean. For each selected dataset:

    sigma_abs = sigma_rel * RMS(clean observed entries with t>0)
    data = clean + sigma_abs * Normal(0,1), for t>0 only.

No noise is applied to ψ0 or t=0 measurements. Noise levels are 0%, 0.5%, 1%,
2%, and 5%. Nonzero levels use seeds 100,101,102,103,104. Reusing these seeds
across levels pairs the same standardized draws at fixed shape; the five seeds
within each level are separate realizations. Five draws are a variability study,
not a confidence interval or a calibrated uncertainty distribution.

Temporal schedules are nested:

| Total snapshots | Informative t>0 snapshots | Times |
|---:|---:|---|
| 9 | 8 | 0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2 |
| 5 | 4 | 0, .5, 1, 1.5, 2 |
| 3 | 2 | 0, 1, 2 |
| 2 | 1 | 0, 2 |

Spatial counts are 32,16,8,4. Partial masks use JAX random sampling without
replacement, sorted afterward, with seeds 200,201,202,203,204. This avoids
basing a conclusion on one highly symmetric four-sensor mask. Exact indices
are saved in the JSON and listed below. No layout was selected after viewing
recovery results, and no optimal-placement claim is made.

## Loss, optimizer, and stability

The loss is exactly

    mean((prediction_observed-data_observed)^2)
    / (mean(data_observed^2)+1e-12),

where both means include only observed time-space entries. No unobserved entry
contributes to the denominator. Noise is fixed before constructing the loss;
finite differences and autodiff evaluate the same data realization. Prediction
quality uses the underlying CLEAN truth, including a full-grid final-time L2
comparison even for sparse sensors.

Optimize q=log(ν4) with ν4=exp(q), using unchanged Stage 3 projected Adam:
800 updates, learning rate .05, beta1=.9, beta2=.999, epsilon=1e-8, and coefficient
bounds [1e-4,.02]. The initial coefficient is .0005 except declared alternate
starts at .01. No case-specific optimizer tuning or shortened production runs.

The predeclared numerical criteria are: matched baseline relative error <1e-4,
continuum baseline agreement <1e-10 absolute, masked gradient relative agreement
<1e-5, and final |dL/dq|<1e-8 for interior fits with local gradients bracketing
a minimum. Bound solutions instead use the appropriate one-sided gradient
condition. Stability and finite histories are required. Poor-data parameter
accuracy is deliberately not a PASS criterion.

At the search upper bound, dt_max=0.0129372079026; dt/dt_max=0.603878368. The reused 101-coefficient/8193-angle range diagnostic has maximum amplification 1, including the neutral constant mode. Bounds keep optimization within the tested interval. This is numerical stability evidence, not an analytical continuous-interval CFL proof.

## Clean matched baseline

Recovered ν4=0.001999999999999995, absolute relative parameter error=2.385245e-15, final normalized loss=2.634823e-32, and clean final-field relative L2 error=1.392342e-16. Machine-level accuracy verifies the pipeline, not real-physics accuracy.

## Study A: noise only

All nine times and all 32 spatial points are observed. Sample standard deviations use n−1; a singleton has no sample SD.

| Design | Runs | Mean ν4 | Median ν4 | Sample SD | Mean abs. relative error | Median abs. relative error | Min ν4 | Max ν4 | Mean clean final L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 1 | 0.002 | 0.002 | — | 0.000000% | 0.000000% | 0.002 | 0.002 | 0.000000% |
| 0.5% | 5 | 0.00200483982 | 0.002014119936 | 1.654870e-05 | 0.766118% | 0.705997% | 0.001986506009 | 0.002020506402 | 0.043615% |
| 1% | 5 | 0.002009694608 | 0.002028247205 | 3.311260e-05 | 1.532992% | 1.412360% | 0.001973008999 | 0.002041052192 | 0.087236% |
| 2% | 5 | 0.002019449246 | 0.002056523598 | 6.628594e-05 | 3.069006% | 2.826180% | 0.001946006188 | 0.00208226207 | 0.174492% |
| 5% | 5 | 0.002049076425 | 0.002141524954 | 1.661693e-04 | 7.695138% | 7.076248% | 0.001864932144 | 0.002206840404 | 0.436378% |

Realized noise RMS is measured only on t>0 observed entries, and is recorded for every run.

| Noise level | Mean realized RMS |
|---:|---:|
| 0% | 0.00000000e+00 |
| 0.5% | 3.92780308e-03 |
| 1% | 7.85560616e-03 |
| 2% | 1.57112123e-02 |
| 5% | 3.92780308e-02 |

In this ensemble, mean absolute relative parameter error rises from 0.766% at
0.5% noise to 1.533%, 3.069%, and 7.695% at 1%, 2%, and 5% noise. The sample
standard deviation likewise grows from 1.65e-5 to 1.66e-4. These are observed
degradation trends for this design, not universal noise-to-error conversion rates.

The seed summaries quantify error and spread rather than relying on one lucky
realization. Adjacent levels are not required to degrade monotonically: five
realizations are too few for precise distributional claims. Fixed seeds across
levels also mean those level comparisons are paired, not independent ensembles.

## Study B: noiseless temporal sparsity

| Snapshots | Informative snapshots | Recovered ν4 | Relative parameter error | Final loss | Curvature in q |
|---:|---:|---:|---:|---:|---:|
| 9 | 8 | 0.002 | 2.385245e-15 | 2.634823e-32 | 2.505568849e-03 |
| 5 | 4 | 0.002 | 2.385245e-15 | 2.602707e-32 | 2.600430296e-03 |
| 3 | 2 | 0.002 | 3.252607e-15 | 3.056432e-32 | 2.790442697e-03 |
| 2 | 1 | 0.002 | 3.252607e-15 | 2.805079e-32 | 3.160402080e-03 |

All four noiseless schedules recover truth to floating-point precision. Their
normalized curvature actually increases from 0.002506 to 0.003160 as snapshots
are reduced from nine to two; flattening is not observed in this specific metric.

Curvature is measured for a normalized average loss, not a summed likelihood.
Removing observations therefore need not reduce this curvature: removing earlier,
less-sensitive snapshots can increase average sensitivity, and each schedule
changes the normalization and fraction of t=0 entries. We report the measured
values instead of asserting that fewer times must flatten this particular loss.
Noiseless scalar recovery with one informative full-field snapshot does not
establish equivalent noise robustness or total information across schedules.

## Study C: noiseless sensor sparsity and layout variation

| Design | Runs | Mean ν4 | Median ν4 | Sample SD | Mean abs. relative error | Median abs. relative error | Min ν4 | Max ν4 | Mean clean final L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 sensors | 1 | 0.002 | 0.002 | — | 0.000000% | 0.000000% | 0.002 | 0.002 | 0.000000% |
| 16 sensors | 5 | 0.002 | 0.002 | 1.226635e-18 | 0.000000% | 0.000000% | 0.002 | 0.002 | 0.000000% |
| 8 sensors | 5 | 0.002 | 0.002 | 2.261803e-18 | 0.000000% | 0.000000% | 0.002 | 0.002 | 0.000000% |
| 4 sensors | 5 | 0.002 | 0.002 | 3.469447e-18 | 0.000000% | 0.000000% | 0.002 | 0.002 | 0.000000% |

| Sensors | Mean curvature | Minimum curvature | Maximum curvature |
|---:|---:|---:|---:|
| 32 | 2.50556885e-03 | 2.50556885e-03 | 2.50556885e-03 |
| 16 | 2.63073151e-03 | 2.24158421e-03 | 2.82113202e-03 |
| 8 | 2.88011014e-03 | 2.75528126e-03 | 3.04123300e-03 |
| 4 | 3.16512255e-03 | 2.45026603e-03 | 4.28262378e-03 |

All tested noiseless layouts recover truth to floating-point precision. Four-
sensor curvature ranges from 0.002450 to 0.004283 across layouts, approximately
a 1.75-fold spread, despite almost indistinguishable recovered coefficients.

Exact noiseless recovery can coexist with substantial layout-dependent curvature.
These quantities distinguish optimizer success from resilience to perturbation.
Because the loss normalizes by each layout's field energy, curvature comparisons
also depend on normalization; they are not sensor-placement scores or posterior
uncertainty. Layout variation is separate from noise-seed variation.

Declared layouts:

| Sensor count | Seed | Sorted indices |
|---:|---:|---|
| 16 | 200 | [0, 2, 3, 5, 7, 11, 12, 13, 20, 22, 23, 24, 25, 26, 28, 30] |
| 16 | 201 | [0, 1, 3, 4, 5, 6, 10, 12, 13, 14, 19, 23, 24, 25, 27, 30] |
| 16 | 202 | [0, 2, 3, 5, 6, 9, 13, 16, 21, 22, 23, 27, 28, 29, 30, 31] |
| 16 | 203 | [2, 8, 10, 11, 12, 13, 16, 17, 18, 19, 21, 24, 26, 27, 29, 31] |
| 16 | 204 | [1, 5, 6, 8, 9, 12, 15, 19, 20, 21, 22, 24, 26, 28, 29, 30] |
| 8 | 200 | [0, 7, 12, 20, 22, 23, 25, 28] |
| 8 | 201 | [0, 4, 6, 14, 19, 24, 27, 30] |
| 8 | 202 | [0, 5, 9, 16, 23, 27, 30, 31] |
| 8 | 203 | [2, 11, 16, 19, 21, 24, 29, 31] |
| 8 | 204 | [9, 12, 15, 19, 20, 21, 24, 26] |
| 4 | 200 | [0, 7, 23, 28] |
| 4 | 201 | [0, 6, 19, 27] |
| 4 | 202 | [0, 5, 27, 31] |
| 4 | 203 | [16, 21, 24, 31] |
| 4 | 204 | [9, 21, 24, 26] |

## Study D: predeclared combined stresses

All use layout seed 200 from Study C, without searching for favorable masks:
case 1 = 1% noise, 5 times, 16 sensors;
case 2 = 2% noise, 3 times, 8 sensors;
case 3 = 5% noise, 3 times, 4 sensors.
Each uses all five noise seeds.

| Design | Runs | Mean ν4 | Median ν4 | Sample SD | Mean abs. relative error | Median abs. relative error | Min ν4 | Max ν4 | Mean clean final L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Case 1 | 5 | 0.001992785036 | 0.002003397802 | 2.738729e-05 | 1.087260% | 1.086560% | 0.001959896536 | 0.002021731207 | 0.061998% |
| Case 2 | 5 | 0.001784952737 | 0.001735435426 | 2.299630e-04 | 13.798003% | 13.228229% | 0.001548823885 | 0.002152282003 | 0.800448% |
| Case 3 | 5 | 0.002204355487 | 0.002271673996 | 1.263393e-03 | 44.161615% | 20.279889% | 0.0003028079683 | 0.00385463865 | 2.507067% |

Mean absolute relative error is 1.087%, 13.798%, and 44.162% for cases 1–3.
Case 3 spans fitted coefficients 0.000302808 to 0.003854639, including errors
of about -84.86% and +92.73%. Its weakest fitted curvature is 6.31e-5, much
smaller than the full-data clean curvature of 0.002506. These are converged
fits, so weak observations can produce unreliable parameters even when the
optimizer works. The mean clean final-field L2 error in case 3 is only 2.507%:
a comparatively small field error does not ensure accurate parameter recovery.

These few cases are illustrative combined stresses, not a full factorial study;
they cannot individually attribute deterioration to noise, times, or sensors.
The declared layout is one placement per stress case. It was not chosen based
on outcomes. Clean-field errors above compare the full final field to truth,
not just the noisy measured entries.

## Masked/noisy gradient and optimizer diagnostics

At q=-6.90775527898 (ν4=.001), use three times [0, 128, 256], sensors [0, 7, 12, 20, 22, 23, 25, 28], 2% noise, and seed 100. With epsilon=1.0e-05 in q:

| Quantity | Value |
|---|---:|
| JAX dL/dq | -9.567664621360e-04 |
| Centered FD | -9.567664616030e-04 |
| Absolute difference | 5.329948e-13 |
| Relative difference | 5.570793e-10 |

Every fit records its final gradient and loss/gradient probes at q±0.001 and q.
Curvature is [g(q+h)−g(q−h)]/(2h). This is local numerical sensitivity of the
chosen objective, not a formal confidence interval, posterior, or global
identifiability proof. The JSON includes all probes and 801-entry coefficient
and loss histories for every 800-update fit.

Three difficult cases were preselected for alternate initialization (.01 versus
.0005): full-field 5% noise seed 100, and combined case 3 seeds 100 and 104.

| Case | Main ν4 | Alternate ν4 | Absolute difference |
|---|---:|---:|---:|
| noise_0.05_seed_100 | 0.00215908261488 | 0.00215908261488 | 2.168404e-18 |
| combined_3_seed_100 | 0.00218705902943 | 0.00218705902943 | 1.344411e-17 |
| combined_3_seed_104 | 0.00385463865002 | 0.00385463865002 | 3.469447e-18 |

Across 69 production fits, maximum absolute final gradient is 8.286324e-18; 0 fits end at a parameter bound, and total proposal boundary hits are 0. Stationarity checks passed: True. Multi-start agreement is evidence for these chosen cases, not a guarantee of global identifiability for all degraded data.

## Continuum cross-check: separate model bias and observation shifts

Clean N=32 full-field continuum recovery gives 0.00236430816380557, versus the stored Stage 4 same-schedule value 0.00236430816380558. Absolute reproduction difference is 2.168404e-18, below 1e-10. Its model bias relative to physical truth is +3.643081638e-04 (+18.215408%).

For each continuum run we distinguish:

    physical error = fitted ν4 − 0.002
    observation-induced shift = fitted ν4 − clean full-data discrete optimum.

A clean sparse control is also included, making deterministic selection effects
visible before adding noise. Both truth and clean-field references here are
analytical continuum fields.

| Continuum design | Runs | Mean fitted ν4 | Sample SD | Mean physical error | Mean observation shift | Mean clean final L2 |
|---|---:|---:|---:|---:|---:|---:|
| Clean/full | 1 | 0.00236430816381 | — | +3.64308164e-04 | +0.00000000e+00 | 7.012215% |
| 1% noise/full | 5 | 0.00237443150831 | 3.378615e-05 | +3.74431508e-04 | +1.01233445e-05 | 7.011963% |
| Clean/3 times/8 sensors | 1 | 0.00135666244532 | — | -6.43337555e-04 | -1.00764572e-03 | 7.678783% |
| 2% noise/3 times/8 sensors | 5 | 0.00113981019307 | 2.302003e-04 | -8.60189807e-04 | -1.22449797e-03 | 8.006869% |

The 1% full-data noise ensemble moves the coefficient mean by only +1.01e-5
from the clean numerical optimum, while the mean physical error remains about
+3.74e-4. By contrast, the declared noiseless sparse design alone shifts the
optimum to 0.001356662445, below physical truth. Adding 2% noise yields a mean
of 0.001139810193. The large sparse-data shift is already present without noise,
showing how observation selection interacts with discretization mismatch.

The full physical error cannot be attributed to measurement noise. Even noiseless
sparse selection can move the optimum of a misspecified numerical model because
it changes which residuals are weighted. Observation-induced shifts relative to
the full clean optimum therefore include this selection effect. The unchanged
Stage 4 numerical-model bias remains present in continuum experiments.

## Individual-run measurements

All individual seed/layout results below include signed physical error,
absolute relative parameter error, fit loss, final gradient, and full clean
final-field relative L2. JSON additionally stores clean observation-space NMSE,
realized noise RMS, sigma_abs, complete masks/seeds, and histories.
Run identifiers encode study, noise level/seed or layout; continuum observation
shifts are given in a separate table after this one.

| Run | Fitted ν4 | Signed physical error | Absolute relative error | Final loss | Final gradient | Clean final relative L2 |
|---|---:|---:|---:|---:|---:|---:|
| matched_clean_full | 0.002 | -4.7704896e-18 | 0.00000% | 2.6348235e-32 | 7.2970e-19 | 0.00000% |
| noise_0.005_seed_100 | 0.002015779101 | +1.5779101e-05 | 0.78896% | 2.2444559e-05 | 8.8076e-19 | 0.04489% |
| noise_0.005_seed_101 | 0.0020141199356 | +1.4119936e-05 | 0.70600% | 2.2859697e-05 | -5.5160e-19 | 0.04017% |
| noise_0.005_seed_102 | 0.0019865060087 | -1.3493991e-05 | 0.67470% | 1.9725489e-05 | -1.1247e-18 | 0.03846% |
| noise_0.005_seed_103 | 0.0019872876503 | -1.2712350e-05 | 0.63562% | 2.0431205e-05 | -2.0519e-18 | 0.03623% |
| noise_0.005_seed_104 | 0.0020205064023 | +2.0506402e-05 | 1.02532% | 2.1520584e-05 | -2.1328e-18 | 0.05832% |
| noise_0.01_seed_100 | 0.0020315867143 | +3.1586714e-05 | 1.57934% | 8.9768348e-05 | -3.3527e-18 | 0.08977% |
| noise_0.01_seed_101 | 0.0020282472049 | +2.8247205e-05 | 1.41236% | 9.1400634e-05 | -3.5643e-18 | 0.08030% |
| noise_0.01_seed_102 | 0.0019730089992 | -2.6991001e-05 | 1.34955% | 7.8875151e-05 | 2.3468e-18 | 0.07699% |
| noise_0.01_seed_103 | 0.0019745779289 | -2.5422071e-05 | 1.27110% | 8.1731123e-05 | 2.1419e-19 | 0.07251% |
| noise_0.01_seed_104 | 0.0020410521918 | +4.1052192e-05 | 2.05261% | 8.6043743e-05 | -1.7545e-18 | 0.11660% |
| noise_0.02_seed_100 | 0.002063287762 | +6.3287762e-05 | 3.16439% | 3.5894583e-04 | -3.4719e-19 | 0.17951% |
| noise_0.02_seed_101 | 0.0020565235985 | +5.6523598e-05 | 2.82618% | 3.6524752e-04 | -3.6737e-19 | 0.16039% |
| noise_0.02_seed_102 | 0.0019460061877 | -5.3993812e-05 | 2.69969% | 3.1524899e-04 | 5.6146e-18 | 0.15429% |
| noise_0.02_seed_103 | 0.0019491666133 | -5.0833387e-05 | 2.54167% | 3.2693474e-04 | 2.2914e-18 | 0.14523% |
| noise_0.02_seed_104 | 0.0020822620699 | +8.2262070e-05 | 4.11310% | 3.4382204e-04 | 9.4671e-19 | 0.23305% |
| noise_0.05_seed_100 | 0.0021590826149 | +1.5908261e-04 | 7.95413% | 2.2386066e-03 | -4.0731e-19 | 0.44851% |
| noise_0.05_seed_101 | 0.0021415249537 | +1.4152495e-04 | 7.07625% | 2.2736755e-03 | -4.9223e-18 | 0.39945% |
| noise_0.05_seed_102 | 0.0018649321443 | -1.3506786e-04 | 6.75339% | 1.9637451e-03 | -1.6526e-18 | 0.38795% |
| noise_0.05_seed_103 | 0.0018730020088 | -1.2699799e-04 | 6.34990% | 2.0415268e-03 | 5.5103e-19 | 0.36458% |
| noise_0.05_seed_104 | 0.0022068404045 | +2.0684040e-04 | 10.34202% | 2.1400835e-03 | 1.0058e-19 | 0.58141% |
| times_5 | 0.002 | -4.7704896e-18 | 0.00000% | 2.6027069e-32 | 6.5846e-19 | 0.00000% |
| times_3 | 0.002 | -6.5052130e-18 | 0.00000% | 3.0564317e-32 | -2.4200e-19 | 0.00000% |
| times_2 | 0.002 | -6.5052130e-18 | 0.00000% | 2.8050789e-32 | 6.6773e-20 | 0.00000% |
| sensors_16_layout_200 | 0.002 | -4.7704896e-18 | 0.00000% | 2.3441552e-32 | 9.1437e-19 | 0.00000% |
| sensors_16_layout_201 | 0.002 | -4.7704896e-18 | 0.00000% | 2.6844399e-32 | 3.2560e-18 | 0.00000% |
| sensors_16_layout_202 | 0.002 | -6.5052130e-18 | 0.00000% | 3.5195110e-32 | 3.5463e-19 | 0.00000% |
| sensors_16_layout_203 | 0.002 | -3.0357661e-18 | 0.00000% | 1.4946449e-32 | 1.5827e-18 | 0.00000% |
| sensors_16_layout_204 | 0.002 | -4.7704896e-18 | 0.00000% | 2.6766467e-32 | -3.6883e-19 | 0.00000% |
| sensors_8_layout_200 | 0.002 | -3.0357661e-18 | 0.00000% | 2.6167429e-32 | 1.0137e-18 | 0.00000% |
| sensors_8_layout_201 | 0.002 | -6.5052130e-18 | 0.00000% | 4.0938070e-32 | 2.3784e-19 | 0.00000% |
| sensors_8_layout_202 | 0.002 | -6.5052130e-18 | 0.00000% | 4.5126835e-32 | -4.6855e-19 | 0.00000% |
| sensors_8_layout_203 | 0.002 | -4.7704896e-18 | 0.00000% | 2.3362687e-32 | 1.9285e-19 | 0.00000% |
| sensors_8_layout_204 | 0.002 | -1.3010426e-18 | 0.00000% | 5.6095640e-33 | -1.0516e-18 | 0.00000% |
| sensors_4_layout_200 | 0.002 | -4.7704896e-18 | 0.00000% | 3.2844220e-32 | 5.2890e-19 | 0.00000% |
| sensors_4_layout_201 | 0.002 | -8.2399365e-18 | 0.00000% | 5.6174390e-32 | -6.5591e-18 | 0.00000% |
| sensors_4_layout_202 | 0.002 | -8.2399365e-18 | 0.00000% | 4.8332847e-32 | -6.1015e-18 | 0.00000% |
| sensors_4_layout_203 | 0.002 | -1.3010426e-18 | 0.00000% | 2.0714192e-32 | 1.2167e-18 | 0.00000% |
| sensors_4_layout_204 | 0.002 | -1.3010426e-18 | 0.00000% | 3.2782701e-33 | -1.3135e-18 | 0.00000% |
| combined_1_seed_100 | 0.0019677030775 | -3.2296923e-05 | 1.61485% | 6.9232711e-05 | -1.2304e-18 | 0.09216% |
| combined_1_seed_101 | 0.002021731207 | +2.1731207e-05 | 1.08656% | 6.5033327e-05 | -6.1956e-19 | 0.06180% |
| combined_1_seed_102 | 0.0019598965365 | -4.0103464e-05 | 2.00517% | 8.0258687e-05 | 8.4009e-19 | 0.11449% |
| combined_1_seed_103 | 0.0020033978017 | +3.3978017e-06 | 0.16989% | 6.6214531e-05 | 9.0728e-19 | 0.00967% |
| combined_1_seed_104 | 0.0020111965576 | +1.1196558e-05 | 0.55983% | 9.2528406e-05 | -2.6177e-19 | 0.03186% |
| combined_2_seed_100 | 0.0021522820031 | +1.5228200e-04 | 7.61410% | 1.6435310e-04 | 1.3972e-18 | 0.42952% |
| combined_2_seed_101 | 0.001830876034 | -1.6912397e-04 | 8.45620% | 1.9272097e-04 | 6.7204e-19 | 0.48682% |
| combined_2_seed_102 | 0.0015488238853 | -4.5117611e-04 | 22.55881% | 2.2803526e-04 | 2.6910e-20 | 1.32239% |
| combined_2_seed_103 | 0.0017354354261 | -2.6456457e-04 | 13.22823% | 1.7883282e-04 | -3.5475e-18 | 0.76619% |
| combined_2_seed_104 | 0.0016573463347 | -3.4265367e-04 | 17.13268% | 2.4550237e-04 | -5.5300e-19 | 0.99732% |
| combined_3_seed_100 | 0.0021870590294 | +1.8705903e-04 | 9.35295% | 4.8438843e-04 | -2.9549e-18 | 0.52646% |
| combined_3_seed_101 | 0.00030280796829 | -1.6971920e-03 | 84.85960% | 2.8424686e-04 | 8.0106e-21 | 5.40255% |
| combined_3_seed_102 | 0.0024055977888 | +4.0559779e-04 | 20.27989% | 7.4423630e-04 | 1.9775e-18 | 1.12605% |
| combined_3_seed_103 | 0.0022716739964 | +2.7167400e-04 | 13.58370% | 7.9490793e-05 | -2.5085e-18 | 0.76056% |
| combined_3_seed_104 | 0.00385463865 | +1.8546387e-03 | 92.73193% | 9.2031801e-04 | -3.8825e-18 | 4.71971% |
| noise_0.05_seed_100_alternate | 0.0021590826149 | +1.5908261e-04 | 7.95413% | 2.2386066e-03 | -3.4172e-18 | 0.44851% |
| combined_3_seed_100_alternate | 0.0021870590294 | +1.8705903e-04 | 9.35295% | 4.8438843e-04 | -7.2228e-18 | 0.52646% |
| combined_3_seed_104_alternate | 0.00385463865 | +1.8546387e-03 | 92.73193% | 9.2031801e-04 | 8.2863e-18 | 4.71971% |
| continuum_clean_full | 0.0023643081638 | +3.6430816e-04 | 18.21541% | 1.8867232e-03 | -2.0251e-20 | 7.01221% |
| continuum_noise_seed_100 | 0.0023969254431 | +3.9692544e-04 | 19.84627% | 1.9842077e-03 | 7.6918e-19 | 7.01022% |
| continuum_noise_seed_101 | 0.0023930637265 | +3.9306373e-04 | 19.65319% | 2.0402390e-03 | 3.9613e-19 | 7.01040% |
| continuum_noise_seed_102 | 0.0023371022163 | +3.3710222e-04 | 16.85511% | 1.9467259e-03 | -3.0229e-19 | 7.01475% |
| continuum_noise_seed_103 | 0.002338531644 | +3.3853164e-04 | 16.92658% | 1.8640258e-03 | -1.5819e-18 | 7.01460% |
| continuum_noise_seed_104 | 0.0024065345116 | +4.0653451e-04 | 20.32673% | 1.9206899e-03 | 4.6740e-19 | 7.00985% |
| continuum_clean_sparse | 0.0013566624453 | -6.4333755e-04 | 32.16688% | 2.3408438e-03 | -5.9983e-19 | 7.67878% |
| continuum_combined_seed_100 | 0.0015057493267 | -4.9425067e-04 | 24.71253% | 1.8232168e-03 | 4.7738e-18 | 7.50337% |
| continuum_combined_seed_101 | 0.0011880735896 | -8.1192641e-04 | 40.59632% | 2.8704797e-03 | -2.2735e-19 | 7.90944% |
| continuum_combined_seed_102 | 0.00090557219807 | -1.0944278e-03 | 54.72139% | 3.0178929e-03 | -1.3693e-18 | 8.37123% |
| continuum_combined_seed_103 | 0.0010981334515 | -9.0186655e-04 | 45.09333% | 2.6612627e-03 | 2.4955e-19 | 8.04634% |
| continuum_combined_seed_104 | 0.0010015223995 | -9.9847760e-04 | 49.92388% | 2.7046823e-03 | -1.0737e-18 | 8.20397% |

| Continuum run | Physical error | Shift from clean full-data discrete optimum |
|---|---:|---:|
| continuum_clean_full | +3.64308164e-04 | +0.00000000e+00 |
| continuum_noise_seed_100 | +3.96925443e-04 | +3.26172793e-05 |
| continuum_noise_seed_101 | +3.93063727e-04 | +2.87555627e-05 |
| continuum_noise_seed_102 | +3.37102216e-04 | -2.72059475e-05 |
| continuum_noise_seed_103 | +3.38531644e-04 | -2.57765198e-05 |
| continuum_noise_seed_104 | +4.06534512e-04 | +4.22263478e-05 |
| continuum_clean_sparse | -6.43337555e-04 | -1.00764572e-03 |
| continuum_combined_seed_100 | -4.94250673e-04 | -8.58558837e-04 |
| continuum_combined_seed_101 | -8.11926410e-04 | -1.17623457e-03 |
| continuum_combined_seed_102 | -1.09442780e-03 | -1.45873597e-03 |
| continuum_combined_seed_103 | -9.01866549e-04 | -1.26617471e-03 |
| continuum_combined_seed_104 | -9.98477601e-04 | -1.36278576e-03 |

## Execution and scientific limits

Executed with Python 3.10, JAX/jaxlib 0.6.2, and pytest 9.1.1:

    .venv/bin/python -m pytest -q
    .venv/bin/python -m experiments.stage5_observation_robustness

Full pytest summary:

    45 passed in 13.17s

The 34 earlier cases pass unchanged. Eleven new cases cover zero-noise identity,
seed reproducibility and variation, exact t=0, positive-time RMS scaling,
time/sensor shapes, backward-compatible full selection, ignoring unobserved
entries, unique in-range layouts, finite masked/noisy loss and gradients,
finite-difference agreement, and short noisy optimization. The full stochastic
study is not executed by pytest. No scientific degradation trend is asserted
as a fragile unit-test invariant.

The detailed JSON records explicit JAX seeds, exact masks and times, realized
noise scales, objective diagnostics, complete optimization histories, aggregate
statistics, and acceptance checks. Rerunning the script overwrites only the
Stage 5 results file, showing IN PROGRESS until completion.

Matched-model noise robustness does not remove numerical-model bias for continuum
physics. ψ0 is exact here. Five seeds/layouts support only modest variability
claims. Sparse noiseless recovery can appear exact while sensitivity changes
substantially, and sensor placement matters as much as count. This is one scalar,
not joint or high-dimensional inference. Iid Gaussian field noise is one simple
measurement model; there is no correlated noise, uncertain initial condition,
additional model-form error, or instrument-response model. The continuum
cross-check does include the already-characterized discretization mismatch.

**PASS** for numerical soundness of the robustness study: previous behavior is preserved, baselines reproduce, fixed-seed noise/masks are reproducible, masked gradients agree, and stable finite fits have reported convergence diagnostics. PASS does not require poor-data estimates to be accurate or turn five realizations into uncertainty intervals.

Recommended Stage 6: joint inference of v and ν4 with explicit identifiability
analysis, including parameter correlations, two-dimensional loss surfaces,
sparse/noisy joint recovery, and Hessian/local conditioning. Stage 6 has not
been implemented.

Git status:

    fatal: not a git repository (or any of the parent directories): .git

No repository was initialized; no commits, pushes, or history changes were made.
