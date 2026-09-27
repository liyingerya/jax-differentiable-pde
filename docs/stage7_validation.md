# Stage 7: sensitivity-aware observation design and sensor placement

> Updated acceptance: **PASS after uniform convergence hardening**. The original-run results below are preserved; see **Convergence hardening** for separate hardened results and the updated checklist.

## Objective, motivation, and preservation

Stage 6 identified a shallow log-diffusion direction despite an accurately
identified velocity. This study asks whether observation timing and placement
can improve that direction under fixed measurement budgets, and whether local
sensitivity improvements predict held-out nonlinear recovery.

Created `src/design.py`, `experiments/stage7_observation_design.py`,
`tests/test_design.py`, `docs/stage7_results.json`, and this report; updated
`README.md`. The only change to an existing numerical module is extraction of
the existing Adam computation in `src/joint_inverse.py` into the pure
`projected_adam_kernel`, allowing independent fits to be vmapped. Its public
wrapper, defaults, validation, update equations, projection, and history format
remain equivalent. Archived Stage 6 optimizer histories match exactly in a
regression check; additional tests compare batched and individual quadratic
and PDE fits. All previous tests remain unchanged. No forward solver or inverse
logic is duplicated and no dependency was added.

## Physics, exact budgets, and candidate sets

Use [0,2π), N=32, T=2, v=1, ν4=.002, and

    ψ0=sin(x)+.5 sin(2x)+.25 cos(3x).

The validated solver uses 256 fixed steps, dt=2/256. θ=(v,q), q=log ν4.
The nominal pair is a local sensitivity design point, not an answer supplied
to the optimizer. Both budgets have exactly three total times including t=0:
eight sensors (moderate) or four sensors (severe). Only two positive times carry
parameter information; ψ0 and all t=0 values remain exact.

Enumerate all C(8,2)=28 schedules from positive candidates .25,.5,...,2.
For spatial design at [0,1,2], each budget contains 500 distinct random masks
plus the Stage 5/6 seed-200 mask and an evenly spaced comparison: 502 layouts.
Random masks use sequential explicit JAX seeds beginning at 10000 (8 sensors)
and 20000 (4 sensors). Duplicate masks are skipped deterministically until
500 unique random candidates are obtained. Every seed, sorted index list,
and score is stored. Evenly spaced four-sensor symmetry is a comparison,
not the sole design baseline.

The joint search retains the top K=5 full-field schedules by the primary
metric, then evaluates all 502 layouts at each of those schedules, separately
for each budget. K was fixed before recovery. This is an exact search over the
retained 5×502 combinations, not global optimal sensor placement or exhaustive
joint search over every possible mask and all 28 schedules.

## Jacobian, scaling, and sensitivity information matrix

For each design, y(theta) contains only selected positive-time observations,
flattened over time and sensors. J=dy/d(v,q) is computed with JAX forward-mode
autodiff through the validated solver. It has 16×2 or 8×2 entries for the two
budgets. t=0 is excluded from scoring because its derivative is zero. The full
9×32×2 derivative tensor is computed once, then indexed to score candidate
subsets; no second forward model is introduced.

Primary scaling is s_v=s_q=1. Define J_scaled=J diag(s), and

    G=J_scaledᵀ J_scaled.

We call G a deterministic **sensitivity information matrix**, not a calibrated
Fisher information matrix. Its eigenvalues, determinant, log determinant and
condition number are recorded. Numerically singular cases receive null
condition/log-determinant, not fictitious finite values.

The predeclared primary E criterion maximizes lambda_min(G). Secondary D
ranking maximizes log(det(G)); condition ranking minimizes lambda_max/lambda_min.
Recovery outcomes never enter selection. A global alternative scaling divides
each column by its RMS over all eight positive times and all 32 spatial points.
It is fixed at the nominal point and is not recomputed for each candidate or
nearby parameter. Rankings under both scalings are saved.

Full-observation column RMS values are [1.3094503558853974, 0.03001047318902372]; alternative scaling constants are [0.7636791998302528, 33.32170051772953]. These alternatives are disclosed comparisons, not a way to select whichever layout later recovers truth best. With fixed global scaling, D ranks are invariant up to a constant log-determinant shift, while E and condition ranks can change.

## Sensitivity finite-difference verification

| Sensors | Coordinate | Epsilon | Maximum absolute difference | Relative L2 difference |
|---:|---|---:|---:|---:|
| 8 | v | 1.0e-06 | 2.325956e-10 | 5.851786e-11 |
| 8 | q | 1.0e-05 | 2.017155e-11 | 2.202400e-10 |
| 4 | v | 1.0e-06 | 6.623141e-10 | 2.219968e-10 |
| 4 | q | 1.0e-05 | 4.105768e-11 | 2.845963e-10 |

## Exact temporal ranking

All 28 schedules use all 32 sensors. The top and bottom five by primary E score are:

| Rank | Times | Minimum eigenvalue | Maximum eigenvalue | Condition | Log determinant |
|---:|---|---:|---:|---:|---:|
| 1 | [0.0, 1.75, 2.0] | 1.17492940e-01 | 2.37822975e+02 | 2024.15 | 3.330150 |
| 2 | [0.0, 1.5, 2.0] | 1.06057271e-01 | 2.11554647e+02 | 1994.72 | 3.110707 |
| 3 | [0.0, 1.25, 2.0] | 9.53422477e-02 | 1.88759613e+02 | 1979.81 | 2.890192 |
| 4 | [0.0, 1.5, 1.75] | 9.41420950e-02 | 1.81967386e+02 | 1932.9 | 2.840877 |
| 5 | [0.0, 1.0, 2.0] | 8.56358242e-02 | 1.69610448e+02 | 1980.6 | 2.675853 |
| 24 | [0.0, 0.5, 1.0] | 2.69045020e-02 | 4.52460542e+01 | 1681.73 | 0.196654 |
| 25 | [0.0, 0.25, 1.0] | 2.25282314e-02 | 3.82900683e+01 | 1699.65 | -0.147795 |
| 26 | [0.0, 0.5, 0.75] | 1.85483222e-02 | 2.99359529e+01 | 1613.94 | -0.588316 |
| 27 | [0.0, 0.25, 0.75] | 1.41720516e-02 | 2.29799669e+01 | 1621.5 | -1.121861 |
| 28 | [0.0, 0.25, 0.5] | 7.56920100e-03 | 1.17254632e+01 | 1549.1 | -2.421905 |

Baseline [0,1,2] ranks 5/28 with minimum eigenvalue 8.563582418e-02. Later informative times win in this measured setup; this was not assumed by the enumerator. Earlier-time damping derivatives are smaller, but the conclusion need not generalize to a different horizon or strongly decayed field.

## Fixed-time 8-sensor candidate pool

Times remain [0,1,2]. Top ten and bottom ten layouts by primary E score:

| Rank | Layout | Sensor indices | Minimum eigenvalue | Condition | Log determinant |
|---:|---|---|---:|---:|---:|
| 1 | seed_10020 | [1, 3, 5, 6, 9, 11, 30, 31] | 3.72855325e-02 | 1658.32 | 0.835258 |
| 2 | seed_10015 | [1, 4, 5, 20, 22, 25, 29, 31] | 3.49062825e-02 | 1015.72 | 0.213178 |
| 3 | seed_10303 | [5, 8, 11, 24, 25, 26, 29, 31] | 3.47643236e-02 | 1092.7 | 0.278081 |
| 4 | seed_10470 | [0, 5, 7, 10, 22, 29, 30, 31] | 3.46740950e-02 | 1462 | 0.564039 |
| 5 | seed_10376 | [0, 4, 5, 19, 22, 23, 24, 31] | 3.46567357e-02 | 966.31 | 0.148958 |
| 6 | seed_10056 | [0, 3, 5, 10, 18, 26, 29, 30] | 3.45303542e-02 | 1169.63 | 0.332610 |
| 7 | seed_10365 | [0, 5, 10, 13, 25, 26, 28, 30] | 3.43406010e-02 | 957.725 | 0.121707 |
| 8 | seed_10177 | [4, 6, 11, 20, 24, 26, 28, 31] | 3.41633239e-02 | 974.27 | 0.128483 |
| 9 | seed_10156 | [4, 6, 18, 20, 22, 25, 30, 31] | 3.38576523e-02 | 856.582 | -0.018231 |
| 10 | seed_10479 | [1, 3, 4, 20, 23, 24, 30, 31] | 3.35676331e-02 | 1186.26 | 0.290174 |
| 493 | seed_10084 | [7, 9, 14, 15, 17, 21, 22, 29] | 9.32118439e-03 | 4733 | -0.888617 |
| 494 | seed_10409 | [7, 9, 15, 17, 19, 22, 23, 27] | 9.19956022e-03 | 4801.42 | -0.900533 |
| 495 | seed_10095 | [2, 9, 14, 16, 17, 20, 22, 29] | 9.18060351e-03 | 4588.17 | -0.950088 |
| 496 | seed_10077 | [8, 10, 11, 13, 18, 19, 21, 22] | 9.05710660e-03 | 5816.38 | -0.739978 |
| 497 | seed_10369 | [1, 2, 9, 14, 17, 18, 20, 23] | 8.77274748e-03 | 5720.06 | -0.820475 |
| 498 | seed_10311 | [1, 9, 14, 18, 22, 23, 28, 29] | 8.68740656e-03 | 5059.92 | -0.962656 |
| 499 | seed_10340 | [8, 9, 10, 17, 18, 19, 28, 29] | 8.11712421e-03 | 6880.44 | -0.791121 |
| 500 | seed_10308 | [2, 9, 10, 15, 17, 18, 24, 28] | 8.06065743e-03 | 6097.51 | -0.925885 |
| 501 | seed_10287 | [0, 1, 14, 17, 20, 21, 22, 27] | 7.77713508e-03 | 4661.61 | -1.266018 |
| 502 | seed_10180 | [3, 10, 12, 13, 17, 18, 28, 29] | 7.33057920e-03 | 4928.12 | -1.328689 |

| Comparison | E rank | Sensor indices | Minimum eigenvalue | Condition | Log determinant |
|---|---:|---|---:|---:|---:|
| baseline | 352 | [0, 7, 12, 20, 22, 23, 25, 28] | 1.67547195e-02 | 2384.42 | -0.401439 |
| evenly_spaced | 187 | [0, 4, 8, 12, 16, 20, 24, 28] | 2.14089560e-02 | 1980.6 | -0.096736 |

Distribution across all 502 masks:

| Metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| lambda_min | 0.0073305792 | 0.019739735 | 0.037285533 |
| condition_number | 696.01955 | 2110.8851 | 6880.4381 |
| log_determinant | -1.3714469 | -0.23926865 | 0.8352576 |

## Fixed-time 4-sensor candidate pool

Times remain [0,1,2]. Top ten and bottom ten layouts by primary E score:

| Rank | Layout | Sensor indices | Minimum eigenvalue | Condition | Log determinant |
|---:|---|---|---:|---:|---:|
| 1 | seed_20423 | [5, 10, 25, 31] | 2.46448270e-02 | 865.364 | -0.643226 |
| 2 | seed_20433 | [5, 6, 30, 31] | 2.30419907e-02 | 768.801 | -0.896043 |
| 3 | seed_20187 | [4, 9, 25, 31] | 2.15975772e-02 | 1183.26 | -0.594316 |
| 4 | seed_20046 | [0, 4, 6, 10] | 2.12676326e-02 | 1544.85 | -0.358457 |
| 5 | seed_20082 | [5, 25, 27, 30] | 2.09336201e-02 | 427.095 | -1.675792 |
| 6 | seed_20265 | [12, 25, 30, 31] | 2.06254457e-02 | 527.966 | -1.493427 |
| 7 | seed_20124 | [0, 4, 19, 30] | 2.00811997e-02 | 854.009 | -1.066000 |
| 8 | seed_20397 | [0, 5, 11, 24] | 1.98862092e-02 | 1095.22 | -0.836744 |
| 9 | seed_20037 | [4, 6, 24, 26] | 1.98392003e-02 | 755.455 | -1.212871 |
| 10 | seed_20092 | [3, 6, 10, 31] | 1.95181970e-02 | 1677.78 | -0.447587 |
| 493 | seed_20484 | [3, 11, 12, 13] | 2.12231723e-03 | 10908.4 | -3.013204 |
| 494 | seed_20064 | [8, 9, 14, 22] | 2.08889610e-03 | 18373.3 | -2.523583 |
| 495 | seed_20319 | [9, 10, 12, 14] | 2.06532356e-03 | 16601 | -2.647716 |
| 496 | seed_20362 | [11, 12, 13, 24] | 2.01868387e-03 | 8052.2 | -3.416919 |
| 497 | seed_20334 | [8, 9, 16, 17] | 1.90983559e-03 | 18335.4 | -2.704885 |
| 498 | seed_20485 | [2, 13, 16, 18] | 1.88933724e-03 | 8704.12 | -3.471507 |
| 499 | seed_20307 | [8, 12, 13, 16] | 1.70639298e-03 | 13915.8 | -3.205963 |
| 500 | seed_20142 | [7, 13, 14, 18] | 1.66019259e-03 | 10447.2 | -3.547555 |
| 501 | seed_20113 | [12, 13, 14, 23] | 1.46105386e-03 | 7603.86 | -4.120783 |
| 502 | seed_20091 | [13, 14, 17, 22] | 9.42686296e-04 | 9947.7 | -4.728457 |

| Comparison | E rank | Sensor indices | Minimum eigenvalue | Condition | Log determinant |
|---|---:|---|---:|---:|---:|
| baseline | 357 | [0, 7, 23, 28] | 6.33634010e-03 | 4239.47 | -1.770714 |
| evenly_spaced | 205 | [0, 8, 16, 24] | 9.91183712e-03 | 2708.72 | -1.323819 |

Distribution across all 502 masks:

| Metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| lambda_min | 0.0009426863 | 0.0088295444 | 0.024644827 |
| condition_number | 391.65233 | 2303.7881 | 18373.335 |
| log_determinant | -4.7284572 | -1.7691062 | -0.35845712 |

## Selected designs and scaling dependence

The following four families per budget were frozen before any recovery:

| Sensors | Family | Times | Sensor indices | Minimum eigenvalue | Maximum eigenvalue | Condition | Log determinant |
|---:|---|---|---|---:|---:|---:|---:|
| 8 | baseline | [0.0, 1.0, 2.0] | [0, 7, 12, 20, 22, 23, 25, 28] | 1.675471950e-02 | 3.995031425e+01 | 2384.422 | -0.401439 |
| 8 | evenly_spaced | [0.0, 1.0, 2.0] | [0, 4, 8, 12, 16, 20, 24, 28] | 2.140895604e-02 | 4.240261195e+01 | 1980.602 | -0.096736 |
| 8 | spatial_E | [0.0, 1.0, 2.0] | [1, 3, 5, 6, 9, 11, 30, 31] | 3.728553250e-02 | 6.183116344e+01 | 1658.315 | 0.835258 |
| 8 | joint_E | [0.0, 1.75, 2.0] | [1, 3, 4, 20, 23, 24, 30, 31] | 5.004811905e-02 | 4.966971453e+01 | 992.4392 | 0.910625 |
| 4 | baseline | [0.0, 1.0, 2.0] | [0, 7, 23, 28] | 6.336340100e-03 | 2.686273604e+01 | 4239.472 | -1.770714 |
| 4 | evenly_spaced | [0.0, 1.0, 2.0] | [0, 8, 16, 24] | 9.911837120e-03 | 2.684841631e+01 | 2708.723 | -1.323819 |
| 4 | spatial_E | [0.0, 1.0, 2.0] | [5, 10, 25, 31] | 2.464482697e-02 | 2.132674389e+01 | 865.3639 | -0.643226 |
| 4 | joint_E | [0.0, 1.75, 2.0] | [3, 4, 18, 30] | 3.079414732e-02 | 1.237049777e+01 | 401.7159 | -0.965116 |

Secondary joint winners are sensitivity-only comparisons; they do not replace the primary E selections during held-out recovery:

| Sensors | Scaling/criterion | Times | Sensors |
|---:|---|---|---|
| 8 | primary_E | [0.0, 1.75, 2.0] | [1, 3, 4, 20, 23, 24, 30, 31] |
| 8 | primary_D | [0.0, 1.75, 2.0] | [1, 3, 5, 6, 9, 11, 30, 31] |
| 8 | primary_condition | [0.0, 1.5, 1.75] | [3, 12, 16, 19, 23, 24, 25, 27] |
| 8 | column_normalized_E | [0.0, 1.75, 2.0] | [1, 3, 5, 6, 9, 11, 30, 31] |
| 8 | column_normalized_D | [0.0, 1.75, 2.0] | [1, 3, 5, 6, 9, 11, 30, 31] |
| 8 | column_normalized_condition | [0.0, 1.5, 2.0] | [0, 6, 11, 13, 16, 17, 24, 25] |
| 4 | primary_E | [0.0, 1.75, 2.0] | [3, 4, 18, 30] |
| 4 | primary_D | [0.0, 1.75, 2.0] | [6, 8, 9, 31] |
| 4 | primary_condition | [0.0, 1.75, 2.0] | [4, 17, 26, 28] |
| 4 | column_normalized_E | [0.0, 1.75, 2.0] | [3, 6, 10, 31] |
| 4 | column_normalized_D | [0.0, 1.75, 2.0] | [6, 8, 9, 31] |
| 4 | column_normalized_condition | [0.0, 1.5, 2.0] | [9, 22, 29, 30] |

Column normalization changes the winning E layout in both budgets. The chosen
primary layouts therefore depend on the declared coordinate scaling. This is
why scaling is stated and alternate rankings are preserved, rather than
presenting the winners as invariant or universal sensor optima.

## Neighborhood robustness without redesign

Keep selected times/masks and global scaling fixed at all nine combinations
v∈{.9,1,1.1}, ν4∈{.0015,.002,.0025}. Compare primary E-score gains against
the seed-200 baseline:

| v | ν4 | Sensors | Baseline lambda_min | Selected lambda_min | Gain |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.0015 | 8 | 9.67434398e-03 | 2.91594835e-02 | 3.01410 |
| 0.9 | 0.0015 | 4 | 4.88881859e-03 | 1.08298715e-02 | 2.21523 |
| 0.9 | 0.002 | 8 | 1.54931175e-02 | 4.62253261e-02 | 2.98360 |
| 0.9 | 0.002 | 4 | 7.91357228e-03 | 1.71716710e-02 | 2.16990 |
| 0.9 | 0.0025 | 8 | 2.18837567e-02 | 6.45270371e-02 | 2.94863 |
| 0.9 | 0.0025 | 4 | 1.12808638e-02 | 2.40034982e-02 | 2.12781 |
| 1.0 | 0.0015 | 8 | 1.03555619e-02 | 3.15668914e-02 | 3.04830 |
| 1.0 | 0.0015 | 4 | 3.85822932e-03 | 1.94506442e-02 | 5.04134 |
| 1.0 | 0.002 | 8 | 1.67547195e-02 | 5.00481191e-02 | 2.98711 |
| 1.0 | 0.002 | 4 | 6.33634010e-03 | 3.07941473e-02 | 4.85993 |
| 1.0 | 0.0025 | 8 | 2.38944842e-02 | 6.98650549e-02 | 2.92390 |
| 1.0 | 0.0025 | 4 | 9.17127902e-03 | 4.29041841e-02 | 4.67810 |
| 1.1 | 0.0015 | 8 | 1.45411520e-02 | 2.58032428e-02 | 1.77450 |
| 1.1 | 0.0015 | 4 | 5.26308169e-03 | 1.19613599e-02 | 2.27269 |
| 1.1 | 0.002 | 8 | 2.33644410e-02 | 4.10455675e-02 | 1.75675 |
| 1.1 | 0.002 | 4 | 8.68956927e-03 | 1.94241464e-02 | 2.23534 |
| 1.1 | 0.0025 | 8 | 3.30612807e-02 | 5.74847260e-02 | 1.73873 |
| 1.1 | 0.0025 | 4 | 1.26338277e-02 | 2.77400583e-02 | 2.19570 |

The selected design remains stronger at every tested nearby point. This small
neighborhood check is not global robustness, and the gain itself varies with
velocity and diffusion.

## Held-out evaluation protocol and optimizer

Primary comparison seeds are 300–319, twenty per design, never the earlier
Stages 5/6 evaluation seeds. Candidate generation uses separate ranges. The
score/recovery association uses seeds 320–324, also held out. Selection uses
only deterministic nominal sensitivities; all masks/time choices and the
association subset were frozen before any inverse recovery.

Frozen-selection SHA-256: `ea7ab7403d5a717d5e56f481cecbf708f7fdbceec50bedbfe95f4d4ea141a707`. The hash is checked again after recovery.

Moderate comparisons use 2% relative noise and eight sensors; severe comparisons
use 5% and four sensors. The same 3×M standardized Gaussian draws are reused
across equal-budget designs at each seed, assigned by sorted measurement slots.
Absolute noise scale remains sigma_rel×RMS(clean selected t>0 observations), as
in Stage 5. Thus standardized draws are paired but sigma_abs can differ by
design; this is not a common global space-time noise field. Realized RMS and
sigma_abs are recorded per fit. Initial measurements remain exact.

The loss is observed-entry normalized MSE, retaining the established denominator
mean(data_observed²)+1e-12. The primary design score is deliberately unweighted
G, not a noise-normalized likelihood metric, so its ranking need not perfectly
predict recovery with design-dependent noise scales.

Use the Stage 6 optimizer and bounds: initial pair (.7,.0005), 2000 updates,
learning rate .05, beta1=.9, beta2=.999, epsilon=1e-8, v∈[.5,1.5], ν4∈[1e-4,.02].
Fits are independent vmapped executions of the same kernel, in batches of at
most twenty. Fixed graph and noise are retained within each fit. Projection
contact, raw gradients and constrained/KKT residuals are recorded. A residual
threshold of 1e-7 is used for convergence, respecting outward gradients at
active bounds. All fit metrics below use actual final iterates.

All field errors compare the full 32-point final field to CLEAN truth, not
only the noisy sensor entries. Quantiles and sample SDs are descriptive;
no confidence intervals or Bayesian inference are claimed.

## Moderate held-out recovery

| Design | Mean v | SD(v) | Mean relative v error | Mean ν4 | SD(ν4) | Mean relative ν4 error | Median ν4 error | Empirical p90 ν4 error | Mean clean field L2 | Bound rate | Unconverged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.00072349 | 2.322740e-03 | 0.17651% | 0.002079963394 | 2.701934e-04 | 11.94648% | 11.24510% | 20.89949% | 0.85009% | 0.0% | 0 |
| evenly_spaced | 1.000865443 | 1.952079e-03 | 0.17743% | 0.001983690937 | 1.775844e-04 | 7.02304% | 6.81835% | 10.71477% | 0.66247% | 0.0% | 0 |
| spatial_E | 0.9999899113 | 2.682655e-03 | 0.21965% | 0.001917967681 | 1.885333e-04 | 7.88704% | 6.28571% | 17.66761% | 0.77073% | 0.0% | 0 |
| joint_E | 1.000665065 | 2.381281e-03 | 0.17502% | 0.001984619408 | 1.479903e-04 | 5.79792% | 4.75514% | 10.39933% | 0.63259% | 0.0% | 0 |

## Severe held-out recovery

| Design | Mean v | SD(v) | Mean relative v error | Mean ν4 | SD(ν4) | Mean relative ν4 error | Median ν4 error | Empirical p90 ν4 error | Mean clean field L2 | Bound rate | Unconverged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9993490708 | 7.624845e-03 | 0.62509% | 0.002124689138 | 9.973349e-04 | 40.16339% | 42.70087% | 79.38006% | 2.98608% | 0.0% | 0 |
| evenly_spaced | 1.000748621 | 9.307252e-03 | 0.76628% | 0.002453560358 | 7.039184e-04 | 33.05932% | 33.19964% | 70.33944% | 2.81947% | 0.0% | 0 |
| spatial_E | 1.002549608 | 9.879618e-03 | 0.79112% | 0.001746518934 | 4.475111e-04 | 20.28388% | 20.95347% | 33.98732% | 2.58988% | 0.0% | 0 |
| joint_E | 0.9993671851 | 1.231561e-02 | 0.91057% | 0.001780828136 | 5.850000e-04 | 25.01226% | 26.32085% | 48.62672% | 3.13147% | 0.0% | 0 |

The joint moderate design reduces mean absolute relative diffusion error from
11.95% to 5.80% on this ensemble. Even spacing (7.02%) outperforms spatial-only
E design (7.89%), despite the latter's larger unweighted sensitivity score.
For severe data, spatial-only E design performs best among these four families
(20.28% mean diffusion error); joint E design reaches 25.01%, versus 40.16% for
the baseline. The joint severe design worsens mean velocity error (.91% versus
.63%) and full-field error (3.13% versus 2.99%). Its larger weakest-direction
score therefore buys diffusion sensitivity at a cost elsewhere; improved
conditioning alone does not establish superior nonlinear recovery. All 160
primary fits satisfy the convergence criterion and finish away from bounds. No severe-budget primary fit contacts a bound anywhere along its optimizer trajectory either.

## Sensitivity score versus held-out recovery

Before recovery, select ten top-ranked, ten middle-ranked, and ten bottom-ranked
four-sensor layouts from the fixed-[0,1,2] primary E ranking. Evaluate each on
five separate held-out noise seeds 320–324 at 5% noise. These recovery errors
are diagnostics only and never feed design selection. Average ranks handle ties
in the descriptive Spearman calculation; no SciPy dependency or p-value is used.

| Tier | Candidate index | Sensitivity score | Median relative ν4 error |
|---|---:|---:|---:|
| top | 425 | 2.464482697e-02 | 14.08376% |
| top | 435 | 2.304199074e-02 | 3.22304% |
| top | 189 | 2.159757725e-02 | 13.42817% |
| top | 48 | 2.126763262e-02 | 15.75961% |
| top | 84 | 2.093362010e-02 | 12.73613% |
| top | 267 | 2.062544566e-02 | 22.01312% |
| top | 126 | 2.008119969e-02 | 19.00506% |
| top | 399 | 1.988620917e-02 | 28.82094% |
| top | 39 | 1.983920030e-02 | 12.34167% |
| top | 94 | 1.951819701e-02 | 21.70333% |
| middle | 18 | 8.924981109e-03 | 35.26974% |
| middle | 352 | 8.903116363e-03 | 19.05743% |
| middle | 263 | 8.873126162e-03 | 31.23721% |
| middle | 258 | 8.851487605e-03 | 30.11162% |
| middle | 490 | 8.837869510e-03 | 27.48674% |
| middle | 345 | 8.821219210e-03 | 16.82054% |
| middle | 81 | 8.792466912e-03 | 23.28214% |
| middle | 88 | 8.742301466e-03 | 45.18773% |
| middle | 460 | 8.740616882e-03 | 11.31034% |
| middle | 273 | 8.722676264e-03 | 21.27790% |
| bottom | 484 | 2.122317231e-03 | 56.23087% |
| bottom | 66 | 2.088896096e-03 | 60.58864% |
| bottom | 321 | 2.065323559e-03 | 37.86841% |
| bottom | 364 | 2.018683873e-03 | 47.53174% |
| bottom | 336 | 1.909835590e-03 | 69.42069% |
| bottom | 485 | 1.889337243e-03 | 43.60089% |
| bottom | 309 | 1.706392975e-03 | 76.91042% |
| bottom | 144 | 1.660192593e-03 | 62.37663% |
| bottom | 115 | 1.461053860e-03 | 26.85714% |
| bottom | 93 | 9.426862956e-04 | 95.00000% |

Descriptive Spearman rank correlation across these thirty designs: **-0.759288**. This is an association within a deliberately stratified subset, not proof of causal or globally optimal sensor placement. Five draws per layout give noisy median-error estimates.

| Association tier | Fits | Median of layout median errors | Final bound rate | Unconverged fits | Maximum KKT residual |
|---|---:|---:|---:|---:|---:|
| top | 50 | 14.92169% | 0.0% | 0 | 2.048156e-15 |
| middle | 50 | 25.38444% | 0.0% | 0 | 2.717233e-15 |
| bottom | 50 | 58.40976% | 8.0% | 1 | 2.151036e-07 |

All final estimates enter this descriptive association, including any unconverged fits explicitly counted above. It is therefore an empirical score-versus-this-optimizer comparison, not a claim about exact global loss minimizers.

## Mechanism from actual sensitivity vectors

The JSON stores every row of dy/dv and dy/dq for selected good, baseline and
poor designs, with times and sensor indices identifying their row order.
The following summaries separate diffusion-column strength from collinearity:

| Sensors | Design | Sum (dy/dq)² by informative time | Sum (dy/dv)² | Column cosine |
|---:|---|---|---:|---:|
| 8 | good_joint | [0.024754756, 0.027125521] | 49.66788 | -0.1878285 |
| 8 | good_spatial | [0.00659776, 0.030708058] | 61.83114 | 0.0233115 |
| 8 | baseline | [0.006137946, 0.011754186] | 39.94918 | -0.2520791 |
| 8 | poor_spatial | [0.000931128, 0.010159816] | 36.12221 | 0.5822194 |
| 4 | good_joint | [0.01662093, 0.01554812] | 12.36912 | -0.2064789 |
| 4 | good_spatial | [0.004881793, 0.020045794] | 21.32646 | 0.1064431 |
| 4 | baseline | [0.002971004, 0.005412782] | 26.86069 | -0.4941228 |
| 4 | poor_spatial | [0.0007465, 0.000328894] | 9.377429 | -0.3512717 |

In these selected designs, later observations carry accumulated damping
sensitivity and placement increases the diffusion-column norm. The four-sensor
joint design also reduces absolute column cosine from about .494 to .206,
reducing local near-collinearity; the eight-sensor case changes it from about
.252 to .188. Poor placement can instead have a small diffusion-column norm,
so collinearity alone is not the whole mechanism.

To assess Fourier content without another solver, independently differentiate
the existing forward solver for each weighted initial Fourier component. Column
norms at the selected joint observations are:

| Mode | Eight-sensor q-column norm | Four-sensor q-column norm |
|---:|---:|---:|
| 1 | 0.01191744 | 0.00948582 |
| 2 | 0.08108975 | 0.05621550 |
| 3 | 0.17089404 | 0.12663164 |

The k=3 component supplies the largest individual diffusion sensitivity in both
selected designs. Partial-sensor modes need not be orthogonal, so these norms
are not additive information fractions. The evidence supports a combination
of accumulated damping, sampling informative mode positions, and reduced
sensitivity collinearity, not an invented universal placement rule.

## Continuum-model cross-check without redesign

Use the same frozen moderate baseline and joint E designs with analytical
continuum data, 2% noise, and seeds 300–319. Keep the clean full continuum joint
optimum from Stage 6 as the discrete-model reference, distinct from physical
truth (1,.002). Changing the observation design can shift a misspecified-model
optimum even without noise; the reported observation shifts therefore include
selection effects, not just measurement noise.

Stage 6 clean joint reference: v=1.02186117041, ν4=0.00220121509265.

| Design | Mean v | Mean ν4 | Mean signed physical v error | Mean signed physical ν4 error | Shift from clean discrete v | Shift from clean discrete ν4 | Mean abs. relative ν4 error | Clean field L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1.020811727 | 0.001733131859 | +2.08117270e-02 | -2.66868141e-04 | -1.04944342e-03 | -4.68083233e-04 | 14.37884% | 4.80450% |
| joint_E | 1.029239274 | 0.002649357716 | +2.92392742e-02 | +6.49357716e-04 | +7.37810380e-03 | +4.48142624e-04 | 32.46789% | 5.16008% |

The matched-model advantage does **not** transfer to this continuum cross-check:
mean absolute relative diffusion error increases from 14.38% (baseline) to
32.47% (joint E), velocity error increases from 2.08% to 2.92%, and clean field
error increases from 4.80% to 5.16%. Both groups converge without final bound
contacts. This is consistent with the selected late-time mask emphasizing a different
projection of numerical-model mismatch; local matched sensitivities are not a guarantee of
physical parameter accuracy under a misspecified forward model. No redesign
was performed after seeing these continuum outcomes.

## Optional initial-condition check and scientific limits

The optional alternate-initial-field recovery study was not run: the required
study already evaluates 350 held-out fits. Designs are specific to the declared
multimode initial field; no generalization to different initial conditions is
claimed. The parameter-neighborhood test is not an initial-condition test.

Local design depends on nominal parameters, coordinates, scaling, and the
initial field. G is an unweighted deterministic sensitivity Gram matrix here,
not a calibrated Fisher matrix. Maximizing its smallest eigenvalue cannot
ensure globally optimal nonlinear recovery, nor can this finite pool establish
global placement optimality. Model mismatch can reduce or reverse gains from a
matched-model design. Held-out iid Gaussian synthetic noise is not real instrument
data; correlated noise and uncertain initial conditions remain untested.

## Tests, reproducibility, and conclusion

Executed:

    .venv/bin/python -m pytest -q
    .venv/bin/python -m experiments.stage7_observation_design --design-only
    .venv/bin/python -m experiments.stage7_observation_design --recover-frozen

Full pytest result: **64 passed in 21.41s** (all Stage 1–7 tests).

The default command without flags runs both stages in order. The split execution
makes the frozen design boundary explicit; recovery can be rerun from that design file. Results include all
28 time schedules, every candidate mask and score at both scalings, all retained
joint-search scores, representative Jacobians, neighborhood/modal diagnostics,
seeds, exact recovery masks/times, noise scales, parameters, convergence/bound
metrics and full clean-field errors. JSON rejects NaN/Infinity.

Experiment checks:

- sensitivity_checks: PASS
- 28_schedules: PASS
- frozen_designs: PASS
- exact_budgets: PASS
- held_out_noise: PASS
- stationary: NEEDS ATTENTION
- stable_bounds: PASS

Supplemental convergence diagnostic: candidate_336, noise seed 322, had residual 2.151036e-07 after 2000 updates. A separate rerun from the same initialization at 4000 updates reduced it to 3.079440e-12; v changed by +1.277820e-05 and nu4 by -3.649768e-06. This extra diagnostic is not substituted into any reported primary or association recovery summary. All 350 original fits retain the same iteration budget; one supplemental fit was run. The conservative experiment flag remains NEEDS ATTENTION because not every original fit met the declared stationarity cutoff. This is an optimizer limitation in one low-scoring association case, not failure of the derivative, budget, leakage, or held-out methodology checks.

## Recommended Stage 8 (not implemented)

Study robustness to observation-model assumptions: compare the current unweighted criterion with a predeclared noise-weighted design under an explicit measurement-noise convention, and test frozen designs across alternate initial fields and continuum mismatch. The severe-budget result already shows that maximizing the weakest local direction need not minimize nonlinear recovery or full-field error. Keep new evaluation seeds separate from design and protocol choices.

**Original-run verdict: NEEDS ATTENTION** under the additional all-fits stationarity check; see the convergence-hardening update below. No selected design was chosen or changed using inverse-recovery error, and identical observation budgets and held-out protocols are preserved. Performance gains, failures, and optimizer limitations are reported rather than made acceptance targets.

Git status:

    fatal: not a git repository (or any of the parent directories): .git

No commits, pushes, or Git history changes were made. Stage 8 is not implemented.

## Convergence hardening

The original Stage 7 run had exactly one fit above its predeclared projected/KKT
residual threshold after 2000 updates. This motivated a **uniform residual-only
continuation policy**, applied to all 350 recovery fits. Designs, rankings,
metrics, seeds, data, bounds, optimizer hyperparameters, and scientific
conclusions remain unchanged.

Every fit begins with exactly 2000 Adam updates. At that endpoint, use the
original projected/KKT norm and strict acceptance criterion **residual < 1e-7**.
A fit with residual >= 1e-7 continues the same optimizer state for another 2000
updates, reaching a hard cap of 4000 total updates. Fits already satisfying the
criterion receive no additional updates. No early intermediate stopping or
case-specific iteration allowance is used.

Continuation preserves theta, first moment, second moment, and the absolute
update count used for Adam bias correction. Learning rate .05, beta1 .9, beta2
.999, epsilon 1e-8, bounds, observations, and the full initialization history
are unchanged. Future production runs save the optimizer state directly.
The legacy JSON did not contain moments: for legacy fits requiring continuation,
the identical first 2000 updates are replayed from the original (.7,.0005)
initialization to reconstruct the missing state. The endpoint parameters and
gradient must agree with the stored result before continuation is allowed.
This is state reconstruction followed by continuation, not a restart from the
2000-step parameters with zero moments or a favorable new initialization.

The original `recovery`, `association_pairs`, `association_rank_correlation`,
`checks`, design tables, and earlier diagnostic remain historical records.
New `hardening.recovery` rows contain hardened iterates and per-fit metadata:
initial budget, whether continuation was required, total updates, final residual,
parameter changes, and state provenance. Original summaries remain under
`recovery`; hardened summaries are under `hardening.recovery`. The overall
`experiment_conclusion` now refers to the hardened acceptance checks; the old
verdict is preserved as `original_experiment_conclusion`.

**1 of 350 fits required continuation**; all others stopped at 2000 updates.

| Group / design / seed | Quantity | Original 2000 updates | Hardened 4000 updates |
|---|---|---:|---:|
| association_bottom / candidate_336 / 322 | v | 1.00028769002463 | 1.00030046822169 |
| association_bottom / candidate_336 / 322 | q | -8.43138308159062 | -8.44827313058997 |
| association_bottom / candidate_336 / 322 | nu4 | 0.000217919880887686 | 0.000214270112569274 |
| association_bottom / candidate_336 / 322 | kkt_residual | 2.15103575939666e-07 | 3.07944047000259e-12 |

Reconstruction maximum parameter difference: 7.105427e-15; maximum gradient difference: 2.496714e-17. Continuation parameter changes: {'v': 1.2778197064022834e-05, 'q': -0.016890048999355756, 'nu4': -3.649768318412708e-06}. Both moments and the step-2000 counter are recorded in `state_at_continuation`.

### Original versus hardened aggregate comparisons

All original and hardened summary statistics (means, SDs, relative-error means,
medians, p90s, ranges, field errors, bounds and convergence counts) are stored
separately in JSON. The table below pools runs within each named group; the
three association tiers each contain 50 fits across ten layouts. Other groups
contain twenty fits of one design. Values are shown as original → hardened.

| Group | Mean v | Mean ν4 | Mean relative ν4 error | Mean clean field L2 | Unconverged fits |
|---|---:|---:|---:|---:|---:|
| matched_8_baseline | 1.00072349 → 1.00072349 | 0.002079963394 → 0.002079963394 | 11.9464840% → 11.9464840% | 0.8500861% → 0.8500861% | 0 → 0 |
| matched_8_evenly_spaced | 1.000865443 → 1.000865443 | 0.001983690937 → 0.001983690937 | 7.0230420% → 7.0230420% | 0.6624741% → 0.6624741% | 0 → 0 |
| matched_8_spatial_E | 0.9999899113 → 0.9999899113 | 0.001917967681 → 0.001917967681 | 7.8870408% → 7.8870408% | 0.7707335% → 0.7707335% | 0 → 0 |
| matched_8_joint_E | 1.000665065 → 1.000665065 | 0.001984619408 → 0.001984619408 | 5.7979157% → 5.7979157% | 0.6325946% → 0.6325946% | 0 → 0 |
| matched_4_baseline | 0.9993490708 → 0.9993490708 | 0.002124689138 → 0.002124689138 | 40.1633946% → 40.1633946% | 2.9860823% → 2.9860823% | 0 → 0 |
| matched_4_evenly_spaced | 1.000748621 → 1.000748621 | 0.002453560358 → 0.002453560358 | 33.0593159% → 33.0593159% | 2.8194696% → 2.8194696% | 0 → 0 |
| matched_4_spatial_E | 1.002549608 → 1.002549608 | 0.001746518934 → 0.001746518934 | 20.2838836% → 20.2838836% | 2.5898779% → 2.5898779% | 0 → 0 |
| matched_4_joint_E | 0.9993671851 → 0.9993671851 | 0.001780828136 → 0.001780828136 | 25.0122572% → 25.0122572% | 3.1314665% → 3.1314665% | 0 → 0 |
| association_top | 1.002761331 → 1.002761331 | 0.001948042502 → 0.001948042502 | 19.8476843% → 19.8476843% | 2.4100676% → 2.4100676% | 0 → 0 |
| association_middle | 1.00211255 → 1.00211255 | 0.002004892065 → 0.002004892065 | 26.7416960% → 26.7416960% | 2.4921657% → 2.4921657% | 0 → 0 |
| association_bottom | 1.004529165 → 1.004529421 | 0.002189096201 → 0.002189023205 | 66.9991309% → 67.0027807% | 4.6232149% → 4.6234780% | 1 → 0 |
| continuum_baseline | 1.020811727 → 1.020811727 | 0.001733131859 → 0.001733131859 | 14.3788448% → 14.3788448% | 4.8045039% → 4.8045039% | 0 → 0 |
| continuum_joint_E | 1.029239274 → 1.029239274 | 0.002649357716 → 0.002649357716 | 32.4678858% → 32.4678858% | 5.1600798% → 5.1600798% | 0 → 0 |

Only one per-layout aggregate changes: `candidate_336` (five seeds).

| Statistic | Original | Hardened |
|---|---:|---:|
| mean_v | 1.00382941488 | 1.00383197052 |
| sd_v | 0.00537794595277 | 0.00537584476137 |
| mean_nu4 | 0.00201330291609 | 0.00201257296243 |
| sd_nu4 | 0.0016253771134 | 0.00162638549889 |
| mean_relative_nu4_error | 0.640750257249 | 0.641115234081 |
| median_relative_nu4_error | 0.694206938456 | 0.694206938456 |
| p90_relative_nu4_error | 1.00407794821 | 1.00480790187 |
| mean_clean_field_l2 | 0.0404777165635 | 0.040504028584 |
| max_kkt_residual | 2.1510357594e-07 | 3.07944047e-12 |
| nonconverged_count | 1 | 0 |

Its mean relative diffusion error changes by 0.03649768 percentage points; its p90 changes by 0.07299537 percentage points. The median is unchanged.

### Score versus recovery and scientific materiality

The same thirty top/middle/bottom layouts and the same five held-out seeds per
layout were used. Every layout's median diffusion error is exactly unchanged.
Recomputed Spearman correlation:

| Original | Hardened | Change |
|---:|---:|---:|
| -0.759288097886541 | -0.759288097886541 | 0.0 |

All eight primary-design comparison summaries and both continuum summaries
are exactly unchanged. Thus the moderate matched-model benefit, the severe
spatial-only versus joint tradeoff, the negative score/error association, and
the reversal under continuum mismatch remain unchanged. The small change in
one low-scoring layout's mean/tail error does not change those conclusions.
No scientific result or layout has been selected again.

### Validation and updated acceptance

Executed:

    .venv/bin/python -m experiments.stage7_observation_design --harden-existing
    .venv/bin/python -m pytest -q

New tests compare split-state continuation with uninterrupted Adam on both a
quadratic loss and the actual PDE loss, including histories, moments and step
count. Additional tests cover the exact cutoff, the 4000 cap, no continuation
for passing fits, and outward-gradient KKT behavior at active bounds.

Pytest: **68 passed in 23.07s**.

- sensitivity_checks: PASS
- 28_schedules: PASS
- frozen_designs: PASS
- exact_budgets: PASS
- held_out_noise: PASS
- stationary: PASS
- stable_bounds: PASS
- within_update_cap: PASS
- historical_results_and_protocol_preserved: PASS
- scientific_conclusions_unchanged: PASS

**Updated Stage 7 acceptance: PASS.** Every hardened fit meets the original stationarity threshold within the cap; historical data, designs and held-out protocol are preserved, and scientific conclusions are materially unchanged.

Git status remains `fatal: not a git repository (or any of the parent directories): .git`. No commits or pushes were made. Stage 8 was not implemented.
