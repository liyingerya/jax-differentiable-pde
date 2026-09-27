# Stage 8: Robust Observation Design under Model Uncertainty

## Objective and motivation

Stage 7 improved matched-model sensitivity and diffusion recovery, but its
moderate joint design raised continuum mean diffusion error from about 14.38%
to 32.47%. Stage 8 tests whether a frozen maximin design across plausible
forward models, initial fields and parameter points improves cross-model
recovery. Success is methodological, not a requirement that one design win.

## Files and preserved behavior

Added `src/robust_design.py`, `experiments/stage8_robust_design.py`,
`experiments/stage8_recovery.py`, `tests/test_robust_design.py`, this report and
`docs/stage8_results.json`; updated README concisely. Existing numerical modules,
Stage 7 designs, candidate pools, historical results and continuation protocol
are preserved. All inference calls the validated finite-difference forward
solver and joint loss; there is no duplicate PDE integrator or new dependency.

## Physical setup and exact observation budgets

The domain is [0,2π), N=32, T=2. The inference solver uses dt=2/256 and 256 RK4
updates. Coordinates are θ=(v,q), q=log ν4. Every candidate has exactly three
times including t=0 and either eight sensors (moderate) or four (severe).
Informative times are chosen from .25,.5,...,2; all 28 pairs are enumerated.
The initial field remains exactly known, and t=0 has zero parameter sensitivity.

## Candidate pools and exhaustive search

Both exact archived Stage 7 pools are reused: 500 seeded random layouts plus
the seed-200 and evenly spaced comparisons. There are 502 masks per budget.
Cached full sensitivity tensors allow all 28×502=14,056 combinations per
budget to be scored. No temporal prefilter is used. This is exhaustive only
within the declared discrete pool, not globally optimal sensor placement.

| Sensors | Archived Stage 7 pool hash | Stage 8 pool hash |
| --- | --- | --- |
| 8 | b6250923e4c913037ecaba7ad7fa6198c225b00dfc96f3ea8d1161bbb94ee6a3 | b6250923e4c913037ecaba7ad7fa6198c225b00dfc96f3ea8d1161bbb94ee6a3 |
| 4 | 7956ed0847c5f0ee4bd0724f616bbe2b8f1fdcfc5de0a410926cd21e7f38fae9 | 7956ed0847c5f0ee4bd0724f616bbe2b8f1fdcfc5de0a410926cd21e7f38fae9 |

## Predeclared 12-scenario ensemble

Two forward families (validated FD and analytical continuum), two initial
fields, and three physical parameter points form exactly twelve scenarios.
Continuum fields are sampled at the same 32 physical grid locations.

- A: sin(x)+.5 sin(2x)+.25 cos(3x).
- B: .8 sin(x)+.35 cos(2x)+.30 sin(4x).
- P1: (.9,.0015); P2: (1,.002); P3: (1.1,.0025).

Held-out field C=.7 sin(x)−.4 sin(3x)+.20 cos(4x)+.10 sin(5x) and held-out
parameters (1.05,.0018) never enter the design ensemble.

| Scenario | Model | Initial | v | ν4 |
| --- | --- | --- | --- | --- |
| fd_A_P1 | fd | A | 0.9 | 0.0015 |
| fd_A_P2 | fd | A | 1.0 | 0.002 |
| fd_A_P3 | fd | A | 1.1 | 0.0025 |
| fd_B_P1 | fd | B | 0.9 | 0.0015 |
| fd_B_P2 | fd | B | 1.0 | 0.002 |
| fd_B_P3 | fd | B | 1.1 | 0.0025 |
| continuum_A_P1 | continuum | A | 0.9 | 0.0015 |
| continuum_A_P2 | continuum | A | 1.0 | 0.002 |
| continuum_A_P3 | continuum | A | 1.1 | 0.0025 |
| continuum_B_P1 | continuum | B | 0.9 | 0.0015 |
| continuum_B_P2 | continuum | B | 1.0 | 0.002 |
| continuum_B_P3 | continuum | B | 1.1 | 0.0025 |

## Analytical continuum sensitivities

For each Fourier component a sin(kx)+b cos(kx), the predictor is

    exp(−exp(q) k^4 t) [a sin(k(x−vt))+b cos(k(x−vt))].

JAX differentiates the summed exact formula directly in float64. This path
never calls the FD solver. Autodiff through the existing RK4 solver supplies
the FD family. Each cached tensor contains all nine times, 32 points and both
parameter columns; scoring excludes t=0.

## Scaling, noise assumptions and information matrices

The PRIMARY scaling is the archived Stage 7 global inverse-RMS column scaling.
It is fixed once for all candidates and all twelve scenarios. Raw [v,q] scores
are retained as diagnostics; they never replace the primary ranking.

Scaling constants: `[0.7636791998302528, 33.32170051772953]`. Reference positive-time field RMS: **0.798041742081**, computed once from nominal FD field A over all eight positive times and all 32 grid points.

The primary design variance is σ_i²=(.01 RMS_ref)²+(.03 |y_i|)², positive even
at field zeros. Each scenario supplies its own predicted y_i, with the same
fixed floor. W=diag(1/σ_i²), and G_w=J_scaledᵀ W J_scaled. The unweighted
comparison sets W=I. We use “noise-weighted sensitivity information matrix”; it
is not a posterior covariance estimate. The score treats local variances as
weights and does not include information from derivatives of the variances.

Recovery retains Stage 7's homoscedastic independent Gaussian convention:
2% (moderate) or 5% (severe) times RMS of that design's clean positive-time
observations. The same standardized draws are paired by sorted measurement
slot across equal-budget designs. Actual absolute scales differ by design and
truth family. The initial observations stay exact. This deliberately tests the
weighted design under a different recovery-noise convention; it does not claim
to be a heteroscedastic likelihood recovery experiment. The loss remains the
validated observed-entry normalized MSE.

## Scenario-relative efficiency and frozen primary criterion

For each scenario, use all eight positive times and all 32 sensors to form
its full-reference matrix. Efficiency is λmin(G_design)/λmin(G_full), using the
same weights and scaling in numerator and denominator. Large efficiency-range
violations raise an error rather than being clipped. Full references have
positive smallest eigenvalues.

The primary score is the minimum efficiency across all twelve scenarios;
select its maximum, breaking exact ties by deterministic candidate index.
`robust_noise_weighted` is the primary design. `robust_unweighted` uses the same
maximin framework with W=I and is a predeclared comparison. Secondary diagnostics
include mean/geometric-mean efficiency, worst condition, minimum raw unscaled
eigenvalue and worst scenario-relative log determinant
min_s(log det G_design,s − log det G_full,s). Singular condition/D diagnostics
are null. None of these secondary diagnostics selects the primary winner.

| Scenario | Full λmin unweighted | Full λmin weighted |
| --- | --- | --- |
| fd_A_P1 | 159.139 | 927447 |
| fd_A_P2 | 256 | 1134942 |
| fd_A_P3 | 249.2603 | 1139105 |
| fd_B_P1 | 174.4748 | 936326.8 |
| fd_B_P2 | 159.4973 | 896327.6 |
| fd_B_P3 | 148.6813 | 860441 |
| continuum_A_P1 | 173.5216 | 889118.9 |
| continuum_A_P2 | 266.0972 | 1127016 |
| continuum_A_P3 | 258.5446 | 1153285 |
| continuum_B_P1 | 183.1679 | 836567.9 |
| continuum_B_P2 | 165.4728 | 772179 |
| continuum_B_P3 | 153.1284 | 728027.6 |

## 8-sensor exhaustive rankings

### weighted: top ten and bottom ten

| Rank | Index | Times | Sensors | Worst efficiency | Mean efficiency | Limiting scenario | Worst condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13953 | [0.0, 1.75, 2.0] | [9, 11, 13, 19, 22, 27, 29, 31] | 0.1511611 | 0.187309 | continuum_B_P3 | 7.305439 |
| 2 | 14050 | [0.0, 1.75, 2.0] | [1, 7, 8, 9, 10, 17, 29, 31] | 0.1502479 | 0.2114912 | continuum_B_P3 | 9.143137 |
| 3 | 14026 | [0.0, 1.75, 2.0] | [0, 5, 7, 10, 22, 29, 30, 31] | 0.1385567 | 0.1856316 | fd_B_P1 | 6.393047 |
| 4 | 13978 | [0.0, 1.75, 2.0] | [0, 9, 10, 11, 20, 22, 27, 30] | 0.1379522 | 0.1974794 | fd_A_P1 | 4.966205 |
| 5 | 13374 | [0.0, 1.5, 2.0] | [8, 10, 14, 17, 20, 24, 28, 30] | 0.1372811 | 0.1524926 | continuum_A_P3 | 6.256786 |
| 6 | 13276 | [0.0, 1.5, 2.0] | [2, 6, 8, 9, 11, 27, 28, 31] | 0.1372652 | 0.1670761 | continuum_B_P2 | 8.262468 |
| 7 | 13297 | [0.0, 1.5, 2.0] | [6, 7, 8, 11, 13, 27, 28, 31] | 0.1365465 | 0.1621353 | continuum_B_P3 | 7.800421 |
| 8 | 14053 | [0.0, 1.75, 2.0] | [4, 9, 12, 13, 24, 27, 28, 31] | 0.1360026 | 0.1523891 | fd_A_P1 | 9.576319 |
| 9 | 13832 | [0.0, 1.75, 2.0] | [0, 2, 5, 9, 10, 23, 27, 30] | 0.1345739 | 0.1883464 | fd_B_P1 | 6.148216 |
| 10 | 13248 | [0.0, 1.5, 2.0] | [2, 8, 9, 13, 18, 28, 29, 30] | 0.1325116 | 0.1681832 | fd_A_P3 | 8.354784 |
| 14047 | 170 | [0.0, 0.25, 0.5] | [5, 13, 16, 17, 18, 19, 26, 27] | 0.0008925777 | 0.002419889 | fd_B_P3 | 120.0127 |
| 14048 | 330 | [0.0, 0.25, 0.5] | [5, 6, 7, 8, 11, 14, 15, 27] | 0.0008585337 | 0.001911523 | fd_A_P1 | 51.80044 |
| 14049 | 489 | [0.0, 0.25, 0.5] | [6, 8, 11, 20, 23, 24, 27, 29] | 0.0008570954 | 0.003568198 | fd_A_P2 | 27.74615 |
| 14050 | 224 | [0.0, 0.25, 0.5] | [2, 6, 8, 9, 11, 27, 28, 31] | 0.0008026913 | 0.004723795 | fd_A_P1 | 16.33954 |
| 14051 | 316 | [0.0, 0.25, 0.5] | [4, 5, 12, 13, 16, 24, 25, 26] | 0.0007128171 | 0.003080839 | fd_B_P2 | 106.9601 |
| 14052 | 245 | [0.0, 0.25, 0.5] | [6, 7, 8, 11, 13, 27, 28, 31] | 0.0007124063 | 0.00174369 | fd_A_P1 | 47.13438 |
| 14053 | 157 | [0.0, 0.25, 0.5] | [12, 15, 16, 17, 18, 24, 26, 30] | 0.000709868 | 0.00426719 | fd_B_P1 | 117.7395 |
| 14054 | 342 | [0.0, 0.25, 0.5] | [8, 9, 10, 17, 18, 19, 28, 29] | 0.000680412 | 0.0014297 | fd_A_P2 | 102.6095 |
| 14055 | 461 | [0.0, 0.25, 0.5] | [7, 8, 9, 12, 20, 27, 29, 30] | 0.0005928155 | 0.00275834 | fd_A_P3 | 48.04609 |
| 14056 | 274 | [0.0, 0.25, 0.5] | [5, 6, 8, 10, 26, 27, 28, 31] | 0.000572437 | 0.001814421 | fd_A_P1 | 43.61985 |

### unweighted: top ten

| Rank | Index | Times | Sensors | Worst efficiency | Mean efficiency | Limiting scenario | Worst condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13626 | [0.0, 1.75, 2.0] | [4, 5, 7, 8, 10, 12, 22, 30] | 0.1570925 | 0.1820969 | fd_A_P1 | 3.613824 |
| 2 | 13832 | [0.0, 1.75, 2.0] | [0, 2, 5, 9, 10, 23, 27, 30] | 0.1406656 | 0.1752446 | fd_B_P1 | 5.027274 |
| 3 | 14019 | [0.0, 1.75, 2.0] | [4, 6, 7, 8, 10, 23, 27, 29] | 0.1369415 | 0.1787039 | continuum_B_P3 | 5.003993 |
| 4 | 13865 | [0.0, 1.75, 2.0] | [3, 5, 7, 8, 13, 25, 29, 30] | 0.1350973 | 0.1842194 | continuum_A_P3 | 4.384185 |
| 5 | 13363 | [0.0, 1.5, 2.0] | [3, 5, 7, 8, 13, 25, 29, 30] | 0.1334985 | 0.1625717 | continuum_A_P3 | 4.694966 |
| 6 | 13992 | [0.0, 1.75, 2.0] | [1, 5, 6, 7, 10, 23, 24, 28] | 0.1326074 | 0.1451887 | fd_A_P1 | 5.612324 |
| 7 | 13963 | [0.0, 1.75, 2.0] | [1, 4, 5, 8, 9, 17, 26, 28] | 0.1320719 | 0.1568538 | fd_B_P3 | 6.035855 |
| 8 | 13517 | [0.0, 1.5, 2.0] | [4, 6, 7, 8, 10, 23, 27, 29] | 0.1313316 | 0.1662811 | fd_A_P2 | 4.324319 |
| 9 | 13330 | [0.0, 1.5, 2.0] | [0, 2, 5, 9, 10, 23, 27, 30] | 0.1301455 | 0.1549275 | fd_B_P1 | 4.781457 |
| 10 | 13461 | [0.0, 1.5, 2.0] | [1, 4, 5, 8, 9, 17, 26, 28] | 0.1297923 | 0.149587 | fd_A_P2 | 5.986872 |

### Historical and selected designs under the new metrics

| Design | Times | Sensors | Unweighted robust E | Weighted robust E | Weighted mean | Weighted limiting scenario | Worst weighted condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | [0.0, 1.0, 2.0] | [0, 7, 12, 20, 22, 23, 25, 28] | 0.0613515 | 0.04381881 | 0.0695878 | fd_B_P3 | 15.7625 |
| evenly_spaced | [0.0, 1.0, 2.0] | [0, 4, 8, 12, 16, 20, 24, 28] | 0.02150046 | 0.01132103 | 0.06748159 | fd_B_P1 | 16.23437 |
| spatial_E | [0.0, 1.0, 2.0] | [1, 3, 5, 6, 9, 11, 30, 31] | 0.08448457 | 0.06621875 | 0.1275027 | fd_B_P1 | 4.280228 |
| joint_E | [0.0, 1.75, 2.0] | [1, 3, 4, 20, 23, 24, 30, 31] | 0.03240196 | 0.01291226 | 0.0787228 | fd_B_P1 | 14.6872 |
| robust_unweighted | [0.0, 1.75, 2.0] | [4, 5, 7, 8, 10, 12, 22, 30] | 0.1570925 | 0.1165482 | 0.1595241 | fd_A_P1 | 5.367869 |
| robust_noise_weighted | [0.0, 1.75, 2.0] | [9, 11, 13, 19, 22, 27, 29, 31] | 0.1097088 | 0.1511611 | 0.187309 | continuum_B_P3 | 7.305439 |

Secondary weighted diagnostics, never used to replace the primary selection:

| Design | Geometric mean efficiency | Minimum raw weighted eigenvalue | Worst relative log determinant |
| --- | --- | --- | --- |
| baseline | 0.06808155 | 77.70075 | -5.407047 |
| evenly_spaced | 0.06188265 | 9.957315 | -7.597376 |
| spatial_E | 0.1240211 | 87.70512 | -5.427077 |
| joint_E | 0.06027651 | 106.4207 | -7.766832 |
| robust_unweighted | 0.1569961 | 105.7198 | -4.337682 |
| robust_noise_weighted | 0.1859088 | 153.6123 | -3.998442 |

### Twelve-scenario weighted efficiency matrix

| Scenario | baseline | evenly_spaced | spatial_E | joint_E | robust_unweighted | robust_noise_weighted |
| --- | --- | --- | --- | --- | --- | --- |
| fd_A_P1 | 0.08885724 | 0.07738568 | 0.1048209 | 0.1145115 | 0.1165482 | 0.1943517 |
| fd_A_P2 | 0.07709339 | 0.06627576 | 0.1592878 | 0.1143473 | 0.1653317 | 0.2052745 |
| fd_A_P3 | 0.07883814 | 0.08236018 | 0.1478257 | 0.1101186 | 0.1571422 | 0.2148334 |
| fd_B_P1 | 0.07104897 | 0.01132103 | 0.06621875 | 0.01291226 | 0.2040824 | 0.1585114 |
| fd_B_P2 | 0.05107961 | 0.06110736 | 0.1146032 | 0.0166391 | 0.1872302 | 0.2184772 |
| fd_B_P3 | 0.04381881 | 0.06313499 | 0.1625399 | 0.06440151 | 0.2122982 | 0.2130988 |
| continuum_A_P1 | 0.09010419 | 0.06528779 | 0.128361 | 0.125686 | 0.1280122 | 0.1805731 |
| continuum_A_P2 | 0.07471288 | 0.07625292 | 0.1408803 | 0.1141014 | 0.1483863 | 0.190361 |
| continuum_A_P3 | 0.07916749 | 0.07841642 | 0.1296341 | 0.09754991 | 0.1252816 | 0.1942884 |
| continuum_B_P1 | 0.0601575 | 0.07142289 | 0.09260241 | 0.01666471 | 0.1594989 | 0.1648842 |
| continuum_B_P2 | 0.05705606 | 0.08252165 | 0.1464363 | 0.04316255 | 0.162316 | 0.1618934 |
| continuum_B_P3 | 0.06311932 | 0.07429237 | 0.1368219 | 0.1145787 | 0.1481608 | 0.1511611 |

## 4-sensor exhaustive rankings

### weighted: top ten and bottom ten

| Rank | Index | Times | Sensors | Worst efficiency | Mean efficiency | Limiting scenario | Worst condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13968 | [0.0, 1.75, 2.0] | [10, 21, 28, 31] | 0.07488499 | 0.1075161 | fd_B_P1 | 8.742812 |
| 2 | 13927 | [0.0, 1.75, 2.0] | [10, 22, 27, 30] | 0.07140002 | 0.1091147 | continuum_B_P3 | 8.366323 |
| 3 | 13533 | [0.0, 1.5, 2.0] | [8, 10, 23, 28] | 0.07031963 | 0.08949438 | continuum_A_P1 | 7.078367 |
| 4 | 13466 | [0.0, 1.5, 2.0] | [10, 21, 28, 31] | 0.06831727 | 0.09095358 | fd_B_P1 | 5.968993 |
| 5 | 13084 | [0.0, 1.5, 2.0] | [0, 6, 10, 29] | 0.06790595 | 0.08885641 | continuum_B_P1 | 9.942266 |
| 6 | 13524 | [0.0, 1.5, 2.0] | [8, 10, 28, 30] | 0.06676003 | 0.1143603 | continuum_B_P3 | 9.214584 |
| 7 | 13606 | [0.0, 1.75, 2.0] | [0, 9, 28, 30] | 0.06581881 | 0.1263164 | continuum_B_P3 | 15.04646 |
| 8 | 13131 | [0.0, 1.5, 2.0] | [7, 10, 25, 28] | 0.06497544 | 0.08346054 | continuum_B_P3 | 9.742328 |
| 9 | 14026 | [0.0, 1.75, 2.0] | [8, 10, 28, 30] | 0.06314585 | 0.1343822 | continuum_B_P3 | 18.2224 |
| 10 | 13586 | [0.0, 1.75, 2.0] | [0, 6, 10, 29] | 0.06260371 | 0.09910092 | fd_B_P1 | 7.659928 |
| 14047 | 105 | [0.0, 0.25, 0.5] | [5, 6, 8, 29] | 0.0001192813 | 0.0005933185 | continuum_A_P2 | 83.05908 |
| 14048 | 364 | [0.0, 0.25, 0.5] | [11, 12, 13, 24] | 0.0001191117 | 0.0003980234 | fd_A_P1 | 66.40177 |
| 14049 | 2433 | [0.0, 0.25, 1.5] | [5, 10, 25, 31] | 9.084562e-05 | 0.004461146 | continuum_A_P1 | 284.1774 |
| 14050 | 215 | [0.0, 0.25, 0.5] | [6, 8, 12, 24] | 9.021435e-05 | 0.0005593901 | fd_A_P1 | 80.13235 |
| 14051 | 978 | [0.0, 0.25, 0.75] | [7, 11, 12, 26] | 8.860883e-05 | 0.001170567 | fd_A_P1 | 66.51088 |
| 14052 | 210 | [0.0, 0.25, 0.5] | [7, 9, 11, 13] | 8.43294e-05 | 0.000784116 | fd_A_P3 | 64.78161 |
| 14053 | 174 | [0.0, 0.25, 0.5] | [2, 3, 4, 10] | 8.208597e-05 | 0.003953928 | fd_A_P1 | 166.8418 |
| 14054 | 31 | [0.0, 0.25, 0.5] | [8, 10, 18, 29] | 8.064801e-05 | 0.0006114728 | fd_A_P2 | 72.79653 |
| 14055 | 269 | [0.0, 0.25, 0.5] | [2, 3, 5, 9] | 6.922111e-05 | 0.003896955 | fd_A_P1 | 191.5613 |
| 14056 | 187 | [0.0, 0.25, 0.5] | [9, 10, 11, 24] | 6.908926e-05 | 0.0008189056 | fd_A_P1 | 99.47631 |

### unweighted: top ten

| Rank | Index | Times | Sensors | Worst efficiency | Mean efficiency | Limiting scenario | Worst condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13157 | [0.0, 1.5, 2.0] | [5, 6, 8, 29] | 0.0751933 | 0.09182356 | fd_A_P1 | 4.430954 |
| 2 | 13750 | [0.0, 1.75, 2.0] | [6, 9, 14, 30] | 0.07438468 | 0.09332648 | fd_A_P1 | 5.653963 |
| 3 | 13187 | [0.0, 1.5, 2.0] | [3, 5, 10, 29] | 0.07167553 | 0.08401566 | fd_A_P2 | 4.732156 |
| 4 | 13586 | [0.0, 1.75, 2.0] | [0, 6, 10, 29] | 0.06936027 | 0.1005433 | fd_A_P1 | 3.574925 |
| 5 | 13541 | [0.0, 1.5, 2.0] | [4, 9, 27, 30] | 0.06912475 | 0.09199215 | fd_A_P3 | 4.237728 |
| 6 | 13212 | [0.0, 1.5, 2.0] | [2, 8, 25, 30] | 0.06886097 | 0.0798576 | continuum_A_P3 | 5.326391 |
| 7 | 13606 | [0.0, 1.75, 2.0] | [0, 9, 28, 30] | 0.0685911 | 0.1099242 | fd_B_P1 | 4.972431 |
| 8 | 13084 | [0.0, 1.5, 2.0] | [0, 6, 10, 29] | 0.06815487 | 0.09018435 | fd_A_P1 | 3.83557 |
| 9 | 13714 | [0.0, 1.75, 2.0] | [2, 8, 25, 30] | 0.06746165 | 0.08910252 | continuum_A_P3 | 4.918564 |
| 10 | 13646 | [0.0, 1.75, 2.0] | [0, 3, 9, 26] | 0.06740208 | 0.07863435 | continuum_B_P1 | 5.649851 |

### Historical and selected designs under the new metrics

| Design | Times | Sensors | Unweighted robust E | Weighted robust E | Weighted mean | Weighted limiting scenario | Worst weighted condition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | [0.0, 1.0, 2.0] | [0, 7, 23, 28] | 0.01856925 | 0.01338393 | 0.02656852 | continuum_A_P3 | 24.30928 |
| evenly_spaced | [0.0, 1.0, 2.0] | [0, 8, 16, 24] | 0.01362489 | 0.006330432 | 0.03747787 | fd_B_P1 | 15.32777 |
| spatial_E | [0.0, 1.0, 2.0] | [5, 10, 25, 31] | 0.0154011 | 0.009935394 | 0.05550047 | continuum_B_P1 | 16.22617 |
| joint_E | [0.0, 1.75, 2.0] | [3, 4, 18, 30] | 0.01831839 | 0.004756967 | 0.01734338 | fd_A_P1 | 24.01721 |
| robust_unweighted | [0.0, 1.5, 2.0] | [5, 6, 8, 29] | 0.0751933 | 0.05239903 | 0.07150532 | continuum_B_P3 | 11.86873 |
| robust_noise_weighted | [0.0, 1.75, 2.0] | [10, 21, 28, 31] | 0.06552348 | 0.07488499 | 0.1075161 | fd_B_P1 | 8.742812 |

Secondary weighted diagnostics, never used to replace the primary selection:

| Design | Geometric mean efficiency | Minimum raw weighted eigenvalue | Worst relative log determinant |
| --- | --- | --- | --- |
| baseline | 0.02538587 | 23.52274 | -7.081091 |
| evenly_spaced | 0.03398474 | 5.45863 | -8.817427 |
| spatial_E | 0.04258667 | 33.9737 | -8.437147 |
| joint_E | 0.01190875 | 9.602059 | -8.777659 |
| robust_unweighted | 0.07006005 | 61.98757 | -5.666776 |
| robust_noise_weighted | 0.1048522 | 74.18522 | -5.940652 |

### Twelve-scenario weighted efficiency matrix

| Scenario | baseline | evenly_spaced | spatial_E | joint_E | robust_unweighted | robust_noise_weighted |
| --- | --- | --- | --- | --- | --- | --- |
| fd_A_P1 | 0.03619956 | 0.0418851 | 0.04784447 | 0.004756967 | 0.08841025 | 0.08825963 |
| fd_A_P2 | 0.02884628 | 0.0476043 | 0.09915242 | 0.006875824 | 0.07886695 | 0.1304405 |
| fd_A_P3 | 0.01558844 | 0.03834611 | 0.09951191 | 0.009777197 | 0.07653413 | 0.1370484 |
| fd_B_P1 | 0.03546776 | 0.006330432 | 0.01234325 | 0.008283158 | 0.1041821 | 0.07488499 |
| fd_B_P2 | 0.02798724 | 0.03432059 | 0.01012721 | 0.009118841 | 0.06990459 | 0.1126778 |
| fd_B_P3 | 0.02116436 | 0.03372345 | 0.04880692 | 0.05124128 | 0.05806231 | 0.1470853 |
| continuum_A_P1 | 0.02697102 | 0.04209405 | 0.06000486 | 0.005937775 | 0.07460569 | 0.08429845 |
| continuum_A_P2 | 0.0192252 | 0.05454371 | 0.09648169 | 0.007641923 | 0.06918418 | 0.1305904 |
| continuum_A_P3 | 0.01338393 | 0.02549279 | 0.07126771 | 0.01077481 | 0.07633803 | 0.1221206 |
| continuum_B_P1 | 0.03564273 | 0.04102671 | 0.009935394 | 0.007479117 | 0.05442284 | 0.08464182 |
| continuum_B_P2 | 0.02711328 | 0.04605663 | 0.04893533 | 0.03527513 | 0.05515379 | 0.09214237 |
| continuum_B_P3 | 0.03123238 | 0.03831057 | 0.06159452 | 0.05095849 | 0.05239903 | 0.08600295 |

Stage 7 joint designs have low worst-case efficiencies in this expanded
ensemble. The new weighted designs improve that score by construction, but
this does not establish superior nonlinear physical recovery. Initial field B
is limiting for many designs; adding continuum alone is not the entire robust
criterion. The raw/unweighted alternative rankings remain archived.

## Noise-model-assumption sensitivity without redesign

A uses W=I (a common homoscedastic factor cancels from relative efficiency).
B uses the primary 1% floor / 3% relative model; C uses 2% / 1%; D uses .5% / 5%.
Each comparison recomputes the full-reference normalization consistently under
that assumption, without changing any times or sensors.

| Sensors | Assumption | Stage 7 joint E | Stage 8 weighted E |
| --- | --- | --- | --- |
| 8 | A_homoscedastic | 0.03240196 | 0.1097088 |
| 8 | B_primary | 0.01291226 | 0.1511611 |
| 8 | C | 0.029576 | 0.1240663 |
| 8 | D | 0.007552699 | 0.1223945 |
| 4 | A_homoscedastic | 0.01831839 | 0.06552348 |
| 4 | B_primary | 0.004756967 | 0.07488499 |
| 4 | C | 0.01662508 | 0.07262334 |
| 4 | D | 0.001495213 | 0.05477586 |

## Frozen protocol and leakage prevention

Twenty primary seeds, 500–519, were chosen before recovery to limit runtime of
48 comparison groups. Both budgets evaluate all six design families under all
four truth families. Three top, three middle and three bottom four-sensor
weighted-score candidates were frozen for a separate diagnostic using seeds
530–539, matched A and continuum A, and 5% noise. This nine-design diagnostic
is smaller than a thirty-design study; its rank association is descriptive and
has limited scope. No SciPy, confidence interval or causal claim is introduced.

Design ranking files, selected designs, scenario definitions, parameter points,
initial coefficients, scaling, noise assumptions and seed/subset protocol were
saved before recovery. Recovery imports only frozen choices and cannot invoke
design search. Resume mode skips already completed groups; it does not choose
new masks or replace outcomes.

Frozen payload SHA-256: `5b78f4434255f96c5be4a1efd6cce0da444fef7d8c8127ddbd4d1ede93ce86ed`.

Full score/ranking SHA-256: `e92c18d8ee5d75f51d5e0a39094391c66757f27b2134342e6a5d2ac14a624509`. Both are verified after recovery.

## Continuum finite-difference verification

Derivative-only checks sample sensors [1,7,18,26] at times [.25,1.75,2], at P1 and P3 for fields A and B. These validation samples do not alter any production observation budget.

| Initial | v | ν4 | Column | Step | Relative L2 error | Maximum absolute error |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.9 | 0.0015 | v | 1e-06 | 7.360458e-11 | 3.357497e-10 |
| A | 0.9 | 0.0015 | q | 1e-05 | 2.205413e-10 | 9.692545e-12 |
| A | 1.1 | 0.0025 | v | 1e-06 | 6.989669e-11 | 2.344438e-10 |
| A | 1.1 | 0.0025 | q | 1e-05 | 1.18858e-10 | 1.171735e-11 |
| B | 0.9 | 0.0015 | v | 1e-06 | 6.66364e-11 | 2.157796e-10 |
| B | 0.9 | 0.0015 | q | 1e-05 | 8.599207e-11 | 9.950187e-12 |
| B | 1.1 | 0.0025 | v | 1e-06 | 1.20195e-10 | 2.210759e-10 |
| B | 1.1 | 0.0025 | q | 1e-05 | 8.815576e-11 | 1.108863e-11 |

## FD versus continuum sensitivity comparison

At identical parameters, initial field, times and sensors, relative differences below use the continuum column norm as denominator. Cosines compare corresponding columns between models, not the two parameter columns within a model.

| Sensors | Design | Initial | Point | Relative Δ(dy/dv) | Relative Δ(dy/dq) | v-column cosine | q-column cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | joint_E | A | 1 | 0.1788316 | 0.1771138 | 0.9848584 | 0.9847694 |
| 8 | joint_E | A | 2 | 0.1947979 | 0.2053644 | 0.9812891 | 0.9797279 |
| 8 | joint_E | A | 3 | 0.1746085 | 0.3089968 | 0.984835 | 0.9603736 |
| 8 | joint_E | B | 1 | 0.3781894 | 0.6812009 | 0.9495025 | 0.7404914 |
| 8 | joint_E | B | 2 | 0.3072951 | 0.6380651 | 0.9691023 | 0.7828026 |
| 8 | joint_E | B | 3 | 0.2429845 | 0.7946867 | 0.9743192 | 0.6996619 |
| 8 | robust_noise_weighted | A | 1 | 0.1314167 | 0.2568465 | 0.991341 | 0.9664524 |
| 8 | robust_noise_weighted | A | 2 | 0.1280144 | 0.290913 | 0.9919916 | 0.9569554 |
| 8 | robust_noise_weighted | A | 3 | 0.123058 | 0.33002 | 0.9929038 | 0.9446039 |
| 8 | robust_noise_weighted | B | 1 | 0.3099535 | 0.5620377 | 0.9510125 | 0.8346119 |
| 8 | robust_noise_weighted | B | 2 | 0.2675931 | 0.6757356 | 0.9635631 | 0.7842472 |
| 8 | robust_noise_weighted | B | 3 | 0.2283583 | 0.8163096 | 0.9745938 | 0.6959817 |
| 4 | joint_E | A | 1 | 0.2622358 | 0.1191954 | 0.9787379 | 0.9940532 |
| 4 | joint_E | A | 2 | 0.3060355 | 0.1690029 | 0.9528007 | 0.9878179 |
| 4 | joint_E | A | 3 | 0.2058823 | 0.3527978 | 0.9808301 | 0.9606422 |
| 4 | joint_E | B | 1 | 0.3550084 | 0.544235 | 0.963626 | 0.8448971 |
| 4 | joint_E | B | 2 | 0.2388035 | 0.7906407 | 0.9856881 | 0.7279315 |
| 4 | joint_E | B | 3 | 0.1950237 | 0.9886728 | 0.981345 | 0.4893796 |
| 4 | robust_noise_weighted | A | 1 | 0.1220121 | 0.2807885 | 0.9925286 | 0.9604822 |
| 4 | robust_noise_weighted | A | 2 | 0.1333705 | 0.2419307 | 0.9910794 | 0.9707212 |
| 4 | robust_noise_weighted | A | 3 | 0.1350734 | 0.2495227 | 0.9908807 | 0.969343 |
| 4 | robust_noise_weighted | B | 1 | 0.2884226 | 0.5897763 | 0.9585694 | 0.8175289 |
| 4 | robust_noise_weighted | B | 2 | 0.2525255 | 0.6552031 | 0.9676781 | 0.7893636 |
| 4 | robust_noise_weighted | B | 3 | 0.2094036 | 0.7794139 | 0.9782232 | 0.7311531 |

## Recovery model and uniform continuation

All inverse fits use the same N=32 FD solver even for continuum truth.
Initial parameters are (.7,.0005), learning rate .05, beta1 .9, beta2 .999,
epsilon 1e-8; v bounds [.5,1.5], ν4 bounds [1e-4,.02]. The validated Stage 7
policy runs exactly 2000 updates, checks the existing projected/KKT residual
against 1e-7, and continues only failing fits to 4000 total updates. Both Adam
moments and its absolute bias-correction count are retained. Each record stores
original and final iterates, residuals, bounds and continuation metadata.
The same policy applies to clean and noisy fits. No favorable restarts occur.

For every primary truth/design pair, the clean zero-noise fit precedes noisy
recovery. All primary clean fits are computed before the first primary noisy
run. Diagnostic clean references likewise precede their noisy diagnostic runs.
These are optimizer-computed stationary effective parameters, not a proof of
global optimality. Any unresolved numerical failure is disclosed below.

## Clean design-specific bias

Clean parameters are compared to each family's own physical truth, including
(1.05,.0018) for the held-out parameter family. Continuum clean effective
parameters are properties of this observed, misspecified inverse model, not
new physical coefficients. Every fit retains the exactly known appropriate
initial field, including C during held-out-initial recovery.

| Budget/truth/design | Clean v | Clean ν4 | v bias | ν4 bias | KKT residual | Updates |
| --- | --- | --- | --- | --- | --- | --- |
| 8/matched_A/baseline | 1 | 0.002 | 0 | -4.77049e-18 | 9.324507e-18 | 2000 |
| 8/matched_A/evenly_spaced | 1 | 0.002 | -1.110223e-16 | -6.505213e-18 | 9.138109e-16 | 2000 |
| 8/matched_A/spatial_E | 1 | 0.002 | 0 | -8.239937e-18 | 1.820655e-16 | 2000 |
| 8/matched_A/joint_E | 1 | 0.002 | 0 | -6.505213e-18 | 3.45389e-16 | 2000 |
| 8/matched_A/robust_unweighted | 1 | 0.002 | 0 | -4.77049e-18 | 3.523032e-16 | 2000 |
| 8/matched_A/robust_noise_weighted | 1 | 0.002 | 0 | -3.035766e-18 | 1.426167e-16 | 2000 |
| 8/continuum_A/baseline | 1.020118 | 0.001664462 | 0.02011776 | -0.0003355384 | 2.019725e-16 | 2000 |
| 8/continuum_A/evenly_spaced | 1.021458 | 0.002214984 | 0.02145784 | 0.0002149838 | 6.807814e-17 | 2000 |
| 8/continuum_A/spatial_E | 1.021577 | 0.002375893 | 0.02157706 | 0.0003758928 | 1.443412e-15 | 2000 |
| 8/continuum_A/joint_E | 1.028608 | 0.002668358 | 0.02860758 | 0.0006683579 | 7.218837e-16 | 2000 |
| 8/continuum_A/robust_unweighted | 1.019287 | 0.002210799 | 0.01928686 | 0.0002107993 | 1.622222e-15 | 2000 |
| 8/continuum_A/robust_noise_weighted | 1.014292 | 0.002348289 | 0.01429244 | 0.0003482887 | 3.68586e-16 | 2000 |
| 8/continuum_C/baseline | 1.03144 | 0.002775714 | 0.03143951 | 0.0007757139 | 6.329763e-16 | 2000 |
| 8/continuum_C/evenly_spaced | 1.041555 | 0.00284693 | 0.04155458 | 0.0008469303 | 1.934497e-15 | 2000 |
| 8/continuum_C/spatial_E | 1.038446 | 0.001948557 | 0.03844609 | -5.144337e-05 | 4.64713e-16 | 2000 |
| 8/continuum_C/joint_E | 1.039604 | 0.001939487 | 0.03960402 | -6.051304e-05 | 9.072292e-16 | 2000 |
| 8/continuum_C/robust_unweighted | 1.029122 | 0.002249951 | 0.02912155 | 0.0002499512 | 8.046342e-16 | 2000 |
| 8/continuum_C/robust_noise_weighted | 1.038411 | 0.002263741 | 0.03841101 | 0.0002637407 | 2.009317e-15 | 2000 |
| 8/continuum_offnominal/baseline | 1.069201 | 0.001471756 | 0.01920106 | -0.0003282437 | 9.375114e-16 | 2000 |
| 8/continuum_offnominal/evenly_spaced | 1.072843 | 0.002015753 | 0.02284341 | 0.0002157532 | 4.323848e-16 | 2000 |
| 8/continuum_offnominal/spatial_E | 1.07359 | 0.002333217 | 0.02358977 | 0.000533217 | 1.116207e-15 | 2000 |
| 8/continuum_offnominal/joint_E | 1.08069 | 0.00270292 | 0.03068951 | 0.0009029203 | 4.401611e-16 | 2000 |
| 8/continuum_offnominal/robust_unweighted | 1.071123 | 0.002123006 | 0.02112313 | 0.0003230062 | 1.788138e-16 | 2000 |
| 8/continuum_offnominal/robust_noise_weighted | 1.065655 | 0.002122875 | 0.01565516 | 0.0003228752 | 3.38559e-15 | 2000 |
| 4/matched_A/baseline | 1 | 0.002 | 2.220446e-16 | -6.505213e-18 | 1.527829e-15 | 2000 |
| 4/matched_A/evenly_spaced | 1 | 0.002 | 0 | -6.505213e-18 | 9.780722e-17 | 2000 |
| 4/matched_A/spatial_E | 1 | 0.002 | 0 | -6.505213e-18 | 1.171056e-16 | 2000 |
| 4/matched_A/joint_E | 1 | 0.002 | -1.110223e-16 | -6.505213e-18 | 6.890225e-17 | 2000 |
| 4/matched_A/robust_unweighted | 1 | 0.002 | -1.110223e-16 | -1.301043e-18 | 1.51248e-15 | 2000 |
| 4/matched_A/robust_noise_weighted | 1 | 0.002 | 0 | -1.301043e-18 | 2.074103e-16 | 2000 |
| 4/continuum_A/baseline | 1.026914 | 0.002011399 | 0.02691417 | 1.139925e-05 | 6.841894e-16 | 2000 |
| 4/continuum_A/evenly_spaced | 1.024578 | 0.002293238 | 0.02457782 | 0.0002932378 | 9.693108e-17 | 2000 |
| 4/continuum_A/spatial_E | 1.013513 | 0.002389857 | 0.01351308 | 0.0003898573 | 3.434801e-16 | 2000 |
| 4/continuum_A/joint_E | 1.025314 | 0.002969074 | 0.02531359 | 0.0009690737 | 7.410567e-17 | 2000 |
| 4/continuum_A/robust_unweighted | 1.023592 | 0.002366581 | 0.02359161 | 0.0003665812 | 1.64038e-15 | 2000 |
| 4/continuum_A/robust_noise_weighted | 1.017389 | 0.00221147 | 0.01738862 | 0.0002114705 | 2.571722e-15 | 2000 |
| 4/continuum_C/baseline | 1.027249 | 0.002053837 | 0.02724924 | 5.383699e-05 | 7.656541e-18 | 2000 |
| 4/continuum_C/evenly_spaced | 1.040187 | 0.002532937 | 0.04018668 | 0.0005329367 | 2.148997e-16 | 2000 |
| 4/continuum_C/spatial_E | 1.035316 | 0.001366452 | 0.03531626 | -0.0006335483 | 1.578032e-15 | 2000 |
| 4/continuum_C/joint_E | 1.036313 | 0.001918057 | 0.03631298 | -8.194347e-05 | 4.947103e-16 | 2000 |
| 4/continuum_C/robust_unweighted | 1.033315 | 0.001583053 | 0.03331481 | -0.0004169473 | 1.997033e-15 | 2000 |
| 4/continuum_C/robust_noise_weighted | 1.044118 | 0.001781565 | 0.04411837 | -0.0002184353 | 3.772583e-15 | 2000 |
| 4/continuum_offnominal/baseline | 1.075236 | 0.001613155 | 0.02523565 | -0.000186845 | 3.315075e-16 | 2000 |
| 4/continuum_offnominal/evenly_spaced | 1.076583 | 0.002349792 | 0.0265831 | 0.0005497918 | 9.922834e-17 | 2000 |
| 4/continuum_offnominal/spatial_E | 1.0643 | 0.002663905 | 0.01429961 | 0.0008639054 | 3.375294e-16 | 2000 |
| 4/continuum_offnominal/joint_E | 1.078119 | 0.003034723 | 0.02811873 | 0.001234723 | 2.404985e-16 | 2000 |
| 4/continuum_offnominal/robust_unweighted | 1.076256 | 0.002228658 | 0.0262558 | 0.0004286575 | 1.847108e-15 | 2000 |
| 4/continuum_offnominal/robust_noise_weighted | 1.068664 | 0.002122328 | 0.01866435 | 0.0003223285 | 4.759617e-16 | 2000 |

## Moderate, 2% noise: all four truth families

| Truth | Design | Mean v | SD(v) | Mean v error | Median v error | Mean ν4 | SD(ν4) | Mean ν4 error | Median ν4 error | p90 ν4 error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A | baseline | 0.9993123 | 0.002989987 | 0.2263% | 0.1902% | 0.001952462 | 0.0001921966 | 7.8023% | 6.0256% | 14.9886% |
| matched_A | evenly_spaced | 0.9998098 | 0.002428919 | 0.1938% | 0.1842% | 0.001976462 | 0.0002233848 | 9.0116% | 8.6364% | 16.9044% |
| matched_A | spatial_E | 1.000608 | 0.00263886 | 0.2326% | 0.2091% | 0.002077647 | 0.0001593786 | 7.2568% | 7.1176% | 12.2246% |
| matched_A | joint_E | 1.000062 | 0.002116592 | 0.1699% | 0.1415% | 0.002014605 | 0.0001422603 | 5.9926% | 5.1304% | 10.2981% |
| matched_A | robust_unweighted | 1.000055 | 0.001829234 | 0.1490% | 0.1271% | 0.002014557 | 0.000147885 | 6.0655% | 5.6298% | 11.7310% |
| matched_A | robust_noise_weighted | 0.9999603 | 0.001203051 | 0.0918% | 0.0765% | 0.001994277 | 0.0001097785 | 3.9329% | 1.9056% | 9.7815% |
| continuum_A | baseline | 1.019436 | 0.002921653 | 1.9436% | 1.9733% | 0.001627866 | 0.000179732 | 18.6067% | 18.8861% | 29.2782% |
| continuum_A | evenly_spaced | 1.021309 | 0.002469364 | 2.1309% | 2.1038% | 0.002192893 | 0.0002399007 | 13.2686% | 13.1301% | 21.6915% |
| continuum_A | spatial_E | 1.022202 | 0.00266475 | 2.2202% | 2.2641% | 0.002448575 | 0.0001604315 | 22.4287% | 24.2167% | 32.4933% |
| continuum_A | joint_E | 1.028699 | 0.002051282 | 2.8699% | 2.8476% | 0.002684004 | 0.0001559588 | 34.2002% | 32.5981% | 44.3215% |
| continuum_A | robust_unweighted | 1.019349 | 0.001902984 | 1.9349% | 1.9502% | 0.002224937 | 0.0001578289 | 11.6872% | 11.7208% | 19.2510% |
| continuum_A | robust_noise_weighted | 1.014252 | 0.001267377 | 1.4252% | 1.4143% | 0.002341468 | 0.0001234803 | 17.0734% | 17.8388% | 25.2957% |
| continuum_C | baseline | 1.031357 | 0.00329341 | 3.1357% | 3.1687% | 0.002761303 | 9.629748e-05 | 38.0652% | 37.6785% | 45.5672% |
| continuum_C | evenly_spaced | 1.042007 | 0.002240717 | 4.2007% | 4.1998% | 0.002854467 | 8.632287e-05 | 42.7234% | 43.0515% | 47.1839% |
| continuum_C | spatial_E | 1.037887 | 0.002209063 | 3.7887% | 3.8152% | 0.002002911 | 0.000111209 | 4.4545% | 4.2770% | 7.5387% |
| continuum_C | joint_E | 1.039463 | 0.002343458 | 3.9463% | 3.9170% | 0.00192845 | 7.589339e-05 | 3.9566% | 2.8488% | 7.0260% |
| continuum_C | robust_unweighted | 1.028952 | 0.001306694 | 2.8952% | 2.9073% | 0.002259077 | 5.546447e-05 | 12.9539% | 12.6149% | 16.2221% |
| continuum_C | robust_noise_weighted | 1.038256 | 0.001371837 | 3.8256% | 3.8166% | 0.002267891 | 5.580069e-05 | 13.3946% | 14.5109% | 16.5449% |
| continuum_offnominal | baseline | 1.068435 | 0.002924031 | 1.7557% | 1.7661% | 0.001446644 | 0.0001680342 | 19.6701% | 20.0486% | 30.3997% |
| continuum_offnominal | evenly_spaced | 1.072764 | 0.002402577 | 2.1680% | 2.1305% | 0.001995464 | 0.0002461785 | 15.4142% | 14.7583% | 22.6410% |
| continuum_offnominal | spatial_E | 1.07431 | 0.002670366 | 2.3153% | 2.3481% | 0.002390684 | 0.0001605075 | 32.8158% | 31.9244% | 46.4497% |
| continuum_offnominal | joint_E | 1.080826 | 0.001854746 | 2.9358% | 2.9434% | 0.002719352 | 0.000168458 | 51.0751% | 50.2241% | 62.1961% |
| continuum_offnominal | robust_unweighted | 1.071217 | 0.002014432 | 2.0207% | 2.0299% | 0.002135245 | 0.0001775091 | 19.0327% | 19.6651% | 29.8958% |
| continuum_offnominal | robust_noise_weighted | 1.065598 | 0.001280156 | 1.4855% | 1.4736% | 0.002115617 | 0.0001336158 | 17.5343% | 18.1552% | 24.7576% |

Field prediction and optimizer diagnostics:

| Truth | Design | Clean field L2 | Final bound rate | Trajectory bound-hit rate | Continuation rate | Unconverged rate |
| --- | --- | --- | --- | --- | --- | --- |
| matched_A | baseline | 0.8563% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | evenly_spaced | 0.7925% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | spatial_E | 0.7669% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | joint_E | 0.6093% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | robust_unweighted | 0.5643% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | robust_noise_weighted | 0.3677% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | baseline | 4.9296% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | evenly_spaced | 4.5818% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | spatial_E | 4.6117% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | joint_E | 5.1211% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | robust_unweighted | 4.5592% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | robust_noise_weighted | 4.8410% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | baseline | 9.8677% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | evenly_spaced | 9.7924% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | spatial_E | 9.3993% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | joint_E | 9.4941% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | robust_unweighted | 9.6938% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | robust_noise_weighted | 9.0469% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | baseline | 5.3106% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | evenly_spaced | 4.8900% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | spatial_E | 4.9960% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | joint_E | 5.6661% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | robust_unweighted | 4.8705% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | robust_noise_weighted | 5.1378% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |

## Severe, 5% noise: all four truth families

| Truth | Design | Mean v | SD(v) | Mean v error | Median v error | Mean ν4 | SD(ν4) | Mean ν4 error | Median ν4 error | p90 ν4 error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A | baseline | 1.000911 | 0.007594051 | 0.6240% | 0.5386% | 0.00213676 | 0.001023462 | 37.8072% | 28.2170% | 77.4935% |
| matched_A | evenly_spaced | 0.9997005 | 0.01034283 | 0.7516% | 0.4806% | 0.002083509 | 0.0007052028 | 27.0315% | 21.5110% | 52.4975% |
| matched_A | spatial_E | 1.000653 | 0.009221005 | 0.7664% | 0.7208% | 0.002041075 | 0.0004269041 | 16.1362% | 13.6433% | 32.0405% |
| matched_A | joint_E | 0.9974091 | 0.01731114 | 1.3152% | 0.9682% | 0.001913468 | 0.0006102133 | 23.4884% | 21.4033% | 48.0808% |
| matched_A | robust_unweighted | 1.000846 | 0.005096802 | 0.4373% | 0.3951% | 0.002017163 | 0.000440638 | 17.7049% | 15.8831% | 33.9678% |
| matched_A | robust_noise_weighted | 1.000141 | 0.002756314 | 0.2135% | 0.1630% | 0.002059295 | 0.000329958 | 12.4212% | 9.4208% | 26.8435% |
| continuum_A | baseline | 1.027824 | 0.007935985 | 2.7824% | 2.9481% | 0.002130092 | 0.001017813 | 37.3471% | 26.3238% | 77.5385% |
| continuum_A | evenly_spaced | 1.024314 | 0.01079203 | 2.4314% | 2.5990% | 0.002385192 | 0.0007580832 | 35.0039% | 38.1469% | 48.6865% |
| continuum_A | spatial_E | 1.014128 | 0.009272582 | 1.4149% | 1.5485% | 0.002430837 | 0.0004408607 | 23.7378% | 19.7265% | 47.7763% |
| continuum_A | joint_E | 1.022688 | 0.01718403 | 2.5355% | 2.5502% | 0.00289274 | 0.0006846971 | 45.1556% | 38.5585% | 76.2357% |
| continuum_A | robust_unweighted | 1.024525 | 0.005505083 | 2.4525% | 2.5642% | 0.002381438 | 0.0004861837 | 24.2395% | 19.9066% | 51.7876% |
| continuum_A | robust_noise_weighted | 1.017505 | 0.002752408 | 1.7505% | 1.6719% | 0.002273061 | 0.0003431682 | 16.9310% | 12.4312% | 39.7122% |
| continuum_C | baseline | 1.026737 | 0.01070455 | 2.6737% | 2.4954% | 0.002059762 | 0.0004141153 | 16.9088% | 13.7862% | 32.9854% |
| continuum_C | evenly_spaced | 1.039607 | 0.0067503 | 3.9607% | 3.7700% | 0.002512866 | 0.0002721678 | 25.6433% | 26.8519% | 38.8178% |
| continuum_C | spatial_E | 1.035215 | 0.004435559 | 3.5215% | 3.6681% | 0.001383857 | 0.0002700706 | 30.8072% | 30.6920% | 49.3967% |
| continuum_C | joint_E | 1.036134 | 0.008855924 | 3.6134% | 3.6057% | 0.001910606 | 0.0003725282 | 14.5381% | 11.5875% | 27.0708% |
| continuum_C | robust_unweighted | 1.033405 | 0.003459808 | 3.3405% | 3.3587% | 0.001607554 | 0.0001984475 | 19.6467% | 18.9855% | 33.8834% |
| continuum_C | robust_noise_weighted | 1.04278 | 0.00689283 | 4.2780% | 4.4392% | 0.001842874 | 0.0003776943 | 17.6504% | 20.8277% | 27.1181% |
| continuum_offnominal | baseline | 1.07636 | 0.009708322 | 2.5105% | 2.7921% | 0.001700515 | 0.0009602084 | 42.2426% | 29.5486% | 94.5893% |
| continuum_offnominal | evenly_spaced | 1.076409 | 0.01180402 | 2.5152% | 2.7062% | 0.002450554 | 0.0008310391 | 48.6003% | 40.8909% | 74.5087% |
| continuum_offnominal | spatial_E | 1.064834 | 0.009631939 | 1.4141% | 1.5513% | 0.002696896 | 0.000465333 | 49.8275% | 44.4231% | 79.4644% |
| continuum_offnominal | joint_E | 1.075419 | 0.0163705 | 2.6037% | 2.7041% | 0.002999773 | 0.0008206425 | 66.6540% | 53.4416% | 111.9753% |
| continuum_offnominal | robust_unweighted | 1.077348 | 0.006322098 | 2.6046% | 2.6582% | 0.002239207 | 0.0005573017 | 33.2451% | 28.5113% | 61.9335% |
| continuum_offnominal | robust_noise_weighted | 1.068837 | 0.002811422 | 1.7940% | 1.7306% | 0.002177608 | 0.0003245728 | 22.0280% | 19.8787% | 47.8957% |

Field prediction and optimizer diagnostics:

| Truth | Design | Clean field L2 | Final bound rate | Trajectory bound-hit rate | Continuation rate | Unconverged rate |
| --- | --- | --- | --- | --- | --- | --- |
| matched_A | baseline | 2.9967% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | evenly_spaced | 2.9816% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | spatial_E | 2.3782% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | joint_E | 3.9674% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | robust_unweighted | 1.6253% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| matched_A | robust_noise_weighted | 1.0021% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | baseline | 5.8532% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | evenly_spaced | 5.6445% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | spatial_E | 5.4273% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | joint_E | 6.4545% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | robust_unweighted | 4.9546% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_A | robust_noise_weighted | 4.7243% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | baseline | 10.9807% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | evenly_spaced | 9.4553% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | spatial_E | 12.4301% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | joint_E | 10.4058% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | robust_unweighted | 11.1649% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_C | robust_noise_weighted | 10.2545% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | baseline | 6.1910% | 5.0000% | 5.0000% | 5.0000% | 0.0000% |
| continuum_offnominal | evenly_spaced | 6.1681% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | spatial_E | 5.9692% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | joint_E | 6.9002% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | robust_unweighted | 5.4293% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| continuum_offnominal | robust_noise_weighted | 5.0293% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |

## Physical error = clean model/design bias + noise-induced shift

The JSON stores this identity for every noisy run in both (v,ν4) and (v,q)
coordinates. The table shows ensemble-mean signed quantities in physical
(v,ν4) coordinates. Absolute-error statistics in recovery tables do not obey
this signed additive identity and are not substituted here.

| Budget/truth/design | Physical Δv | Clean v bias | Noise Δv | Physical Δν4 | Clean ν4 bias | Noise Δν4 |
| --- | --- | --- | --- | --- | --- | --- |
| 8/matched_A/baseline | -0.0006877419 | 0 | -0.0006877419 | -4.753834e-05 | -4.77049e-18 | -4.753834e-05 |
| 8/matched_A/evenly_spaced | -0.0001901938 | -1.110223e-16 | -0.0001901938 | -2.353827e-05 | -6.505213e-18 | -2.353827e-05 |
| 8/matched_A/spatial_E | 0.0006077724 | 0 | 0.0006077724 | 7.764721e-05 | -8.239937e-18 | 7.764721e-05 |
| 8/matched_A/joint_E | 6.247428e-05 | 0 | 6.247428e-05 | 1.460481e-05 | -6.505213e-18 | 1.460481e-05 |
| 8/matched_A/robust_unweighted | 5.484045e-05 | 0 | 5.484045e-05 | 1.455719e-05 | -4.77049e-18 | 1.455719e-05 |
| 8/matched_A/robust_noise_weighted | -3.970724e-05 | 0 | -3.970724e-05 | -5.723078e-06 | -3.035766e-18 | -5.723078e-06 |
| 8/continuum_A/baseline | 0.0194364 | 0.02011776 | -0.0006813613 | -0.0003721343 | -0.0003355384 | -3.659597e-05 |
| 8/continuum_A/evenly_spaced | 0.02130855 | 0.02145784 | -0.0001492908 | 0.0001928929 | 0.0002149838 | -2.209093e-05 |
| 8/continuum_A/spatial_E | 0.02220242 | 0.02157706 | 0.00062536 | 0.0004485749 | 0.0003758928 | 7.26821e-05 |
| 8/continuum_A/joint_E | 0.02869915 | 0.02860758 | 9.156832e-05 | 0.0006840039 | 0.0006683579 | 1.564597e-05 |
| 8/continuum_A/robust_unweighted | 0.01934925 | 0.01928686 | 6.239831e-05 | 0.0002249371 | 0.0002107993 | 1.413783e-05 |
| 8/continuum_A/robust_noise_weighted | 0.01425229 | 0.01429244 | -4.01578e-05 | 0.0003414684 | 0.0003482887 | -6.820372e-06 |
| 8/continuum_C/baseline | 0.0313573 | 0.03143951 | -8.221272e-05 | 0.0007613033 | 0.0007757139 | -1.441063e-05 |
| 8/continuum_C/evenly_spaced | 0.04200699 | 0.04155458 | 0.000452406 | 0.0008544675 | 0.0008469303 | 7.537186e-06 |
| 8/continuum_C/spatial_E | 0.03788744 | 0.03844609 | -0.0005586443 | 2.911239e-06 | -5.144337e-05 | 5.435461e-05 |
| 8/continuum_C/joint_E | 0.03946258 | 0.03960402 | -0.0001414396 | -7.155039e-05 | -6.051304e-05 | -1.103734e-05 |
| 8/continuum_C/robust_unweighted | 0.0289515 | 0.02912155 | -0.0001700449 | 0.0002590771 | 0.0002499512 | 9.125884e-06 |
| 8/continuum_C/robust_noise_weighted | 0.03825646 | 0.03841101 | -0.0001545484 | 0.0002678914 | 0.0002637407 | 4.15071e-06 |
| 8/continuum_offnominal/baseline | 0.01843482 | 0.01920106 | -0.0007662467 | -0.0003533559 | -0.0003282437 | -2.511218e-05 |
| 8/continuum_offnominal/evenly_spaced | 0.02276421 | 0.02284341 | -7.919578e-05 | 0.0001954642 | 0.0002157532 | -2.028908e-05 |
| 8/continuum_offnominal/spatial_E | 0.02431032 | 0.02358977 | 0.0007205493 | 0.0005906843 | 0.000533217 | 5.746724e-05 |
| 8/continuum_offnominal/joint_E | 0.03082603 | 0.03068951 | 0.000136516 | 0.0009193523 | 0.0009029203 | 1.6432e-05 |
| 8/continuum_offnominal/robust_unweighted | 0.02121707 | 0.02112313 | 9.39367e-05 | 0.0003352448 | 0.0003230062 | 1.22386e-05 |
| 8/continuum_offnominal/robust_noise_weighted | 0.01559788 | 0.01565516 | -5.727425e-05 | 0.0003156175 | 0.0003228752 | -7.257747e-06 |
| 4/matched_A/baseline | 0.0009112875 | 2.220446e-16 | 0.0009112875 | 0.0001367604 | -6.505213e-18 | 0.0001367604 |
| 4/matched_A/evenly_spaced | -0.000299469 | 0 | -0.000299469 | 8.35094e-05 | -6.505213e-18 | 8.35094e-05 |
| 4/matched_A/spatial_E | 0.0006534033 | 0 | 0.0006534033 | 4.107496e-05 | -6.505213e-18 | 4.107496e-05 |
| 4/matched_A/joint_E | -0.002590934 | -1.110223e-16 | -0.002590934 | -8.653161e-05 | -6.505213e-18 | -8.653161e-05 |
| 4/matched_A/robust_unweighted | 0.0008456126 | -1.110223e-16 | 0.0008456126 | 1.716337e-05 | -1.301043e-18 | 1.716337e-05 |
| 4/matched_A/robust_noise_weighted | 0.0001407301 | 0 | 0.0001407301 | 5.929452e-05 | -1.301043e-18 | 5.929452e-05 |
| 4/continuum_A/baseline | 0.02782437 | 0.02691417 | 0.0009102081 | 0.0001300922 | 1.139925e-05 | 0.0001186929 |
| 4/continuum_A/evenly_spaced | 0.02431356 | 0.02457782 | -0.0002642635 | 0.0003851921 | 0.0002932378 | 9.195428e-05 |
| 4/continuum_A/spatial_E | 0.0141283 | 0.01351308 | 0.0006152231 | 0.000430837 | 0.0003898573 | 4.097966e-05 |
| 4/continuum_A/joint_E | 0.02268767 | 0.02531359 | -0.002625923 | 0.0008927402 | 0.0009690737 | -7.633352e-05 |
| 4/continuum_A/robust_unweighted | 0.02452469 | 0.02359161 | 0.0009330742 | 0.0003814378 | 0.0003665812 | 1.485664e-05 |
| 4/continuum_A/robust_noise_weighted | 0.01750534 | 0.01738862 | 0.0001167254 | 0.0002730613 | 0.0002114705 | 6.159087e-05 |
| 4/continuum_C/baseline | 0.02673666 | 0.02724924 | -0.0005125806 | 5.976159e-05 | 5.383699e-05 | 5.924597e-06 |
| 4/continuum_C/evenly_spaced | 0.0396073 | 0.04018668 | -0.0005793796 | 0.0005128664 | 0.0005329367 | -2.007028e-05 |
| 4/continuum_C/spatial_E | 0.03521531 | 0.03531626 | -0.0001009495 | -0.0006161434 | -0.0006335483 | 1.740485e-05 |
| 4/continuum_C/joint_E | 0.03613374 | 0.03631298 | -0.0001792338 | -8.93936e-05 | -8.194347e-05 | -7.45013e-06 |
| 4/continuum_C/robust_unweighted | 0.03340528 | 0.03331481 | 9.046914e-05 | -0.0003924464 | -0.0004169473 | 2.450089e-05 |
| 4/continuum_C/robust_noise_weighted | 0.04277991 | 0.04411837 | -0.001338468 | -0.0001571257 | -0.0002184353 | 6.130956e-05 |
| 4/continuum_offnominal/baseline | 0.02635993 | 0.02523565 | 0.001124273 | -9.948518e-05 | -0.000186845 | 8.735984e-05 |
| 4/continuum_offnominal/evenly_spaced | 0.02640932 | 0.0265831 | -0.0001737875 | 0.0006505543 | 0.0005497918 | 0.0001007625 |
| 4/continuum_offnominal/spatial_E | 0.01483396 | 0.01429961 | 0.000534343 | 0.0008968958 | 0.0008639054 | 3.299038e-05 |
| 4/continuum_offnominal/joint_E | 0.02541882 | 0.02811873 | -0.002699913 | 0.001199773 | 0.001234723 | -3.495026e-05 |
| 4/continuum_offnominal/robust_unweighted | 0.02734804 | 0.0262558 | 0.00109224 | 0.0004392073 | 0.0004286575 | 1.054975e-05 |
| 4/continuum_offnominal/robust_noise_weighted | 0.01883659 | 0.01866435 | 0.0001722354 | 0.0003776079 | 0.0003223285 | 5.527946e-05 |

## Stage 7 reversal versus frozen robust design

| Sensors | Truth | Stage 7 joint mean ν4 error | Stage 8 weighted mean ν4 error | Stage 7 joint field L2 | Stage 8 weighted field L2 |
| --- | --- | --- | --- | --- | --- |
| 8 | matched_A | 5.9926% | 3.9329% | 0.6093% | 0.3677% |
| 8 | continuum_A | 34.2002% | 17.0734% | 5.1211% | 4.8410% |
| 4 | matched_A | 23.4884% | 12.4212% | 3.9674% | 1.0021% |
| 4 | continuum_A | 45.1556% | 16.9310% | 6.4545% | 4.7243% |

Recovery absolute-noise scales for the two focal designs (constant across seeds within a group):

| Sensors | Truth | Stage 7 joint sigma_abs | Stage 8 weighted sigma_abs |
| --- | --- | --- | --- |
| 8 | matched_A | 0.01666434 | 0.01077406 |
| 8 | continuum_A | 0.01605565 | 0.01099278 |
| 8 | continuum_C | 0.0122518 | 0.008217017 |
| 8 | continuum_offnominal | 0.01539376 | 0.01088565 |
| 4 | matched_A | 0.05577265 | 0.01817006 |
| 4 | continuum_A | 0.05434823 | 0.01820805 |
| 4 | continuum_C | 0.03871703 | 0.01939145 |
| 4 | continuum_offnominal | 0.05309351 | 0.01799651 |

Thus relative-noise comparisons combine placement effects with design-dependent absolute noise amplitudes. For moderate matched A, sigma_abs is .0166643 for joint E and .0107741 for robust weighted; gains cannot be attributed solely to an unweighted column norm at a common absolute noise level.

## Post-hoc worst-case recovery across four truth families

These maxima are evaluation metrics only. They never feed design selection.

| Sensors | Design | Worst mean ν4 error | Worst p90 ν4 error | Worst mean v error | Worst field L2 |
| --- | --- | --- | --- | --- | --- |
| 8 | baseline | 38.0652% | 45.5672% | 3.1357% | 9.8677% |
| 8 | evenly_spaced | 42.7234% | 47.1839% | 4.2007% | 9.7924% |
| 8 | spatial_E | 32.8158% | 46.4497% | 3.7887% | 9.3993% |
| 8 | joint_E | 51.0751% | 62.1961% | 3.9463% | 9.4941% |
| 8 | robust_unweighted | 19.0327% | 29.8958% | 2.8952% | 9.6938% |
| 8 | robust_noise_weighted | 17.5343% | 25.2957% | 3.8256% | 9.0469% |
| 4 | baseline | 42.2426% | 94.5893% | 2.7824% | 10.9807% |
| 4 | evenly_spaced | 48.6003% | 74.5087% | 3.9607% | 9.4553% |
| 4 | spatial_E | 49.8275% | 79.4644% | 3.5215% | 12.4301% |
| 4 | joint_E | 66.6540% | 111.9753% | 3.6134% | 10.4058% |
| 4 | robust_unweighted | 33.2451% | 61.9335% | 3.3405% | 11.1649% |
| 4 | robust_noise_weighted | 22.0280% | 47.8957% | 4.2780% | 10.2545% |

## Robust score versus cross-model recovery

| Tier | Candidate | Robust score | Matched median ν4 error | Continuum median ν4 error | Worst median ν4 error |
| --- | --- | --- | --- | --- | --- |
| top | 13968 | 0.07488499 | 12.7283% | 6.5343% | 12.7283% |
| top | 13927 | 0.07140002 | 6.9144% | 6.1970% | 6.9144% |
| top | 13533 | 0.07031963 | 14.6549% | 22.5224% | 22.5224% |
| middle | 5534 | 0.005427431 | 64.9240% | 74.7067% | 74.7067% |
| middle | 5153 | 0.005423455 | 46.0509% | 47.7949% | 47.7949% |
| middle | 9482 | 0.005421412 | 39.2935% | 92.3147% | 92.3147% |
| bottom | 31 | 8.064801e-05 | 101.5768% | 104.6997% | 104.6997% |
| bottom | 269 | 6.922111e-05 | 113.1963% | 95.0000% | 113.1963% |
| bottom | 187 | 6.908926e-05 | 95.0000% | 95.0000% | 95.0000% |

Descriptive Spearman correlation: **-0.91666667**. This nine-design stratified diagnostic measures association for this fixed solver, optimizer, initial field and noise convention, not causal or universal optimality.

Diagnostic final-bound rates (matched / continuum) are shown below. Several bottom-tier median errors are 95%, the relative error of the lower-bound estimate nu4=.0001 against truth .002. The association includes these constrained estimates; it is not an unconstrained local-covariance validation.

| Candidate | Matched final-bound rate | Continuum final-bound rate |
| --- | --- | --- |
| 13968 | 0.0000% | 0.0000% |
| 13927 | 0.0000% | 0.0000% |
| 13533 | 0.0000% | 0.0000% |
| 5534 | 10.0000% | 10.0000% |
| 5153 | 0.0000% | 0.0000% |
| 9482 | 0.0000% | 0.0000% |
| 31 | 20.0000% | 20.0000% |
| 269 | 50.0000% | 60.0000% |
| 187 | 20.0000% | 30.0000% |

## Mechanism from measured scenario sensitivities

Full per-scenario Jacobian arrays, column strengths/cosines, raw and weighted time contributions, modal column norms, and noise-weight ranges are in JSON. The table below shows nominal P2 for each model and design initial field. Partial-sensor modes need not be orthogonal; modal norms are not additive information fractions.

| Sensors | Design | Scenario | v norm | q norm | Column cosine | q squared norm by time | Modal q norms | Weight range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | joint_E | fd_A_P2 | 7.047544 | 0.2277724 | -0.1878285 | [0.0247548, 0.0271255] | {'1': 0.0119174, '2': 0.0810898, '3': 0.170894} | 495.2–1.554e+04 |
| 8 | joint_E | fd_B_P2 | 4.753044 | 0.2971648 | 0.7058781 | [0.0344566, 0.0538503] | {'1': 0.009534, '2': 0.052543, '4': 0.3142859} | 863.6–4077 |
| 8 | joint_E | continuum_A_P2 | 6.971814 | 0.2222369 | -0.1742815 | [0.0265488, 0.0228404] | {'1': 0.0120141, '2': 0.0836735, '3': 0.1676351} | 515.2–1.568e+04 |
| 8 | joint_E | continuum_B_P2 | 6.048455 | 0.3216437 | 0.5922359 | [0.0540139, 0.0494407] | {'1': 0.0096113, '2': 0.0533745, '4': 0.3395344} | 790.8–5321 |
| 8 | robust_noise_weighted | fd_A_P2 | 7.842164 | 0.1675854 | 0.06271366 | [0.0131572, 0.0149277] | {'1': 0.0092285, '2': 0.0833148, '3': 0.152595} | 998.6–1.563e+04 |
| 8 | robust_noise_weighted | fd_B_P2 | 7.053913 | 0.3471677 | 0.1609206 | [0.0596853, 0.0608401] | {'1': 0.0073828, '2': 0.0508087, '4': 0.3246691} | 863.6–1.555e+04 |
| 8 | robust_noise_weighted | continuum_A_P2 | 8.075244 | 0.1715647 | 0.1575019 | [0.0139418, 0.0154927] | {'1': 0.0092511, '2': 0.0838274, '3': 0.1579309} | 948.2–1.415e+04 |
| 8 | robust_noise_weighted | continuum_B_P2 | 7.262307 | 0.3300245 | -0.04664193 | [0.0617084, 0.0472078] | {'1': 0.0074009, '2': 0.0532561, '4': 0.3099281} | 859.1–1.559e+04 |
| 4 | joint_E | fd_A_P2 | 3.516976 | 0.1793573 | -0.2064789 | [0.0166209, 0.0155481] | {'1': 0.0094858, '2': 0.0562155, '3': 0.1266316} | 495.2–1.491e+04 |
| 4 | joint_E | fd_B_P2 | 4.121766 | 0.2173538 | 0.7650229 | [0.0274062, 0.0198365] | {'1': 0.0075887, '2': 0.0379856, '4': 0.2405202} | 1093–3065 |
| 4 | joint_E | continuum_A_P2 | 3.547621 | 0.1702097 | 0.04690929 | [0.0176792, 0.0112921] | {'1': 0.0095546, '2': 0.0553624, '3': 0.1173327} | 515.2–1.385e+04 |
| 4 | joint_E | continuum_B_P2 | 5.047791 | 0.1937842 | 0.6073702 | [0.0261951, 0.0113572] | {'1': 0.0076437, '2': 0.0404707, '4': 0.2182434} | 974.7–4024 |
| 4 | robust_noise_weighted | fd_A_P2 | 6.065275 | 0.1311754 | -0.04983469 | [0.0069021, 0.0103049] | {'1': 0.0066693, '2': 0.0601029, '3': 0.1112823} | 2812–1.56e+04 |
| 4 | robust_noise_weighted | fd_B_P2 | 5.408901 | 0.2483242 | 0.2494363 | [0.0294791, 0.0321858] | {'1': 0.0053354, '2': 0.0349476, '4': 0.2180828} | 863.6–1.546e+04 |
| 4 | robust_noise_weighted | continuum_A_P2 | 6.151598 | 0.1392653 | 0.01505541 | [0.0083684, 0.0110264] | {'1': 0.0066917, '2': 0.0606832, '3': 0.1277422} | 2517–1.443e+04 |
| 4 | robust_noise_weighted | continuum_B_P2 | 5.665923 | 0.243875 | 0.01457767 | [0.0317668, 0.0277083] | {'1': 0.0053533, '2': 0.036542, '4': 0.2200573} | 859.1–1.448e+04 |

Both weighted robust winners retain [0,1.75,2], so they do not avoid late
times in this candidate set. Their change is spatial placement across the
ensemble. The twelve-scenario efficiency table identifies what they protect;
raw column norms and noise weights explain which strengths are sacrificed or
gained. A stronger maximin score alone does not establish robustness to
unmodeled discrepancy or the held-out field C.

At nominal FD field A, the robust weighted design reduces the raw q-column
norm from .22777 to .16759 (eight sensors) and .17936 to .13118 (four sensors):
it sacrifices this nominal diffusion-column strength. For FD field B at the
same parameters, q norms instead rise from .29716 to .34717 and .21735 to
.24832. Absolute within-model column cosine falls from .70588 to .16092 and
.76502 to .24944, respectively. These measured changes support protection
against field-B sensitivity collinearity, not merely maximizing field-A damping
response. Field B is not the only weakness: the historical four-sensor joint
design is limited by FD field A at P1 under the weighted criterion. The full
scenario table, rather than the nominal slice alone, determines the maximin
selection. Mode k=4 still dominates individual field-B diffusion-column norms;
the data do not support claiming a uniform spread of information over modes.

## Optimizer stationarity and numerical limitations

| Quantity | Count |
| --- | --- |
| clean_fits | 66 |
| noisy_fits | 1140 |
| clean_continued | 0 |
| noisy_continued | 2 |
| clean_unconverged | 0 |
| noisy_unconverged | 0 |

Fits requiring continuation:

| Group | Seed | Initial KKT | Final KKT | Initial nu4 | Final nu4 | Total updates |
| --- | --- | --- | --- | --- | --- | --- |
| 4/continuum_offnominal/baseline | 513 | 2.315606e-06 | 5.194955e-16 | 0.0001116904 | 0.0001 | 4000 |
| continuum_A/269 | 535 | 2.949062e-07 | 1.562098e-17 | 0.0001245345 | 0.0001 | 4000 |

Both continued fits finish at the lower diffusion bound. Their projected/KKT residuals account for the outward gradient at an active constraint; a small projected residual does not assert a zero unconstrained gradient or accurate physical recovery.

Every clean and noisy fit satisfies the original projected/KKT criterion within the fixed cap.

## Scientific caveats

- The finite model ensemble cannot represent all model-form uncertainty; robustness applies only to declared scenarios.
- Maximin design can be conservative and depends on coordinates and the fixed scaling.
- The heteroscedastic design variance is an assumption, not measured instrument behavior; recovery here uses the preserved Stage 7 noise convention.
- Local linearized sensitivity does not guarantee accurate nonlinear recovery or global loss minimization. The maximin objective compares sensitivity strength across models; it does not directly penalize their prediction discrepancy or the resulting effective-parameter bias.
- Continuum truth is still an idealized analytical model, not experimental data.
- Initial conditions remain exactly known during inversion.
- Sensor optimization is restricted to archived candidate subsets of the 32-point grid.
- Clean continuum effective coefficients reflect model/design mismatch, not different physical truth.
- The nine-design score/recovery diagnostic is small and stratified; no statistical significance or causal optimality is claimed.

## Tests, reproducibility and acceptance

Executed:

    .venv/bin/python -m pytest -q
    .venv/bin/python -m experiments.stage8_robust_design --design-only
    .venv/bin/python -m experiments.stage8_robust_design --recover-frozen

Pytest: **82 passed in 26.93s**. Tests do not run the full search or stochastic study.
All previous stage files were audited against pre-Stage-8 hashes. The full
results JSON rejects NaN/Infinity and preserves original 2000-step and hardened
final recovery values separately. Recovery never calls design selection.

- continuum_derivatives: PASS

- 12_scenarios: PASS

- exhaustive_candidates: PASS

- candidate_pool_preserved: PASS

- frozen_integrity: PASS

- all_primary_groups: PASS

- all_clean_references: PASS

- new_noise_seeds: PASS

- decomposition_identity: PASS

- uniform_continuation_policy: PASS

- within_update_cap: PASS

- clean_stationarity: PASS

- noisy_stationarity: PASS

- bounds: PASS

**PASS** for the implemented study. Measured recovery failures are disclosed rather than used to choose replacement designs.

## Interpretation and recommended Stage 9

Robust weighted design reduces the matched/continuum reversal and the worst
mean diffusion error across the four truth families: 51.08% to 17.53% for eight
sensors, and 66.65% to 22.03% for four, relative to the frozen Stage 7 joint
design. It also improves matched-model diffusion recovery in this ensemble.
This does not imply uniform superiority: joint E remains better for diffusion
on held-out field C at both budgets, and robust unweighted is better on moderate
field-A continuum truth. Under the severe budget, the weighted design's worst
mean velocity error is 4.28%, above joint E's 3.61%; evenly spaced sensors also
have a lower worst field error (9.46% versus 10.25%). The reported objective
protects sensitivity across a finite design ensemble, not every evaluation
metric or every unseen field.

The score/recovery association is strongly negative within the predeclared
nine-design diagnostic, but the subset is stratified and bound-limited fits
matter. No significance, causal or universal-optimality claim follows.

Recommend **Stage 9 Option C: an explicit model-error/discrepancy term**. Clean
design-specific continuum biases remain substantial even when noise and
optimizer error are small. A transparent discrepancy model should be tested
for separation of numerical-model error from the physical diffusion coefficient,
with fresh held-out validation and an identifiability check. This recommendation
follows the measured bias bottleneck; no discrepancy model or Stage 9 code is
implemented here.

Stage 9 is not implemented.

Git status:

    fatal: not a git repository (or any of the parent directories): .git

No commits or pushes were made.

