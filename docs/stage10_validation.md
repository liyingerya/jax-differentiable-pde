# Stage 10: Practical Identifiability, Profile-Based Uncertainty, and Uncertainty Calibration

**PASS**. Profiles, local curvature, and repeated-noise distributions address different aspects of inference. All numerical-model and design definitions from Stages 1–9 are preserved. No Stage 11 implementation or commit/push is performed.

## Scope and files

Stage 9 found strong q–c6 compensation despite stationary optimizers and accurate predicted fields. Stage 10 compares fixed theoretical discrepancy M1, free discrepancy M4, and fixed calibrated M5. New files are `src/identifiability.py`, `experiments/stage10_identifiability.py`, `experiments/stage10_report.py`, `tests/test_identifiability.py`, and Stage 10 documents/results. README is updated. Existing source, tests, experiments, and Stage 1–9 results are hash-checked unchanged. Coordinates are q=log(ν4), v∈[.5,1.5], ν4∈[1e−4,.02], and main c3,c6∈[0,2]. M1/M5 infer only [v,q]; M4 infers [v,q,c3,c6]. M1 fixes c=(1,1), while M5 fixes the exact Stage 9 calibrated values shown in the pseudo-truth table.

## Observation model and meaning of clean geometry

At positive observation times, y=clean+ε with independent ε∼N(0,σ_abs²), where σ_abs=σ_rel×RMS(clean selected positive-time observations). NLL=SSE_positive/(2σ_abs²), omitting the additive constant. The exactly known initial field is deterministic: t=0 contributes no stochastic NLL. Normalized MSE over all observed entries is saved separately.

Truly zero-noise data have σ=0 and do not define finite Gaussian likelihood intervals. Full-clean geometry therefore uses a clearly declared hypothetical 2%-RMS measurement scale without adding noise. Its intervals describe sensitivity at that assumed precision, not uncertainty induced by a zero-variance experiment. Moderate and severe clean pseudo-truth references use their corresponding 2% and 5% scales; noisy analyses use the actual known scales. M5 calibration uncertainty is not propagated.

Full observations use all 32 points and the nine times 0,.25,…,2. Main inference retains the Stage 9 fixed 1024-step RK4 graph. The sparse masks are loaded from the stored Stage 8 design, with physical times remapped exactly to this graph:

| regime | sensors | times | relative noise |
| --- | --- | --- | --- |
| moderate | [9, 11, 13, 19, 22, 27, 29, 31] | [0, 1.75, 2] | 0.02 |
| severe | [10, 21, 28, 31] | [0, 1.75, 2] | 0.05 |

## Optimizer units and retained scaling pilot

Adam uses the unchanged Stage 9 learning rate .05, β1=.9, β2=.999, ε=1e−8, and 2000 updates followed only when needed by continuation of the same state to 4000. The original KKT threshold 1e−7 is defined in the Stage 9 NMSE scale. We minimize NLL multiplied by the fixed data-dependent scalar 2σ²/[number_of_observed_entries×(mean(y²)+1e−12)]. This is exactly the original NMSE gradient scale: deterministic t=0 residuals are parameter-independent. The minimizer is unchanged by this positive scalar. Raw NLL, raw NLL KKT, scaled KKT, and the scalar are all saved. Likelihood thresholds and Hessians are always in raw NLL units; NMSE is never compared with an LR cutoff.

An initial clean-only raw-NLL optimization pilot applied the numerical threshold to different gradient units and exposed this mismatch. It is preserved in `docs/stage10_scaling_pilot.json`, including failed endpoints. The scaling correction preceded bootstrap production and changed no optimizer hyperparameters. Failed pilot cases: continuum_A/full/M4, continuum_A/4/M4, matched_A/4/M4.

## Protocol and nuisance optimization

Every profile point fixes its target coordinate exactly and optimizes all nuisance coordinates with three predeclared starts: central optimum, alternate [.8,log(.004),.2,1.8] restricted to model dimension, and the preceding point’s best preliminary endpoint (a batched warm-start pass). Threshold refinements use the nearest existing endpoint as the warm start. Every attempt is retained, and only the lowest converged NLL is selected. No selection uses physical accuracy. The unrestricted fits likewise use three declared starts. Fixed coordinates have zero optimizer derivatives and equal projection bounds.

Primary q profiles begin with at least 61 points; full-clean starts with ν4∈[.0005,.005] and expands if needed, while sparse profiles cover the full inverse bounds [1e−4,.02]. c6 has at least 61 points over its full bounds; c3 and v have 41 initial points. The central and pseudo-true coordinates are explicitly included. Eight bisection rounds refine crossings. Invalid points are retained and never bridged. Disconnected accepted components are not merged. Neighboring nuisance jumps above 15% of coordinate range are flagged; all points already have alternate starts. No artificial curve smoothing is applied.

Nominal scalar LR cutoffs are 2ΔNLL≤1 (68%) and ≤3.841459 (95%). These are asymptotic references, not exact confidence guarantees. A converged profile improving the unrestricted baseline by >1e−5 NLL triggers the predeclared unrestricted reconciliation attempt; original fits remain recorded. An unresolved baseline invalidates interval use.

## Clean pseudo-true parameters

Continuum physical truth is [v,ν4]=[1,.002]; each model/design’s clean stationary optimum is its pseudo-truth. The latter, rather than physical truth, centers sampling-deviation and curvature calculations. Full-clean reference precision is hypothetical as stated above.

| truth/regime/model | v | ν4 | c3 | c6 | NLL | NMSE | scaled KKT | raw NLL KKT | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/full/M1 | 1.00092 | 0.002006258 | 1 | 1 | 1.480455 | 4.098049e-06 | 7.405338e-16 | 2.675242e-10 | 2000 |
| continuum_A/full/M4 | 0.999318 | 0.001996882 | 1.076854 | 1.092513 | 0.1221835 | 3.382162e-07 | 1.431473e-12 | 5.171318e-07 | 2000 |
| continuum_A/full/M5 | 0.9997471 | 0.001999859 | 1.056254 | 1.061871 | 0.219559 | 6.077613e-07 | 8.45813e-16 | 3.055572e-10 | 2000 |
| continuum_A/8/M1 | 1.000376 | 0.002009357 | 1 | 1 | 0.5081232 | 1.259927e-05 | 1.217758e-13 | 4.911167e-09 | 2000 |
| continuum_A/8/M4 | 0.9992457 | 0.001985941 | 1.077398 | 1.234922 | 0.02066262 | 5.12344e-07 | 1.922218e-12 | 7.752228e-08 | 2000 |
| continuum_A/8/M5 | 0.9995511 | 0.002002532 | 1.056254 | 1.061871 | 0.05710629 | 1.41599e-06 | 2.168219e-09 | 8.74434e-05 | 2000 |
| continuum_A/4/M1 | 1.000542 | 0.002003529 | 1 | 1 | 0.07252467 | 1.311941e-05 | 6.563724e-14 | 3.628454e-10 | 2000 |
| continuum_A/4/M4 | 0.9993423 | 0.002026846 | 1.071012 | 0.7614993 | 0.001342134 | 2.427865e-07 | 3.930298e-09 | 2.172685e-05 | 2000 |
| continuum_A/4/M5 | 0.9996019 | 0.001997085 | 1.056254 | 1.061871 | 0.005419229 | 9.803161e-07 | 1.505721e-09 | 8.323688e-06 | 2000 |
| continuum_C/full/M1 | 1.003494 | 0.00201941 | 1 | 1 | 17.45345 | 4.776982e-05 | 2.370842e-15 | 8.66224e-10 | 2000 |
| continuum_C/full/M4 | 0.9983194 | 0.001977022 | 1.124569 | 1.268167 | 3.288637 | 9.000951e-06 | 2.808457e-15 | 1.026113e-09 | 2000 |
| continuum_C/full/M5 | 1.001154 | 0.002008796 | 1.056254 | 1.061871 | 7.588009 | 2.076827e-05 | 3.877064e-16 | 1.416545e-10 | 2000 |
| continuum_C/8/M1 | 1.00255 | 0.002006803 | 1 | 1 | 3.185145 | 8.24738e-05 | 4.191828e-12 | 1.618887e-07 | 2000 |
| continuum_C/8/M4 | 0.9983144 | 0.001910026 | 1.116143 | 1.76611 | 0.5250324 | 1.35948e-05 | 8.807663e-13 | 3.401527e-08 | 2000 |
| continuum_C/8/M5 | 1.000577 | 0.00200044 | 1.056254 | 1.061871 | 1.193347 | 3.089964e-05 | 3.470307e-14 | 1.340235e-09 | 2000 |
| continuum_C/4/M1 | 1.003115 | 0.001975796 | 1 | 1 | 0.1361266 | 4.611341e-05 | 3.783199e-14 | 1.116798e-10 | 2000 |
| continuum_C/4/M4 | 0.9983312 | 0.001877551 | 1.10858 | 2 | 0.02390191 | 8.096867e-06 | 5.457169e-14 | 1.610954e-10 | 2000 |
| continuum_C/4/M5 | 1.000845 | 0.001985047 | 1.056254 | 1.061871 | 0.04692883 | 1.589733e-05 | 5.043285e-14 | 1.488775e-10 | 2000 |
| matched_A/full/M1 | 0.9790755 | 0.001977337 | 1 | 1 | 232.8825 | 0.0006447161 | 2.534155e-15 | 9.153802e-10 | 2000 |
| matched_A/full/M4 | 1 | 0.002 | 0 | 0 | 1.019828e-15 | 2.823329e-21 | 1.641495e-10 | 5.92936e-05 | 4000 |
| matched_A/full/M5 | 0.9779172 | 0.001979699 | 1.056254 | 1.061871 | 259.0809 | 0.0007172442 | 7.642279e-12 | 2.760522e-06 | 2000 |
| matched_A/8/M1 | 0.9853049 | 0.002025861 | 1 | 1 | 86.30861 | 0.002096733 | 3.719311e-14 | 1.530994e-09 | 2000 |
| matched_A/8/M4 | 1 | 0.002 | 2.741385e-13 | 0 | 9.279159e-23 | 2.254572e-27 | 6.205995e-15 | 2.554597e-10 | 2000 |
| matched_A/8/M5 | 0.9844663 | 0.00203713 | 1.056254 | 1.061871 | 95.98716 | 0.002331859 | 1.066874e-09 | 4.391612e-05 | 2000 |
| matched_A/4/M1 | 0.9832236 | 0.002058063 | 1 | 1 | 14.08195 | 0.002539812 | 1.472477e-08 | 8.164131e-05 | 2000 |
| matched_A/4/M4 | 0.9999996 | 0.001999496 | 3.741034e-05 | 0.005105818 | 4.089396e-08 | 7.375608e-12 | 4.049098e-09 | 2.245017e-05 | 4000 |
| matched_A/4/M5 | 0.9823033 | 0.00206485 | 1.056254 | 1.061871 | 15.62591 | 0.002818278 | 1.211476e-14 | 6.71701e-11 | 2000 |

## Primary nominal profiles

For q, endpoints below are transformed to ν4; q endpoints and all components remain in JSON. Other coordinates are reported directly. A bound-limited interval is constrained by the declared model, not a physical limit.

| case | target | 68% interval | 95% interval | invalid points | nuisance jumps |
| --- | --- | --- | --- | --- | --- |
| continuum_A/full/M1/0 | v | [1.000172, 1.001667] | [0.9994553, 1.002384] | 0 | 0 |
| continuum_A/full/M1/1 | q → ν4 | [0.001942461, 0.002070483] | [0.001881617, 0.002132544] | 0 | 0 |
| continuum_A/full/M4/0 | v | [0.9980922, 1.000543] | [0.9969151, 1.001721] | 0 | 6 |
| continuum_A/full/M4/1 | q → ν4 | [0.001856287, 0.002167199] | [0.001796666, 0.002235208] | 0 | 3 |
| continuum_A/full/M4/2 | c3 | [1.030184, 1.123615] | [0.9854659, 1.16859] | 0 | 0 |
| continuum_A/full/M4/3 | c6 | [0, 2] (bound limited) | [0, 2] (bound limited) | 0 | 0 |
| continuum_A/full/M5/0 | v | [0.9990018, 1.000493] | [0.9982841, 1.00121] | 0 | 0 |
| continuum_A/full/M5/1 | q → ν4 | [0.001936248, 0.002063897] | [0.001875582, 0.002125779] | 0 | 0 |
| continuum_A/8/M1/0 | v | [0.9989963, 1.001755] | [0.9976692, 1.003079] | 0 | 2 |
| continuum_A/8/M1/1 | q → ν4 | [0.001880316, 0.002140596] | [0.001758451, 0.002268711] | 0 | 0 |
| continuum_A/8/M4/0 | v | [0.9974384, 1.00105] | [0.9957009, 1.00278] | 0 | 10 |
| continuum_A/8/M4/1 | q → ν4 | [0.00179343, 0.002244186] | [0.001676523, 0.002380432] | 0 | 2 |
| continuum_A/8/M4/2 | c3 | [0.9990104, 1.156042] | [0.9239754, 1.231806] | 0 | 0 |
| continuum_A/8/M4/3 | c6 | [0, 2] (bound limited) | [0, 2] (bound limited) | 0 | 0 |
| continuum_A/8/M5/0 | v | [0.9981732, 1.000928] | [0.9968481, 1.00225] | 0 | 2 |
| continuum_A/8/M5/1 | q → ν4 | [0.001873866, 0.002133392] | [0.001752356, 0.002261138] | 0 | 0 |
| continuum_A/4/M1/0 | v | [0.9975797, 1.003503] | [0.9947332, 1.006342] | 0 | 1 |
| continuum_A/4/M1/1 | q → ν4 | [0.001745457, 0.002270897] | [0.001505934, 0.00253688] | 0 | 0 |
| continuum_A/4/M4/0 | v | [0.9948465, 1.003799] | [0.9906104, 1.008005] | 0 | 8 |
| continuum_A/4/M4/1 | q → ν4 | [0.001664058, 0.002386433] | [0.001435649, 0.002665375] | 0 | 2 |
| continuum_A/4/M4/2 | c3 | [0.8723577, 1.275429] | [0.6898935, 1.465883] | 0 | 2 |
| continuum_A/4/M4/3 | c6 | [0, 2] (bound limited) | [0, 2] (bound limited) | 0 | 0 |
| continuum_A/4/M5/0 | v | [0.9966426, 1.002559] | [0.9937992, 1.005396] | 0 | 2 |
| continuum_A/4/M5/1 | q → ν4 | [0.00174009, 0.002263336] | [0.001501565, 0.002528207] | 0 | 0 |

The same primary diffusion intervals in the internal q coordinate:

| case | 68% q interval | 95% q interval |
| --- | --- | --- |
| continuum_A/full/M1/1 | [-6.2438, -6.179973] | [-6.275624, -6.150439] |
| continuum_A/full/M4/1 | [-6.289177, -6.13432] | [-6.321822, -6.103421] |
| continuum_A/full/M5/1 | [-6.247003, -6.183159] | [-6.278836, -6.153617] |
| continuum_A/8/M1/1 | [-6.276315, -6.146671] | [-6.343322, -6.088544] |
| continuum_A/8/M4/1 | [-6.323625, -6.099412] | [-6.391033, -6.040473] |
| continuum_A/8/M5/1 | [-6.279752, -6.150042] | [-6.346794, -6.091887] |
| continuum_A/4/M1/1 | [-6.350739, -6.08758] | [-6.498342, -5.97682] |
| continuum_A/4/M4/1 | [-6.398496, -6.037956] | [-6.546138, -5.927411] |
| continuum_A/4/M5/1 | [-6.353819, -6.090915] | [-6.501248, -5.980245] |

Representative noisy coordinate intervals use seed 800, the first predeclared A seed in each regime. Its q intervals are also included in the coverage study:

| case | 68% interval | 95% interval | invalid points |
| --- | --- | --- | --- |
| continuum_A/8/M1/800/0 | [0.9981164, 1.000879] | [0.9967881, 1.002203] | 0 |
| continuum_A/8/M4/800/0 | [0.9977584, 1.001378] | [0.9960169, 1.003113] | 0 |
| continuum_A/8/M4/800/2 | [0.9168289, 1.078255] | [0.8396418, 1.156085] | 0 |
| continuum_A/8/M4/800/3 | [1.126805, 2] (bound limited, one sided) | [0, 2] (bound limited) | 0 |
| continuum_A/8/M5/800/0 | [0.9973019, 1.00006] | [0.9959761, 1.001383] | 0 |
| continuum_A/4/M1/800/0 | [0.9916204, 0.9976256] | [0.9887339, 1.000504] | 0 |
| continuum_A/4/M4/800/0 | [0.9843029, 0.9933396] | [0.979947, 0.9976605] | 0 |
| continuum_A/4/M4/800/2 | [1.154544, 1.581072] | [0.9540437, 1.790275] | 0 |
| continuum_A/4/M4/800/3 | [0, 2] (bound limited) | [0, 2] (bound limited) | 0 |
| continuum_A/4/M5/800/0 | [0.9907272, 0.9967261] | [0.9878439, 0.9996018] | 0 |

## Profiled q–c6 surface and compensation valley

The 31×31 surface fixes q,c6 and reoptimizes v,c3 at every point with three starts. It differs from Stage 9’s conditional slices. The sampled minimum is q=-6.216168, ν4=0.001996882, c6=1.066667, NLL=0.1229636. Invalid points: 0. Full nuisance optima and diagnostics are stored. This is a local finite grid; full 1D profiles determine interval extent.

| q | ν4 | best sampled c6 | NLL | 2ΔNLL |
| --- | --- | --- | --- | --- |
| -6.366168 | 0.001718733 | 2 | 5.392707 | 10.54105 |
| -6.356168 | 0.001736006 | 2 | 4.502398 | 8.760429 |
| -6.346168 | 0.001753453 | 2 | 3.689025 | 7.133683 |
| -6.336168 | 0.001771076 | 2 | 2.95459 | 5.664814 |
| -6.326168 | 0.001788875 | 2 | 2.30113 | 4.357892 |
| -6.316168 | 0.001806854 | 2 | 1.73071 | 3.217054 |
| -6.306168 | 0.001825013 | 2 | 1.245433 | 2.2465 |
| -6.296168 | 0.001843355 | 2 | 0.8474331 | 1.450499 |
| -6.286168 | 0.001861881 | 2 | 0.538877 | 0.8333871 |
| -6.276168 | 0.001880593 | 2 | 0.321967 | 0.399567 |
| -6.266168 | 0.001899493 | 2 | 0.1989389 | 0.1535108 |
| -6.256168 | 0.001918584 | 1.933333 | 0.1634655 | 0.08256405 |
| -6.246168 | 0.001937866 | 1.733333 | 0.1464131 | 0.04845916 |
| -6.236168 | 0.001957341 | 1.466667 | 0.1334441 | 0.02252126 |
| -6.226168 | 0.001977013 | 1.266667 | 0.125584 | 0.006801019 |
| -6.216168 | 0.001996882 | 1.066667 | 0.1229636 | 0.001560276 |
| -6.206168 | 0.002016951 | 0.8666667 | 0.1257354 | 0.007103756 |
| -6.196168 | 0.002037222 | 0.6666667 | 0.134084 | 0.02380107 |
| -6.186168 | 0.002057697 | 0.5333333 | 0.147726 | 0.05108504 |
| -6.176168 | 0.002078377 | 0.3333333 | 0.1666583 | 0.08894969 |
| -6.166168 | 0.002099265 | 0.1333333 | 0.1915533 | 0.1387396 |
| -6.156168 | 0.002120363 | 0 | 0.2289816 | 0.2135963 |
| -6.146168 | 0.002141673 | 0 | 0.3481506 | 0.4519342 |
| -6.136168 | 0.002163197 | 0 | 0.569794 | 0.895221 |
| -6.126168 | 0.002184937 | 0 | 0.8963335 | 1.5483 |
| -6.116168 | 0.002206896 | 0 | 1.330227 | 2.416087 |
| -6.106168 | 0.002229076 | 0 | 1.87397 | 3.503573 |
| -6.096168 | 0.002251479 | 0 | 2.530094 | 4.815822 |
| -6.086168 | 0.002274106 | 0 | 3.30117 | 6.357973 |
| -6.076168 | 0.002296962 | 0 | 4.189804 | 8.135241 |
| -6.066168 | 0.002320046 | 0 | 5.198642 | 10.15292 |

| nominal level | sampled valley ν4 range |
| --- | --- |
| 68 | [0.001861881, 0.002163197] |
| 95 | [0.001806854, 0.002229076] |

The sampled compensation curve has overall dc6/dq≈-9.327957 (including clipping at discrepancy bounds). Increased physical damping can be offset by reduced numerical damping correction. The full curve, rather than a straight-line extrapolation, defines the reported geometry.

Within the unclipped portion of the sampled valley, dc6/dq≈-19.79798. This local compensation direction differs from the slope averaged over bound plateaus.

## Local inverse curvature versus nonlinear profiles

Hessians use raw NLL at each clean pseudo-truth. Their inverses, when numerically positive definite, are local inverse-curvature approximations, not calibrated covariances. Intervals below use q±sqrt(LR cutoff)×local q scale, transformed to ν4, with no artificial clipping to hide bound violations. Positive definiteness does not make an active-bound Gaussian approximation valid.

| case | H(q) condition | local q scale | local 95% via q | local 95% via physical ν4 | nonlinear 95% ν4 profile |
| --- | --- | --- | --- | --- | --- |
| continuum_A/full/M1 | 1821.57 | 0.0319054 | [0.001884642, 0.002135721] | [0.001880799, 0.002131716] | [0.001881617, 0.002132544] |
| continuum_A/full/M4 | 1.437265e+07 | 0.1378034 | [0.001524242, 0.00261608] | [0.001457545, 0.00253622] | [0.001796666, 0.002235208] |
| continuum_A/full/M5 | 1826.954 | 0.0319143 | [0.001878598, 0.002128947] | [0.001874766, 0.002124952] | [0.001875582, 0.002125779] |
| continuum_A/8/M1 | 2258.547 | 0.06476163 | [0.001769831, 0.0022813] | [0.001754308, 0.002264406] | [0.001758451, 0.002268711] |
| continuum_A/8/M4 | 1.690211e+07 | 0.2752701 | [0.00115786, 0.003406252] | [0.0009144873, 0.003057396] | [0.001676523, 0.002380432] |
| continuum_A/8/M5 | 2266.978 | 0.06479408 | [0.001763708, 0.002273695] | [0.001748222, 0.002256842] | [0.001752356, 0.002261138] |
| continuum_A/4/M1 | 1959.42 | 0.1310795 | [0.001549604, 0.002590423] | [0.001488801, 0.002518258] | [0.001505934, 0.00253688] |
| continuum_A/4/M4 | 3.205398e+07 | 0.8405046 | [0.000390285, 0.01052591] | [-0.001312096, 0.005365788] | [0.001435649, 0.002665375] |
| continuum_A/4/M5 | 1959.737 | 0.1309534 | [0.001545001, 0.002581454] | [0.001484505, 0.002509665] | [0.001501565, 0.002528207] |

A negative endpoint in the unconstrained physical-coordinate quadratic approximation is deliberately displayed as a sign of local-approximation failure; it is not an admissible physical interval. Nonlinear profiles retain the original positive diffusion bounds.

For full-clean M4, raw-NLL Hessian eigenvalues are [0.1249646, 458.0321, 983.8922, 1796073]; the smallest-eigenvalue direction in [v,q,c3,c6] is [4.464331e-08, -0.0473954, -1.580193e-05, 0.9988762]. The full matrix and eigensystem for every clean case are retained in JSON.

## Coordinate sensitivity

The physical-coordinate Hessian is computed by direct autodiff and independently checked against H_physical=JᵀH_logJ plus the gradient-weighted second-derivative term −g_q/ν4² in the ν4 diagonal. That term is retained even at numerically stationary or constrained endpoints. A pure Jacobian sandwich alone need not be exact at an active bound.

| case | condition [v,q,…] | condition [v,ν4,…] | q-coordinate diagonals | physical-coordinate diagonals | transform relative difference |
| --- | --- | --- | --- | --- | --- |
| continuum_A/full/M1 | 1821.57 | 136.3951 | [1789442, 982.3823] | [1789442, 2.440659e+08] | 1.221077e-16 |
| continuum_A/full/M4 | 1.437265e+07 | 1.96563e+09 | [1795294, 981.6808, 1236.914, 2.334823] | [1795294, 2.461871e+08, 1236.914, 2.334823] | 1.210556e-16 |
| continuum_A/full/M5 | 1826.954 | 136.8597 | [1793729, 981.8166] | [1793729, 2.454888e+08] | 2.427999e-16 |
| continuum_A/8/M1 | 2258.547 | 115.4424 | [538497.5, 244.5772] | [538497.5, 6.057616e+07] | 1.229953e-16 |
| continuum_A/8/M4 | 1.690211e+07 | 1.930044e+09 | [540622.3, 244.0364, 280.5983, 0.5864136] | [540622.3, 6.187592e+07, 280.5983, 0.5864136] | 1.204116e-16 |
| continuum_A/8/M5 | 2266.978 | 115.7058 | [539966.7, 244.2357] | [539966.7, 6.09046e+07] | 3.669959e-16 |
| continuum_A/4/M1 | 1959.42 | 127.1953 | [114040.3, 58.21367] | [114040.3, 1.450219e+07] | 3.853167e-16 |
| continuum_A/4/M4 | 3.205398e+07 | 3.976284e+09 | [114291.6, 58.40381, 58.65286, 0.1492695] | [114291.6, 1.421673e+07, 58.65286, 0.1492695] | 3.930535e-16 |
| continuum_A/4/M5 | 1959.737 | 127.9939 | [114278.3, 58.32509] | [114278.3, 1.462387e+07] | 2.547404e-16 |
| continuum_C/full/M1 | 275.4493 | 890.5438 | [2464518, 8948.772] | [2464518, 2.194393e+09] | 2.172979e-16 |
| continuum_C/full/M4 | 503380 | 4.554045e+08 | [2492442, 8877.819, 6045.54, 54.42729] | [2492442, 2.271347e+09, 6045.54, 54.42729] | 1.140399e-18 |
| continuum_C/full/M5 | 277.4514 | 893.2986 | [2476910, 8927.935] | [2476910, 2.212479e+09] | 1.52591e-18 |
| continuum_C/8/M1 | 272.0698 | 952.1179 | [510278.1, 1915.761] | [510278.1, 4.756986e+08] | 3.758976e-16 |
| continuum_C/8/M4 | 1371086 | 1.394574e+09 | [514438.1, 1919.603, 1074.085, 8.712821] | [514438.1, 5.26178e+08, 1074.085, 8.712821] | 2.26557e-16 |
| continuum_C/8/M5 | 271.0824 | 962.5398 | [511567.2, 1928.467] | [511567.2, 4.819045e+08] | 3.865175e-18 |
| continuum_C/4/M1 | 680.2779 | 570.0263 | [58863.21, 106.4539] | [58863.21, 2.726953e+07] | 1.280719e-17 |
| continuum_C/4/M4 | 2697040 | 1.357424e+09 | [59925.19, 106.8654, 113.4254, 0.5761072] | [59925.19, 3.031473e+07, 113.4254, 0.5761072] | 1.228871e-16 |
| continuum_C/4/M5 | 670.8472 | 563.8038 | [59301.3, 107.9096] | [59301.3, 2.738536e+07] | 1.360322e-16 |
| matched_A/full/M1 | 1870.556 | 137.5536 | [1790449, 960.0292] | [1790449, 2.455405e+08] | 3.792949e-18 |
| matched_A/full/M4 | 1.228802e+07 | 1.60883e+09 | [1723086, 905.0539, 1260.153, 2.403266] | [1723086, 2.262635e+08, 1260.153, 2.403266] | 2.634303e-16 |
| matched_A/full/M5 | 1862.399 | 137.9223 | [1793862, 966.4052] | [1793862, 2.465818e+08] | 3.625854e-16 |
| matched_A/8/M1 | 2094.301 | 116.8522 | [547823.8, 262.1451] | [547823.8, 6.387376e+07] | 1.166454e-16 |
| matched_A/8/M4 | 1.364342e+07 | 1.553369e+09 | [529801.7, 241.9434, 291.3948, 0.6563425] | [529801.7, 6.048586e+07, 291.3948, 0.6563425] | 3.695366e-16 |
| matched_A/8/M5 | 2072.827 | 116.7351 | [548884.6, 265.3454] | [548884.6, 6.39402e+07] | 7.282762e-18 |
| matched_A/4/M1 | 1824.304 | 130.4442 | [113840.1, 62.64773] | [113840.1, 1.479068e+07] | 2.518675e-16 |
| matched_A/4/M4 | 3.56528e+07 | 4.15976e+09 | [111427.2, 52.11798, 64.57421, 0.1338232] | [111427.2, 1.303607e+07, 64.57421, 0.1338232] | 9.488386e-18 |
| matched_A/4/M5 | 1804.541 | 131.0147 | [113985, 63.41624] | [113985, 1.487386e+07] | 2.504589e-16 |

A monotone change from q to ν4 keeps the same fixed physical predictions and nuisance problems. Interpolating the same refined LR samples on the physical axis instead of transforming q crossings produces only finite-grid interpolation differences:

| profile | 68% max relative endpoint difference | 95% max relative endpoint difference | direct NLL max difference |
| --- | --- | --- | --- |
| continuum_A/full/M1/1 | 2.440953e-09 | 2.649067e-09 | 0 |
| continuum_A/full/M4/1 | 1.974295e-09 | 2.769876e-09 | 0 |
| continuum_A/full/M5/1 | 2.789692e-09 | 2.809014e-09 | 0 |
| continuum_A/8/M1/1 | 1.293844e-08 | 1.446838e-08 | 0 |
| continuum_A/8/M4/1 | 1.285806e-08 | 1.471846e-08 | 0 |
| continuum_A/8/M5/1 | 4.999005e-09 | 1.487311e-08 | 0 |
| continuum_A/4/M1/1 | 4.420871e-09 | 8.593446e-09 | 0 |
| continuum_A/4/M4/1 | 1.385163e-08 | 1.412253e-08 | 0 |
| continuum_A/4/M5/1 | 1.290171e-08 | 1.112015e-08 | 0 |

## Diagnostic discrepancy-bound sensitivity

These diagnostics use clean continuum A under the moderate observation design, c3,c6∈[0,B] for B=2,3,4, and a common 2048-step graph. Main production bounds and the Stage 9 graph remain unchanged. The larger box is separately checked for RK4 stability on a dense angular grid and sampled parameter combinations. Both discrepancy bounds widen together; this is not a different basis.

| B | pseudo [v,ν4,c3,c6] | NLL | KKT | 95% ν4 interval | 95% c6 interval | sampled max amplification |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | [0.9992457, 0.001985941, 1.077398, 1.234922] | 0.02066262 | 1.920572e-12 | [0.001676523, 0.002380432] | [0, 2] (bound limited) | 1 |
| 3 | [0.9992457, 0.001985941, 1.077398, 1.234922] | 0.02066262 | 1.920572e-12 | [0.001603785, 0.002380432] | [0, 3] (bound limited) | 1 |
| 4 | [0.9992457, 0.001985941, 1.077398, 1.234922] | 0.02066262 | 1.920572e-12 | [0.001538134, 0.002380432] | [0, 4] (bound limited) | 1 |

## Paired ensembles and declared coverage subsets

Continuum A uses seeds 800–899 (100 per model, moderate) and 800–849 (50 per model, severe). Held-out C uses 900–949 (50 per model, moderate). Matched original-FD controls use 800–819 (20 per model/regime) for M1 and M4. The same seed on a different truth/design is a different conditional dataset. Within each truth/design/seed, observations are generated once, saved and hashed, and reused exactly across models. M5 is never recalibrated.

Full q-profile coverage uses the first 20 moderate A seeds, first 10 severe A seeds, and first 10 held-out C seeds, for every M1/M4/M5 model. Subsets were declared before recovery. Sample SD uses ddof=1; percentiles use NumPy linear interpolation. Coverage changes in 5% increments for 20 profiles and 10% increments for 10 profiles, so interpolated LR quantiles need not imply an exactly matching empirical coverage fraction. Optional off-nominal ensembles and prediction envelopes were skipped. These are repeated synthetic truth-plus-noise ensembles, conditional on the stated model; no Gaussian shape is imposed on parameter estimates.

## Primary continuum A empirical distributions

v:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M1 | 100 | 1.000306 | 1.000133 | 0.001366324 | 0.9980037 | 0.9994903 | 1.001153 | 1.002661 |
| continuum_A/8/M4 | 100 | 0.9991467 | 0.9991741 | 0.001846168 | 0.9960936 | 0.9980581 | 1.000399 | 1.002189 |
| continuum_A/8/M5 | 100 | 0.9994796 | 0.9993028 | 0.001362033 | 0.9971916 | 0.9986735 | 1.000323 | 1.001812 |
| continuum_A/4/M1 | 50 | 0.9995117 | 0.9993038 | 0.003537654 | 0.9946976 | 0.9964617 | 1.002656 | 1.004771 |
| continuum_A/4/M4 | 50 | 0.9973735 | 0.9970221 | 0.004736456 | 0.9905243 | 0.9936611 | 1.000698 | 1.005769 |
| continuum_A/4/M5 | 50 | 0.9985715 | 0.9983445 | 0.003531067 | 0.9937753 | 0.995529 | 1.001691 | 1.003827 |

ν4:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M1 | 100 | 0.001996349 | 0.002011556 | 0.0001456787 | 0.001804606 | 0.001901308 | 0.002076586 | 0.002214324 |
| continuum_A/8/M4 | 100 | 0.002002881 | 0.001994201 | 0.0001782679 | 0.001735754 | 0.001877748 | 0.002129854 | 0.002290432 |
| continuum_A/8/M5 | 100 | 0.001989635 | 0.002004328 | 0.0001451479 | 0.001798326 | 0.001894734 | 0.002070092 | 0.00220636 |
| continuum_A/4/M1 | 50 | 0.00197387 | 0.001960071 | 0.0002438514 | 0.001597565 | 0.001806832 | 0.002136461 | 0.002362788 |
| continuum_A/4/M4 | 50 | 0.001968587 | 0.001937377 | 0.0002560779 | 0.001552857 | 0.001823163 | 0.002150647 | 0.002419485 |
| continuum_A/4/M5 | 50 | 0.001967163 | 0.001953526 | 0.0002426326 | 0.001592555 | 0.001800175 | 0.002128905 | 0.002353878 |

c3:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M4 | 100 | 1.078468 | 1.073913 | 0.08084471 | 0.9683806 | 1.023158 | 1.121013 | 1.228356 |
| continuum_A/4/M4 | 50 | 1.128543 | 1.147284 | 0.1490815 | 0.9028545 | 1.013946 | 1.224805 | 1.357423 |

c6:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M4 | 100 | 0.97358 | 0.712778 | 0.9553317 | 0 | 0 | 2 | 2 |
| continuum_A/4/M4 | 50 | 1.078518 | 1.77423 | 0.9787242 | 0 | 0 | 2 | 2 |

| case | discrepancy-bound fraction | continuation fraction (selected fit) | unconverged fraction | mean final-field L2 | SD final-field L2 | mean clean observed NMSE |
| --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M1 | 0 | 0.28 | 0 | 0.006207495 | 0.002335888 | 3.980954e-05 |
| continuum_A/8/M4 | 0.86 | 0.21 | 0 | 0.006133933 | 0.002636054 | 4.118276e-05 |
| continuum_A/8/M5 | 0 | 0.21 | 0 | 0.005135913 | 0.002616369 | 2.859426e-05 |
| continuum_A/4/M1 | 0 | 0.14 | 0 | 0.01179288 | 0.004652727 | 0.0002282574 |
| continuum_A/4/M4 | 0.92 | 0.08 | 0 | 0.01271295 | 0.004366247 | 0.000273503 |
| continuum_A/4/M5 | 0 | 0.2 | 0 | 0.01120016 | 0.00476023 | 0.0002158771 |

## Matched original-FD controls empirical distributions

v:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A/8/M1 | 20 | 0.9851009 | 0.9848368 | 0.001279228 | 0.9831545 | 0.9845559 | 0.9858737 | 0.9870962 |
| matched_A/8/M4 | 20 | 0.9993596 | 0.9993874 | 0.00167947 | 0.9966954 | 0.9985897 | 1.000264 | 1.001873 |
| matched_A/4/M1 | 20 | 0.9822816 | 0.982292 | 0.003249477 | 0.9775309 | 0.9800194 | 0.9839495 | 0.9869115 |
| matched_A/4/M4 | 20 | 0.9971812 | 0.9965936 | 0.004072989 | 0.9912199 | 0.9944779 | 1.000074 | 1.00387 |

ν4:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A/8/M1 | 20 | 0.002011513 | 0.001975105 | 0.0001279209 | 0.001841401 | 0.001916678 | 0.002104465 | 0.002222651 |
| matched_A/8/M4 | 20 | 0.001895518 | 0.001901978 | 0.0001467736 | 0.001692819 | 0.001792531 | 0.001986529 | 0.002126025 |
| matched_A/4/M1 | 20 | 0.002085356 | 0.002022278 | 0.0002187993 | 0.001844952 | 0.001902125 | 0.002211677 | 0.002452582 |
| matched_A/4/M4 | 20 | 0.00194252 | 0.001937202 | 0.0001897281 | 0.001643725 | 0.00184282 | 0.002041331 | 0.002189603 |

c3:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A/8/M4 | 20 | 0.03510987 | 0.001552927 | 0.05905974 | 0 | 0 | 0.0339106 | 0.1718749 |
| matched_A/4/M4 | 20 | 0.1107122 | 0.08686528 | 0.1280895 | 0 | 0 | 0.1899122 | 0.3756064 |

c6:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A/8/M4 | 20 | 0.9287103 | 0.6007151 | 0.9754322 | 0 | 0 | 2 | 2 |
| matched_A/4/M4 | 20 | 1.051223 | 1.512233 | 0.9986339 | 0 | 0 | 2 | 2 |

| case | discrepancy-bound fraction | continuation fraction (selected fit) | unconverged fraction | mean final-field L2 | SD final-field L2 | mean clean observed NMSE |
| --- | --- | --- | --- | --- | --- | --- |
| matched_A/8/M1 | 0 | 0.4 | 0 | 0.04393902 | 0.001078553 | 0.002119509 |
| matched_A/8/M4 | 1 | 0.25 | 0 | 0.00520597 | 0.002098371 | 3.179724e-05 |
| matched_A/4/M1 | 0 | 0.2 | 0 | 0.04315061 | 0.001907181 | 0.002711838 |
| matched_A/4/M4 | 1 | 0.15 | 0 | 0.01181584 | 0.005071284 | 0.0002502156 |

## Held-out field C empirical distributions

v:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_C/8/M1 | 50 | 1.002458 | 1.002589 | 0.001224277 | 1.0003 | 1.001837 | 1.003283 | 1.004168 |
| continuum_C/8/M4 | 50 | 0.998509 | 0.9984082 | 0.002125629 | 0.9949545 | 0.9972789 | 0.9996659 | 1.002175 |
| continuum_C/8/M5 | 50 | 1.000491 | 1.000627 | 0.001225193 | 0.9983234 | 0.9998648 | 1.001342 | 1.002201 |

ν4:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_C/8/M1 | 50 | 0.002019123 | 0.002018131 | 4.645986e-05 | 0.001940939 | 0.001991524 | 0.002045683 | 0.002089268 |
| continuum_C/8/M4 | 50 | 0.001954054 | 0.001920655 | 0.0001089296 | 0.001843549 | 0.00188034 | 0.001980124 | 0.002179306 |
| continuum_C/8/M5 | 50 | 0.002012712 | 0.002012384 | 4.624722e-05 | 0.00193483 | 0.001985346 | 0.002038921 | 0.002082091 |

c3:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_C/8/M4 | 50 | 1.109914 | 1.110646 | 0.04331824 | 1.033622 | 1.079373 | 1.135671 | 1.17193 |

c6:

| case | n | mean | median | sample SD | p05 | p25 | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_C/8/M4 | 50 | 1.545461 | 2 | 0.6834422 | 0 | 1.221669 | 2 | 2 |

| case | discrepancy-bound fraction | continuation fraction (selected fit) | unconverged fraction | mean final-field L2 | SD final-field L2 | mean clean observed NMSE |
| --- | --- | --- | --- | --- | --- | --- |
| continuum_C/8/M1 | 0 | 0.06 | 0 | 0.01063963 | 0.001490068 | 0.0001055654 |
| continuum_C/8/M4 | 0.62 | 0.14 | 0 | 0.007277854 | 0.002165313 | 5.005566e-05 |
| continuum_C/8/M5 | 0 | 0.1 | 0 | 0.007397486 | 0.001916906 | 5.402747e-05 |

Boundary point masses in M4 distributions (fractions, not a Gaussian approximation):

| case | c3 lower | c3 upper | c6 lower | c6 upper |
| --- | --- | --- | --- | --- |
| continuum_A/8/M4 | 0 | 0 | 0.45 | 0.41 |
| continuum_A/4/M4 | 0 | 0 | 0.42 | 0.5 |
| continuum_C/8/M4 | 0 | 0 | 0.08 | 0.54 |
| matched_A/8/M4 | 0.5 | 0 | 0.5 | 0.4 |
| matched_A/4/M4 | 0.4 | 0 | 0.45 | 0.5 |

## Physical bias versus pseudo-true sampling deviations

Every run stores the exact identity physical error=clean model/design bias+noise-induced shift, in [v,ν4]. The ensemble tables below use signed means. The full mean/median/SD/quantile distributions of physical errors and pseudo-deviations are also stored. For free discrepancy, c3/c6 deviations from their clean pseudo-values are recorded separately.

| case | clean physical bias [v,ν4] | mean physical error [v,ν4] | mean pseudo deviation [v,ν4] |
| --- | --- | --- | --- |
| continuum_A/8/M1 | [0.0003764286, 9.357054e-06] | [0.0003058189, -3.650891e-06] | [-7.060976e-05, -1.300795e-05] |
| continuum_A/8/M4 | [-0.0007543242, -1.405859e-05] | [-0.0008533488, 2.880993e-06] | [-9.902457e-05, 1.693958e-05] |
| continuum_A/8/M5 | [-0.0004489361, 2.532043e-06] | [-0.0005203699, -1.036461e-05] | [-7.143381e-05, -1.289665e-05] |
| continuum_A/4/M1 | [0.0005420545, 3.529382e-06] | [-0.0004883025, -2.612971e-05] | [-0.001030357, -2.965909e-05] |
| continuum_A/4/M4 | [-0.0006577352, 2.684582e-05] | [-0.002626543, -3.14132e-05] | [-0.001968808, -5.825902e-05] |
| continuum_A/4/M5 | [-0.0003980875, -2.914773e-06] | [-0.001428463, -3.283692e-05] | [-0.001030375, -2.992215e-05] |
| continuum_C/8/M1 | [0.002549547, 6.802834e-06] | [0.002458334, 1.912257e-05] | [-9.121314e-05, 1.231973e-05] |
| continuum_C/8/M4 | [-0.001685591, -8.997378e-05] | [-0.001491007, -4.5946e-05] | [0.0001945844, 4.402778e-05] |
| continuum_C/8/M5 | [0.0005771842, 4.401389e-07] | [0.0004908307, 1.271185e-05] | [-8.635357e-05, 1.227171e-05] |
| matched_A/8/M1 | [-0.01469511, 2.586084e-05] | [-0.01489912, 1.151281e-05] | [-0.0002040016, -1.434803e-05] |
| matched_A/8/M4 | [-1.998401e-15, 1.18178e-15] | [-0.0006403793, -0.0001044821] | [-0.0006403793, -0.0001044821] |
| matched_A/4/M1 | [-0.01677636, 5.806301e-05] | [-0.01771839, 8.535572e-05] | [-0.0009420309, 2.729271e-05] |
| matched_A/4/M4 | [-3.508567e-07, -5.041924e-07] | [-0.002818785, -5.748041e-05] | [-0.002818434, -5.697622e-05] |

| case | physical RMSE [v,ν4] | pseudo-deviation RMSE [v,ν4] |
| --- | --- | --- |
| continuum_A/8/M1 | [0.001393448, 0.0001449944] | [0.001361308, 0.000145531] |
| continuum_A/8/M4 | [0.002025453, 0.0001773977] | [0.001839581, 0.0001781813] |
| continuum_A/8/M5 | [0.001451677, 0.0001447918] | [0.001357087, 0.0001449951] |
| continuum_A/4/M1 | [0.003535978, 0.0002428107] | [0.003650525, 0.0002432158] |
| continuum_A/4/M4 | [0.005374389, 0.000255443] | [0.005085424, 0.0002601124] |
| continuum_A/4/M5 | [0.003776185, 0.0002424282] | [0.003644275, 0.0002420506] |
| continuum_C/8/M1 | [0.002740854, 4.980985e-05] | [0.0012154, 4.761433e-05] |
| continuum_C/8/M4 | [0.00257896, 0.0001172151] | [0.002113243, 0.0001164765] |
| continuum_C/8/M5 | [0.001308431, 4.751442e-05] | [0.001215949, 4.739856e-05] |
| matched_A/8/M1 | [0.0149512, 0.0001252123] | [0.001263416, 0.0001255048] |
| matched_A/8/M4 | [0.001757747, 0.0001771493] | [0.001757747, 0.0001771493] |
| matched_A/4/M1 | [0.01799924, 0.0002297065] | [0.003304326, 0.0002149985] |
| matched_A/4/M4 | [0.004868811, 0.0001936515] | [0.004868608, 0.0001935025] |

## Nominal coverage and empirical LR calibration

Coverage of pseudo-truth and physical truth is reported separately. Indeterminate profiles are neither silently discarded nor counted as ordinary misses: resolved-subset rates and worst/best all-subset coverage bounds are both shown. Known missing intervals therefore cannot improve a reported denominator. The empirical LR quantiles are descriptive estimates from only 20 or 10 replicates, not replacement thresholds or a validated recalibration. No seed was selected by fit quality.

| case | subset n | valid LR count | empirical LR p68 | nominal 68 threshold | empirical LR p95 | nominal 95 threshold |
| --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M1 | 20 | 20 | 1.667298 | 1 | 2.384133 | 3.841459 |
| continuum_A/8/M4 | 20 | 20 | 0.7118254 | 1 | 1.99694 | 3.841459 |
| continuum_A/8/M5 | 20 | 20 | 1.697363 | 1 | 2.372173 | 3.841459 |
| continuum_A/4/M1 | 10 | 10 | 0.947715 | 1 | 3.260298 | 3.841459 |
| continuum_A/4/M4 | 10 | 10 | 0.5422396 | 1 | 1.801595 | 3.841459 |
| continuum_A/4/M5 | 10 | 10 | 0.9359876 | 1 | 3.234066 | 3.841459 |
| continuum_C/8/M1 | 10 | 10 | 0.3486562 | 1 | 1.301602 | 3.841459 |
| continuum_C/8/M4 | 10 | 10 | 0.168605 | 1 | 1.646653 | 3.841459 |
| continuum_C/8/M5 | 10 | 10 | 0.3727197 | 1 | 1.286176 | 3.841459 |

| case | level | resolved n | indeterminate n | pseudo coverage resolved | physical coverage resolved | pseudo all-subset bounds | physical all-subset bounds | mean ν4 width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M1 | 68 | 20 | 0 | 0.5 | 0.55 | [0.5, 0.5] | [0.55, 0.55] | 0.000259982 |
| continuum_A/8/M1 | 95 | 20 | 0 | 1 | 0.95 | [1, 1] | [0.95, 0.95] | 0.0005096768 |
| continuum_A/8/M4 | 68 | 20 | 0 | 0.85 | 0.8 | [0.85, 0.85] | [0.8, 0.8] | 0.0003715928 |
| continuum_A/8/M4 | 95 | 20 | 0 | 1 | 1 | [1, 1] | [1, 1] | 0.0006714041 |
| continuum_A/8/M5 | 68 | 20 | 0 | 0.5 | 0.5 | [0.5, 0.5] | [0.5, 0.5] | 0.0002592351 |
| continuum_A/8/M5 | 95 | 20 | 0 | 0.95 | 0.95 | [0.95, 0.95] | [0.95, 0.95] | 0.0005082126 |
| continuum_A/4/M1 | 68 | 10 | 0 | 0.7 | 0.7 | [0.7, 0.7] | [0.7, 0.7] | 0.0005361488 |
| continuum_A/4/M1 | 95 | 10 | 0 | 0.9 | 0.9 | [0.9, 0.9] | [0.9, 0.9] | 0.001051979 |
| continuum_A/4/M4 | 68 | 10 | 0 | 0.9 | 0.8 | [0.9, 0.9] | [0.8, 0.8] | 0.000706913 |
| continuum_A/4/M4 | 95 | 10 | 0 | 1 | 1 | [1, 1] | [1, 1] | 0.001244743 |
| continuum_A/4/M5 | 68 | 10 | 0 | 0.7 | 0.7 | [0.7, 0.7] | [0.7, 0.7] | 0.0005339038 |
| continuum_A/4/M5 | 95 | 10 | 0 | 0.9 | 0.9 | [0.9, 0.9] | [0.9, 0.9] | 0.001047574 |
| continuum_C/8/M1 | 68 | 10 | 0 | 0.9 | 0.9 | [0.9, 0.9] | [0.9, 0.9] | 9.269866e-05 |
| continuum_C/8/M1 | 95 | 10 | 0 | 1 | 1 | [1, 1] | [1, 1] | 0.0001817111 |
| continuum_C/8/M4 | 68 | 10 | 0 | 0.8 | 0.9 | [0.8, 0.8] | [0.9, 0.9] | 0.0002169181 |
| continuum_C/8/M4 | 95 | 10 | 0 | 1 | 1 | [1, 1] | [1, 1] | 0.0004012796 |
| continuum_C/8/M5 | 68 | 10 | 0 | 0.9 | 0.9 | [0.9, 0.9] | [0.9, 0.9] | 9.211984e-05 |
| continuum_C/8/M5 | 95 | 10 | 0 | 1 | 1 | [1, 1] | [1, 1] | 0.0001805762 |

## M1/M4/M5 uncertainty and state-prediction comparison

| case | clean ν4 bias | clean 95% ν4 width | bootstrap ν4 SD | mean final L2 | SD final L2 | discrepancy-bound fraction |
| --- | --- | --- | --- | --- | --- | --- |
| continuum_A/8/M1 | 9.357054e-06 | 0.00051026 | 0.0001456787 | 0.006207495 | 0.002335888 | 0 |
| continuum_A/8/M4 | -1.405859e-05 | 0.0007039085 | 0.0001782679 | 0.006133933 | 0.002636054 | 0.86 |
| continuum_A/8/M5 | 2.532043e-06 | 0.0005087818 | 0.0001451479 | 0.005135913 | 0.002616369 | 0 |
| continuum_A/4/M1 | 3.529382e-06 | 0.001030946 | 0.0002438514 | 0.01179288 | 0.004652727 | 0 |
| continuum_A/4/M4 | 2.684582e-05 | 0.001229725 | 0.0002560779 | 0.01271295 | 0.004366247 | 0.92 |
| continuum_A/4/M5 | -2.914773e-06 | 0.001026642 | 0.0002426326 | 0.01120016 | 0.00476023 | 0 |

For full-clean reference data, the M4/M1 nominal 95% diffusion-width ratio is 1.748. M4 profile width is 0.0004385415, versus an unconstrained local inverse-curvature width 0.001091837. The quadratic approximation is therefore not automatically an underestimate; discrepancy bounds constrain compensation that the unconstrained inverse Hessian permits.

For moderate data, the M4/M1 nominal 95% diffusion-width ratio is 1.380. M4 profile width is 0.0007039085, versus an unconstrained local inverse-curvature width 0.002248392. The quadratic approximation is therefore not automatically an underestimate; discrepancy bounds constrain compensation that the unconstrained inverse Hessian permits.

For severe data, the M4/M1 nominal 95% diffusion-width ratio is 1.193. M4 profile width is 0.001229725, versus an unconstrained local inverse-curvature width 0.01013562. The quadratic approximation is therefore not automatically an underestimate; discrepancy bounds constrain compensation that the unconstrained inverse Hessian permits.

Widening both discrepancy bounds from 2 to 4 increases the moderate clean diffusion profile width by 19.66%. The c6 profile reaches its declared limits, so its interval is bound-limited. This is dependence on the discrepancy model, not an intrinsic physical uncertainty limit; diffusion itself still has finite threshold crossings on the tested boxes.

In the moderate A ensemble, diffusion SDs for M1/M4/M5 are 0.0001456787, 0.0001782679, 0.0001451479. The M4 discrepancy-bound fraction is 86.0%. Its mean final-field relative L2 is 0.0061339, compared with 0.0062075 for M1. Small state error can coexist with broad or boundary-concentrated discrepancy estimates.

M5/M1 diffusion SD ratio is 0.9964. Their clean signed diffusion biases are 2.532043e-06 and 9.357054e-06, respectively. Fixed calibration changes model bias while leaving only v,q free; these conditional SDs exclude uncertainty in the calibration itself.

In the severe A ensemble, diffusion SDs for M1/M4/M5 are 0.0002438514, 0.0002560779, 0.0002426326. The M4 discrepancy-bound fraction is 92.0%. Its mean final-field relative L2 is 0.012713, compared with 0.011793 for M1. Small state error can coexist with broad or boundary-concentrated discrepancy estimates.

M5/M1 diffusion SD ratio is 0.9950. Their clean signed diffusion biases are -2.914773e-06 and 3.529382e-06, respectively. Fixed calibration changes model bias while leaving only v,q free; these conditional SDs exclude uncertainty in the calibration itself.

Coverage tables show both pseudo-truth and physical-truth inclusion. The empirical LR quantiles are not forced to equal the asymptotic references, and they are not substituted into the reported nominal intervals. With only 10 or 20 profile replicates per case, deviations are descriptive rather than evidence of precisely estimated coverage.

For held-out field C, diffusion SDs for M1/M4/M5 are 4.645986e-05, 0.0001089296, 4.624722e-05; M4 hits a discrepancy bound in 62.0% of fits. M5 retains the same N32 calibration throughout. Field-C coverage is evaluated independently on seeds 900–909, with physical and pseudo-targets separated.

## Optimizer diagnostics and failed points

| quantity | value |
| --- | --- |
| attempts | 48336 |
| continued_attempts | 13641 |
| nonstationary_attempts | 5759 |
| selected_invalid_profile_points | 0 |
| invalid_surface_points | 0 |
| invalid_bootstrap_fits | 0 |
| invalid_pseudo_fits | 0 |
| max_selected_pseudo_KKT | 1.472477e-08 |

Spread across converged starts near the nominal thresholds (ΔNLL units): {'near_threshold_converged_start_NLL_spread_max': 0.00010768729084463524, 'near_threshold_converged_start_NLL_spread_p95': 8.471552348510169e-11}. These diagnostics expose finite-tolerance and possible branch differences; the smallest converged value is retained.

Adjacent LR slopes and second divided differences are stored for every curve without crossing failed points. Nuisance jumps and finite-grid extrema are inspection flags, not automatic claims of branch switching or nonsmoothness. Every flagged point already received central, alternate, and neighboring warm-start optimizations; no smoothing is applied.

Counts distinguish all attempted starts from selected endpoints. A failed alternate start does not invalidate a point if another predeclared start converges. A point with no converged start is invalid and never interpolated through. Contact arrays refer to the final optimization segment; fixed-coordinate contacts are not evidence of nuisance-bound activity. Both original 2000-update and final optimizer states retain moments and absolute counts. No failed run is deleted or extended beyond its cap.

| attempt group | nonstationary attempts |
| --- | --- |
| bootstrap/continuum_A/4/M1 | 25 |
| bootstrap/continuum_A/4/M4 | 28 |
| bootstrap/continuum_A/4/M5 | 20 |
| bootstrap/continuum_A/8/M1 | 16 |
| bootstrap/continuum_A/8/M4 | 22 |
| bootstrap/continuum_A/8/M5 | 28 |
| bootstrap/continuum_C/8/M1 | 21 |
| bootstrap/continuum_C/8/M4 | 22 |
| bootstrap/continuum_C/8/M5 | 18 |
| bootstrap/matched_A/4/M1 | 10 |
| bootstrap/matched_A/4/M4 | 10 |
| bootstrap/matched_A/8/M1 | 3 |
| bootstrap/matched_A/8/M4 | 6 |
| bound-profile/2/1 | 54 |
| bound-profile/2/3 | 32 |
| bound-profile/3/1 | 51 |
| bound-profile/3/3 | 35 |
| bound-profile/4/1 | 45 |
| bound-profile/4/3 | 35 |
| coverage/continuum_A/4/M1/800 | 8 |
| coverage/continuum_A/4/M1/801 | 6 |
| coverage/continuum_A/4/M1/802 | 7 |
| coverage/continuum_A/4/M1/803 | 3 |
| coverage/continuum_A/4/M1/804 | 1 |
| coverage/continuum_A/4/M1/805 | 2 |
| coverage/continuum_A/4/M1/806 | 7 |
| coverage/continuum_A/4/M1/807 | 7 |
| coverage/continuum_A/4/M1/808 | 2 |
| coverage/continuum_A/4/M1/809 | 6 |
| coverage/continuum_A/4/M4/800 | 29 |
| coverage/continuum_A/4/M4/801 | 11 |
| coverage/continuum_A/4/M4/802 | 30 |
| coverage/continuum_A/4/M4/803 | 26 |
| coverage/continuum_A/4/M4/804 | 22 |
| coverage/continuum_A/4/M4/805 | 7 |
| coverage/continuum_A/4/M4/806 | 26 |
| coverage/continuum_A/4/M4/807 | 19 |
| coverage/continuum_A/4/M4/808 | 16 |
| coverage/continuum_A/4/M4/809 | 18 |
| coverage/continuum_A/4/M5/800 | 8 |
| coverage/continuum_A/4/M5/801 | 7 |
| coverage/continuum_A/4/M5/802 | 8 |
| coverage/continuum_A/4/M5/803 | 4 |
| coverage/continuum_A/4/M5/804 | 1 |
| coverage/continuum_A/4/M5/806 | 8 |
| coverage/continuum_A/4/M5/807 | 7 |
| coverage/continuum_A/4/M5/808 | 1 |
| coverage/continuum_A/4/M5/809 | 9 |
| coverage/continuum_A/8/M1/800 | 33 |
| coverage/continuum_A/8/M1/801 | 48 |
| coverage/continuum_A/8/M1/802 | 39 |
| coverage/continuum_A/8/M1/803 | 40 |
| coverage/continuum_A/8/M1/804 | 38 |
| coverage/continuum_A/8/M1/805 | 41 |
| coverage/continuum_A/8/M1/806 | 29 |
| coverage/continuum_A/8/M1/807 | 42 |
| coverage/continuum_A/8/M1/808 | 39 |
| coverage/continuum_A/8/M1/809 | 28 |
| coverage/continuum_A/8/M1/810 | 37 |
| coverage/continuum_A/8/M1/811 | 36 |
| coverage/continuum_A/8/M1/812 | 34 |
| coverage/continuum_A/8/M1/813 | 34 |
| coverage/continuum_A/8/M1/814 | 41 |
| coverage/continuum_A/8/M1/815 | 32 |
| coverage/continuum_A/8/M1/816 | 43 |
| coverage/continuum_A/8/M1/817 | 40 |
| coverage/continuum_A/8/M1/818 | 37 |
| coverage/continuum_A/8/M1/819 | 38 |
| coverage/continuum_A/8/M4/800 | 39 |
| coverage/continuum_A/8/M4/801 | 50 |
| coverage/continuum_A/8/M4/802 | 48 |
| coverage/continuum_A/8/M4/803 | 59 |
| coverage/continuum_A/8/M4/804 | 29 |
| coverage/continuum_A/8/M4/805 | 51 |
| coverage/continuum_A/8/M4/806 | 48 |
| coverage/continuum_A/8/M4/807 | 51 |
| coverage/continuum_A/8/M4/808 | 50 |
| coverage/continuum_A/8/M4/809 | 51 |
| coverage/continuum_A/8/M4/810 | 61 |
| coverage/continuum_A/8/M4/811 | 55 |
| coverage/continuum_A/8/M4/812 | 51 |
| coverage/continuum_A/8/M4/813 | 52 |
| coverage/continuum_A/8/M4/814 | 55 |
| coverage/continuum_A/8/M4/815 | 43 |
| coverage/continuum_A/8/M4/816 | 53 |
| coverage/continuum_A/8/M4/817 | 61 |
| coverage/continuum_A/8/M4/818 | 54 |
| coverage/continuum_A/8/M4/819 | 48 |
| coverage/continuum_A/8/M5/800 | 38 |
| coverage/continuum_A/8/M5/801 | 31 |
| coverage/continuum_A/8/M5/802 | 34 |
| coverage/continuum_A/8/M5/803 | 37 |
| coverage/continuum_A/8/M5/804 | 28 |
| coverage/continuum_A/8/M5/805 | 37 |
| coverage/continuum_A/8/M5/806 | 47 |
| coverage/continuum_A/8/M5/807 | 38 |
| coverage/continuum_A/8/M5/808 | 39 |
| coverage/continuum_A/8/M5/809 | 26 |
| coverage/continuum_A/8/M5/810 | 33 |
| coverage/continuum_A/8/M5/811 | 35 |
| coverage/continuum_A/8/M5/812 | 39 |
| coverage/continuum_A/8/M5/813 | 42 |
| coverage/continuum_A/8/M5/814 | 35 |
| coverage/continuum_A/8/M5/815 | 42 |
| coverage/continuum_A/8/M5/816 | 32 |
| coverage/continuum_A/8/M5/817 | 44 |
| coverage/continuum_A/8/M5/818 | 36 |
| coverage/continuum_A/8/M5/819 | 34 |
| coverage/continuum_C/8/M1/900 | 57 |
| coverage/continuum_C/8/M1/901 | 48 |
| coverage/continuum_C/8/M1/902 | 54 |
| coverage/continuum_C/8/M1/903 | 48 |
| coverage/continuum_C/8/M1/904 | 45 |
| coverage/continuum_C/8/M1/905 | 34 |
| coverage/continuum_C/8/M1/906 | 51 |
| coverage/continuum_C/8/M1/907 | 40 |
| coverage/continuum_C/8/M1/908 | 58 |
| coverage/continuum_C/8/M1/909 | 48 |
| coverage/continuum_C/8/M4/900 | 42 |
| coverage/continuum_C/8/M4/901 | 61 |
| coverage/continuum_C/8/M4/902 | 65 |
| coverage/continuum_C/8/M4/903 | 50 |
| coverage/continuum_C/8/M4/904 | 62 |
| coverage/continuum_C/8/M4/905 | 58 |
| coverage/continuum_C/8/M4/906 | 50 |
| coverage/continuum_C/8/M4/907 | 65 |
| coverage/continuum_C/8/M4/908 | 59 |
| coverage/continuum_C/8/M4/909 | 60 |
| coverage/continuum_C/8/M5/900 | 48 |
| coverage/continuum_C/8/M5/901 | 59 |
| coverage/continuum_C/8/M5/902 | 49 |
| coverage/continuum_C/8/M5/903 | 52 |
| coverage/continuum_C/8/M5/904 | 43 |
| coverage/continuum_C/8/M5/905 | 53 |
| coverage/continuum_C/8/M5/906 | 50 |
| coverage/continuum_C/8/M5/907 | 50 |
| coverage/continuum_C/8/M5/908 | 59 |
| coverage/continuum_C/8/M5/909 | 52 |
| profile/continuum_A/4/M1/0 | 19 |
| profile/continuum_A/4/M1/1 | 6 |
| profile/continuum_A/4/M1/800/0 | 15 |
| profile/continuum_A/4/M4/0 | 18 |
| profile/continuum_A/4/M4/1 | 18 |
| profile/continuum_A/4/M4/2 | 48 |
| profile/continuum_A/4/M4/3 | 1 |
| profile/continuum_A/4/M4/800/0 | 11 |
| profile/continuum_A/4/M4/800/2 | 54 |
| profile/continuum_A/4/M4/800/3 | 3 |
| profile/continuum_A/4/M5/0 | 16 |
| profile/continuum_A/4/M5/1 | 4 |
| profile/continuum_A/4/M5/800/0 | 20 |
| profile/continuum_A/8/M1/0 | 6 |
| profile/continuum_A/8/M1/1 | 37 |
| profile/continuum_A/8/M1/800/0 | 3 |
| profile/continuum_A/8/M4/0 | 6 |
| profile/continuum_A/8/M4/1 | 49 |
| profile/continuum_A/8/M4/2 | 51 |
| profile/continuum_A/8/M4/3 | 31 |
| profile/continuum_A/8/M4/800/0 | 8 |
| profile/continuum_A/8/M4/800/2 | 46 |
| profile/continuum_A/8/M4/800/3 | 26 |
| profile/continuum_A/8/M5/0 | 8 |
| profile/continuum_A/8/M5/1 | 34 |
| profile/continuum_A/8/M5/800/0 | 7 |
| profile/continuum_A/full/M1/0 | 5 |
| profile/continuum_A/full/M1/1 | 1 |
| profile/continuum_A/full/M4/0 | 19 |
| profile/continuum_A/full/M4/1 | 7 |
| profile/continuum_A/full/M4/2 | 20 |
| profile/continuum_A/full/M4/3 | 6 |
| profile/continuum_A/full/M5/0 | 4 |
| profile/continuum_A/full/M5/1 | 4 |
| pseudo/continuum_A/full/M1 | 1 |
| pseudo/continuum_A/full/M5 | 1 |
| pseudo/continuum_C/full/M1 | 1 |
| pseudo/matched_A/4/M4 | 1 |
| surface | 302 |

| profile | invalid selected points | large nuisance jumps | baseline valid |
| --- | --- | --- | --- |
| continuum_A/full/M4/0 | 0 | 6 | yes |
| continuum_A/full/M4/1 | 0 | 3 | yes |
| continuum_A/8/M1/0 | 0 | 2 | yes |
| continuum_A/8/M4/0 | 0 | 10 | yes |
| continuum_A/8/M4/1 | 0 | 2 | yes |
| continuum_A/8/M5/0 | 0 | 2 | yes |
| continuum_A/4/M1/0 | 0 | 1 | yes |
| continuum_A/4/M4/0 | 0 | 8 | yes |
| continuum_A/4/M4/1 | 0 | 2 | yes |
| continuum_A/4/M4/2 | 0 | 2 | yes |
| continuum_A/4/M5/0 | 0 | 2 | yes |
| continuum_A/8/M4/800 | 0 | 2 | yes |
| continuum_A/8/M4/801 | 0 | 2 | yes |
| continuum_A/8/M4/802 | 0 | 2 | yes |
| continuum_A/8/M4/803 | 0 | 2 | yes |
| continuum_A/8/M4/804 | 0 | 2 | yes |
| continuum_A/8/M4/805 | 0 | 1 | yes |
| continuum_A/8/M4/806 | 0 | 2 | yes |
| continuum_A/8/M4/807 | 0 | 3 | yes |
| continuum_A/8/M4/808 | 0 | 2 | yes |
| continuum_A/8/M4/809 | 0 | 3 | yes |
| continuum_A/8/M4/810 | 0 | 3 | yes |
| continuum_A/8/M4/811 | 0 | 3 | yes |
| continuum_A/8/M4/812 | 0 | 1 | yes |
| continuum_A/8/M4/813 | 0 | 2 | yes |
| continuum_A/8/M4/814 | 0 | 3 | yes |
| continuum_A/8/M4/815 | 0 | 1 | yes |
| continuum_A/8/M4/816 | 0 | 3 | yes |
| continuum_A/8/M4/817 | 0 | 2 | yes |
| continuum_A/8/M4/818 | 0 | 1 | yes |
| continuum_A/8/M4/819 | 0 | 2 | yes |
| continuum_A/4/M4/800 | 0 | 2 | yes |
| continuum_A/4/M4/801 | 0 | 2 | yes |
| continuum_A/4/M4/802 | 0 | 2 | yes |
| continuum_A/4/M4/803 | 0 | 2 | yes |
| continuum_A/4/M4/804 | 0 | 1 | yes |
| continuum_A/4/M4/805 | 0 | 2 | yes |
| continuum_A/4/M4/806 | 0 | 2 | yes |
| continuum_A/4/M4/807 | 0 | 1 | yes |
| continuum_A/4/M4/808 | 0 | 2 | yes |
| continuum_A/4/M4/809 | 0 | 1 | yes |
| continuum_C/8/M4/900 | 0 | 2 | yes |
| continuum_C/8/M4/901 | 0 | 2 | yes |
| continuum_C/8/M4/902 | 0 | 2 | yes |
| continuum_C/8/M4/903 | 0 | 2 | yes |
| continuum_C/8/M4/904 | 0 | 3 | yes |
| continuum_C/8/M4/905 | 0 | 2 | yes |
| continuum_C/8/M4/906 | 0 | 2 | yes |
| continuum_C/8/M4/907 | 0 | 2 | yes |
| continuum_C/8/M4/908 | 0 | 3 | yes |
| continuum_C/8/M4/909 | 0 | 3 | yes |

## Tests and acceptance

```text
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 34.60s
```

| criterion | pass |
| --- | --- |
| old_and_new_tests | yes |
| Stage1_9_preserved | yes |
| pseudo_truth_separate_and_stationary | yes |
| positive_time_Gaussian_NLL | yes |
| primary_profiles_complete | yes |
| profiled_surface_complete | yes |
| profile_failures_retained_and_excluded | yes |
| Hessian_profile_comparison | yes |
| coordinate_sensitivity | yes |
| bound_sensitivity | yes |
| fresh_ensembles_complete | yes |
| paired_noise | yes |
| nominal_and_empirical_thresholds_separate | yes |
| physical_and_pseudo_coverage_separate | yes |
| heldout_C_without_recalibration | yes |
| parameter_and_field_distributions | yes |
| uniform_continuation | yes |
| frozen_M5_calibration | yes |
| primary_profiles_informative | yes |

**PASS**. Exact nominal coverage, narrower M4 uncertainty, physical-truth inclusion under misspecification, Hessian/profile agreement, and identifiable c6 are not acceptance requirements. Numerical failures remain explicit limitations.

## Caveats

Nominal profile-likelihood intervals depend on the declared synthetic Gaussian noise model. Wilks thresholds can fail under small samples, bounds, weak identifiability, nonlinear compensation, and misspecification. Pseudo-truth coverage is not physical calibration. c3/c6 are numerical discrepancy multipliers, not physical random variables. M5 conditions on frozen calibration and excludes calibration uncertainty. Hessian approximations depend on coordinates and local quadratic assumptions; nonlinear profiles expose asymmetry and bounds. Ensembles are synthetic and conditional on exactly known initial fields and iid noise. Parameter intervals do not imply equal uncertainty in predicted fields. A discrepancy-bound endpoint is a modeling restriction, not a physical limit. Full-clean nominal intervals use hypothetical precision, not a zero-noise sampling distribution. Coverage subsets of 10/20 have substantial Monte Carlo uncertainty.

## Recommended Stage 11 (not implemented)

Exponential integration is the most direct next reference for this periodic constant-coefficient linear operator: use the exact semi-discrete Fourier propagator, then benchmark an ETD or semi-implicit implementation against it. Validate forward accuracy, parameter/initial-state derivatives, frozen inverse-result equivalence, and runtime/memory scaling against stencil/scan RK4. Stage 9 already diagonalizes the RK4 polynomial for inverse evaluation; an exponential propagator would instead remove the explicit stability restriction and time-discretization error. It does not cure model bias or identifiability. Larger independent coverage studies remain necessary before claiming calibrated finite-sample confidence. No Stage 11 code is added.

## Reproduction and resume

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.stage10_identifiability --clean-profiles
.venv/bin/python -m experiments.stage10_identifiability --profile-surfaces
.venv/bin/python -m experiments.stage10_identifiability --bootstrap-moderate
.venv/bin/python -m experiments.stage10_identifiability --bootstrap-severe
.venv/bin/python -m experiments.stage10_identifiability --heldout-transfer
.venv/bin/python -m experiments.stage10_identifiability --matched-controls
.venv/bin/python -m experiments.stage10_identifiability --representative-profiles
.venv/bin/python -m experiments.stage10_identifiability --finalize
```

Atomic checkpoints reject NaN/Infinity, preserve failed attempts, and validate the Stage 9 source-result hash and ensemble seeds on resume. Finalization requires all declared groups and coverage profiles. Git status: this directory is not a Git repository. No commit, push, or repository initialization was attempted.
