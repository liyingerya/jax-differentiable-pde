# Stage 9: Explicit Model Discrepancy and Physics-Parameter Separation

**PASS**. All results are synthetic. Calibration uses known physical coefficients; inverse-physics fits do not. No Stage 10 implementation, commits, or pushes.

## Objective and preservation

Stage 8 improved worst-case recovery with robust observation design, but retained continuum-versus-discrete clean parameter bias. Stage 9 holds its masks and times fixed and asks whether explicit numerical discrepancy separates that bias from physical coefficients. New files: `src/discrepancy.py`, `experiments/stage9_model_discrepancy.py`, `experiments/stage9_report.py`, `tests/test_discrepancy.py`, and this report/results/preservation record. D3/D6 are appended to operators; the only old optimizer change is `zeros(2)` to `zeros_like(theta)` for moments. The Stage 1–8 numerical reports/results are byte-preserved. README is updated separately.

## Modified equation, stencils, and Fourier symbols

For smooth periodic fields, D1=∂x+(dx²/6)∂x³+O(dx⁴) and D4=∂x⁴+(dx²/6)∂x⁶+O(dx⁴). Thus the original negative advection and damping terms have negative leading defects. Adding the positive corrections below cancels them at c3=c6=1:

```text
D3 f_i = (f_(i+2)-2 f_(i+1)+2 f_(i-1)-f_(i-2))/(2 dx³)
D6 f_i = (f_(i-3)-6 f_(i-2)+15 f_(i-1)-20 f_i
          +15 f_(i+1)-6 f_(i+2)+f_(i+3))/dx⁶
RHS = -v D1 ψ - ν4 D4 ψ + c3 v dx² D3 ψ/6 + c6 ν4 dx² D6 ψ/6
```

D3 sin(kx)=−k³ cos(kx), D6 sin(kx)=−k⁶ sin(kx). c3 and c6 are dimensionless discrepancy multipliers tied to the finite-difference scheme, not material coefficients. D3 controls phase and D6 damping. Substituting exp(i iθ) into each stencil yields:

| Operator | Symbol |
| --- | --- |
| D1 | i sin(theta)/dx |
| D4 | 16 sin(theta/2)^4/dx^4 |
| D3 | -4 i sin(theta) sin(theta/2)^2/dx^3 |
| D6 | -64 sin(theta/2)^6/dx^6 |

In particular, sin(2θ)−2sinθ=−4sinθ sin²(θ/2) gives D3, while the cube of −4sin²(θ/2) gives D6. Their small-angle limits are −ik³ and −k⁶. The corrected eigenvalue is the identical linear combination of these symbols; continuum λ=−ivk−ν4 k⁴. Classical RK4 uses R(z)=1+z+z²/2+z³/6+z⁴/24.

The public corrected trajectory uses the requested stencils, classical RK4, and lax.scan including t=0. Production uses FFT diagonalization of that same RK4 update, R(dtλ)^integer_step. This is algebraically the finite-difference RK4 model, not exact time evolution or an exact-continuum symbol correction. It makes repeated inverse fits practical even at fine grids. Float64 is retained throughout.

Maximum full-trajectory scan/evaluator difference: 1.157e-13; loss-gradient difference: 1.377e-14. Zero-correction RHS, step, trajectory, and differentiation through the initial field are covered by pytest.

## Independent D3/D6 convergence

Relative L2 errors for sin(3x), sampled on each grid:

| N | D3 error | D6 error |
| --- | --- | --- |
| 16 | 0.3023946 | 0.2960741 |
| 24 | 0.1450225 | 0.1435961 |
| 32 | 0.0837927 | 0.08331995 |
| 48 | 0.03796373 | 0.03786722 |
| 64 | 0.02149894 | 0.02146805 |

Both second-order derivative stencils converge independently. Their use in dx²-weighted corrections cancels a lower-order defect without making either derivative stencil fourth order.

## Phase and damping cancellation

Absolute rate errors against continuum, at v=1, ν4=.002. These isolate spatial symbols, without RK4 temporal error:

| N | k | c3=c6 | phase error | damping error |
| --- | --- | --- | --- | --- |
| 16 | 1 | 0 | 0.02550464 | 5.081376e-05 |
| 16 | 1 | 1 | 0.0007782942 | 1.3561e-06 |
| 16 | 2 | 0 | 0.1993674 | 0.003141811 |
| 16 | 2 | 1 | 0.02356967 | 0.0003243547 |
| 16 | 3 | 0 | 0.6473601 | 0.03380641 |
| 16 | 3 | 1 | 0.1632522 | 0.007427734 |
| 24 | 1 | 0 | 0.01138407 | 2.272923e-05 |
| 24 | 1 | 1 | 0.0001553136 | 2.712722e-07 |
| 24 | 2 | 0 | 0.09014068 | 0.001432467 |
| 24 | 2 | 1 | 0.004849806 | 6.737572e-05 |
| 24 | 3 | 0 | 0.2990511 | 0.01590542 |
| 24 | 3 | 1 | 0.03535451 | 0.001642046 |
| 32 | 1 | 0 | 0.006413149 | 1.281395e-05 |
| 32 | 1 | 1 | 4.931794e-05 | 8.621224e-08 |
| 32 | 2 | 0 | 0.05100928 | 0.0008130201 |
| 32 | 2 | 1 | 0.001556588 | 2.16976e-05 |
| 32 | 3 | 0 | 0.170504 | 0.00912838 |
| 32 | 3 | 1 | 0.01155202 | 0.0005405427 |
| 48 | 1 | 0 | 0.002853343 | 5.704242e-06 |
| 48 | 1 | 1 | 9.766689e-06 | 1.708341e-08 |
| 48 | 2 | 0 | 0.02276814 | 0.0003636677 |
| 48 | 2 | 1 | 0.0003106271 | 4.340356e-06 |
| 48 | 3 | 0 | 0.07651392 | 0.004115914 |
| 48 | 3 | 1 | 0.002334883 | 0.0001098441 |
| 64 | 1 | 0 | 0.001605607 | 3.21044e-06 |
| 64 | 1 | 1 | 3.093001e-06 | 5.411273e-09 |
| 64 | 2 | 0 | 0.0128263 | 0.0002050233 |
| 64 | 2 | 1 | 9.863589e-05 | 1.379396e-06 |
| 64 | 3 | 0 | 0.04318456 | 0.002326925 |
| 64 | 3 | 1 | 0.0007447308 | 3.510232e-05 |

Orders fitted over N=32,48,64:

| mode/model | phase order | damping order |
| --- | --- | --- |
| k1/c0 | 1.997878 | 1.996818 |
| k1/c1 | 3.994947 | 3.993745 |
| k2/c0 | 1.991509 | 1.987285 |
| k2/c1 | 3.979784 | 3.974998 |
| k3/c0 | 1.98089 | 1.971445 |
| k3/c1 | 3.954501 | 3.943806 |

These measured orders support approximately fourth-order phase and damping cancellation on the tested resolved modes; this is not a claim about arbitrary under-resolved fields.

## Stability and frozen graph

Bounds: v∈[.5,1.5], ν4∈[1e−4,.02], c3,c6∈[0,2]. Each grid samples 625 box combinations (five values per coordinate) and 8193 Fourier angles including all discrete angles. Bisection estimates the first RK4 boundary. This is a dense numerical estimate, not an analytic proof over a continuous box. The selected power-of-two step count is at least 256, reaches T=2 exactly, represents quarter times exactly, and uses at most half the estimated limit. No fit adapts dt to its parameters.

| N | dt_max | dt | steps | fraction |
| --- | --- | --- | --- | --- |
| 16 | 0.08871228 | 0.0078125 | 256 | 0.0880656 |
| 24 | 0.01752341 | 0.0078125 | 256 | 0.4458321 |
| 32 | 0.005544518 | 0.001953125 | 1024 | 0.3522624 |
| 48 | 0.001095213 | 0.0004882812 | 4096 | 0.4458321 |
| 64 | 0.0003465324 | 0.0001220703 | 16384 | 0.3522624 |

## Model families and optimizer calibration

| Model | Free coordinates | Fixed discrepancy |
| --- | --- | --- |
| M0 | v,q | 0,0 |
| M1 | v,q | 1,1 |
| M2 | v,q,c3 | c6=0 |
| M3 | v,q,c6 | c3=0 |
| M4 | v,q,c3,c6 | none |
| M5 | v,q | [1.056254, 1.061871] |

q=log(ν4); discrepancy uses direct bounded coordinates. Adam retains β1=.9, β2=.999, ε=1e−8 and box projection. Every fit begins with 2000 updates. Only KKT≥1e−7 triggers continuation of the same parameters, moments, and absolute bias-correction count. The KKT residual zeros outward-blocked gradient components at active bounds (tolerance 1e−8), then takes the Euclidean norm. No restart or per-seed tuning occurs.

Before any production inference, clean full matched, clean full continuum, and both sparse continuum controls were run at four common learning rates. The frozen selection rule first minimizes failures, then maximum residual; the original .05 is retained if it passes every control. M4 6000-update candidates are considered only when every tested 4000-update setting has failures. All pilot endpoints are retained in JSON; pilot observations contain no held-out recovery noise.

| Model | lr | cap | pilot failures | max pilot KKT |
| --- | --- | --- | --- | --- |
| M0 | 0.05 | 4000 | 0 | 1.813224e-14 |
| M1 | 0.05 | 4000 | 0 | 6.916902e-14 |
| M2 | 0.05 | 4000 | 0 | 2.491451e-14 |
| M3 | 0.05 | 4000 | 0 | 1.641303e-13 |
| M4 | 0.05 | 4000 | 0 | 8.698222e-11 |
| M5 | 0.05 | 4000 | 0 | 1.281958e-14 |

| Model | pilot lr | cap | failures | max KKT |
| --- | --- | --- | --- | --- |
| M0 | 0.05 | 4000 | 0 | 1.813224e-14 |
| M0 | 0.02 | 4000 | 0 | 1.95907e-14 |
| M0 | 0.01 | 4000 | 0 | 1.34913e-14 |
| M0 | 0.005 | 4000 | 0 | 4.40153e-15 |
| M1 | 0.05 | 4000 | 0 | 6.916902e-14 |
| M1 | 0.02 | 4000 | 0 | 2.544229e-14 |
| M1 | 0.01 | 4000 | 0 | 3.134721e-14 |
| M1 | 0.005 | 4000 | 0 | 7.116689e-15 |
| M2 | 0.05 | 4000 | 0 | 2.491451e-14 |
| M2 | 0.02 | 4000 | 0 | 4.909207e-15 |
| M2 | 0.01 | 4000 | 0 | 1.420532e-14 |
| M2 | 0.005 | 4000 | 0 | 5.703155e-14 |
| M3 | 0.05 | 4000 | 0 | 1.641303e-13 |
| M3 | 0.02 | 4000 | 0 | 2.135807e-09 |
| M3 | 0.01 | 4000 | 0 | 7.539556e-08 |
| M3 | 0.005 | 4000 | 1 | 2.501836e-07 |
| M4 | 0.05 | 4000 | 0 | 8.698222e-11 |
| M4 | 0.02 | 4000 | 0 | 2.914675e-08 |
| M4 | 0.01 | 4000 | 0 | 4.97963e-08 |
| M4 | 0.005 | 4000 | 1 | 1.127861e-07 |
| M5 | 0.05 | 4000 | 0 | 1.281958e-14 |
| M5 | 0.02 | 4000 | 0 | 6.648958e-15 |
| M5 | 0.01 | 4000 | 0 | 5.937626e-15 |
| M5 | 0.005 | 4000 | 0 | 2.930698e-15 |

## Four-coordinate gradient verification

At [v,q,c3,c6]=[.9,log(.0015),.7,1.3], centered h=1e−5 differences use the identical fixed graph and observations. The sparse check uses frozen 8-sensor data and seed 699, separate from production seeds.

| dataset | coordinate | JAX | finite difference | absolute difference | relative difference |
| --- | --- | --- | --- | --- | --- |
| full | v | -0.5223437 | -0.5223437 | 6.170331e-11 | 1.181278e-10 |
| full | q | -0.002024088 | -0.002024088 | 4.937189e-11 | 2.439217e-08 |
| full | c3 | -0.01028747 | -0.01028747 | 2.300858e-11 | 2.236564e-09 |
| full | c6 | -8.968694e-05 | -8.968703e-05 | 8.490608e-11 | 9.466939e-07 |
| sparse_noisy | v | -1.24318 | -1.24318 | 2.818543e-10 | 2.267205e-10 |
| sparse_noisy | q | -0.005597157 | -0.005597158 | 5.821458e-10 | 1.040074e-07 |
| sparse_noisy | c3 | -0.01719516 | -0.01719516 | 2.540225e-10 | 1.477291e-08 |
| sparse_noisy | c6 | -0.0001641401 | -0.0001641412 | 1.0844e-09 | 6.60655e-06 |

## Known-physics calibration, not inverse-physics recovery

Field A, all nine times, all spatial points, noiseless continuum data; v=1 and ν4=.002 are held at truth. Only c3,c6 are optimized, from six predeclared starts. Calibration uses lr=.05 and the same 2000→4000 rule. Every parameter/loss history is saved. A common optimum is accepted only if all residuals pass and coefficient spread is below 1e−4; the lowest-loss endpoint then defines frozen M5.

| N32 start | c3 | c6 | loss | KKT | updates |
| --- | --- | --- | --- | --- | --- |
| [0, 0] | 1.056254 | 1.061871 | 7.665065e-07 | 1.237645e-17 | 2000 |
| [0.5, 0.5] | 1.056254 | 1.061871 | 7.665065e-07 | 1.237645e-17 | 2000 |
| [1, 1] | 1.056254 | 1.061871 | 7.665065e-07 | 1.8363e-16 | 2000 |
| [1.5, 1.5] | 1.056254 | 1.061871 | 7.665065e-07 | 2.963937e-17 | 2000 |
| [0, 2] | 1.056254 | 1.061871 | 7.665065e-07 | 2.900081e-17 | 2000 |
| [2, 0] | 1.056254 | 1.061871 | 7.665065e-07 | 1.237645e-17 | 2000 |

Cross-grid calibration (the same six starts at each resolution):

| N | c3 | c6 | best loss | max KKT | coefficient spread | common |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1.056254 | 1.061871 | 7.665065e-07 | 1.8363e-16 | [5.329071e-14, 1.112954e-11] | yes |
| 16 | 1.239437 | 1.337991 | 0.0002087082 | 6.316756e-17 | [1.554312e-15, 9.348078e-14] | yes |
| 24 | 1.101739 | 1.115838 | 7.788028e-06 | 4.620399e-17 | [5.551115e-15, 6.488143e-13] | yes |
| 48 | 1.024689 | 1.026765 | 2.953667e-08 | 3.907329e-10 | [5.345852e-07, 2.035357e-09] | yes |
| 64 | 1.013826 | 1.014939 | 2.943941e-09 | 8.709708e-15 | [3.667799e-11, 4.196798e-11] | yes |

The table quantifies both finite-grid departure from leading theory and approach toward one. Each grid uses its globally safe timestep above. M5 always retains the N=32 coefficients, including the independent cross-grid transfer below; calibration at other grids is diagnostic only.

## Clean matched original-model control

Truth is generated by the original validated stencil/scan solver at the Stage 9 fixed dt. M1 is intentionally misspecified for these data. Free discrepancy may reach active zero bounds; projected stationarity is appropriate there.

| case/model | v | ν4 | c3 | c6 | rel v error | rel ν4 error | NMSE | final L2 | KKT | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 1 | 0.002 | 0 | 0 | 0 | 5.171644e-13 | 5.474965e-28 | 3.944093e-14 | 5.169506e-16 | 2000 |
| M1 | 0.9790755 | 0.001977337 | 1 | 1 | 0.02092446 | 0.01133157 | 0.0006447161 | 0.04142268 | 8.996157e-15 | 2000 |
| M2 | 1 | 0.002 | 1.839852e-15 | 0 | 1.110223e-16 | 3.254775e-13 | 5.47955e-28 | 3.948991e-14 | 1.011864e-15 | 2000 |
| M3 | 1 | 0.002 | 0 | 6.80438e-08 | 0 | 3.412999e-09 | 9.015716e-22 | 4.996845e-11 | 3.741042e-14 | 2000 |
| M4 | 1 | 0.002 | 3.73275e-15 | 2.84664e-07 | 4.440892e-16 | 1.427686e-08 | 1.577912e-20 | 2.090383e-10 | 1.525438e-13 | 2000 |
| M5 | 0.9779172 | 0.001979699 | 1.056254 | 1.061871 | 0.02208277 | 0.01015072 | 0.0007172442 | 0.04368607 | 5.451266e-15 | 2000 |

## Clean continuum comparison and phase/damping ablations

| case/model | v | ν4 | c3 | c6 | rel v error | rel ν4 error | NMSE | final L2 | KKT | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 1.021861 | 0.002201215 | 0 | 0 | 0.02186117 | 0.1006075 | 0.0007666544 | 0.04499283 | 6.112537e-16 | 2000 |
| M1 | 1.00092 | 0.002006258 | 1 | 1 | 0.0009196993 | 0.003128766 | 4.098049e-06 | 0.003258612 | 4.503791e-16 | 2000 |
| M2 | 0.999318 | 0.002105733 | 1.076873 | 0 | 0.0006820245 | 0.05286651 | 5.68164e-07 | 0.001252511 | 4.345284e-16 | 2000 |
| M3 | 1.021847 | 0.002002278 | 0 | 2 | 0.0218471 | 0.00113913 | 0.0007652456 | 0.04493638 | 2.986658e-15 | 2000 |
| M4 | 0.999318 | 0.001996882 | 1.076854 | 1.092514 | 0.0006819682 | 0.001558842 | 3.382162e-07 | 0.0009663798 | 4.420703e-13 | 2000 |
| M5 | 0.9997471 | 0.001999859 | 1.056254 | 1.061871 | 0.0002528619 | 7.053685e-05 | 6.077613e-07 | 0.001252235 | 1.199401e-15 | 2000 |

Signed clean physical biases and changes from M0 (these are fitted interactions, not an assumed additive decomposition):

| model | v bias | ν4 bias | v change from M0 | ν4 change from M0 |
| --- | --- | --- | --- | --- |
| M0 | 0.02186117 | 0.0002012151 | 0 | 0 |
| M1 | 0.0009196993 | 6.257532e-06 | -0.02094147 | -0.0001949575 |
| M2 | -0.0006820245 | 0.000105733 | -0.02254319 | -9.548204e-05 |
| M3 | 0.0218471 | 2.278259e-06 | -1.407168e-05 | -0.0001989368 |
| M4 | -0.0006819682 | -3.117685e-06 | -0.02254314 | -0.0002043327 |
| M5 | -0.0002528619 | -1.410737e-07 | -0.02211403 | -0.0002013561 |

## Clean M4 multistart and identifiability

All eight starts were frozen before calibration/recovery. The main clean result uses the separately declared start [.9,log(.0015),1,1], not whichever estimate is closest to truth. Bound contacts count history iterates within each optimization segment; endpoint bound flags are also saved. All multistart histories remain available.

| start [v,q,c3,c6] | final [v,ν4,c3,c6] | loss | KKT | updates | endpoint bounds | final-segment contacts |
| --- | --- | --- | --- | --- | --- | --- |
| [0.8, -6.907755, 0.1, 0.1] | [0.999318, 0.001997375, 1.076854, 1.087324] | 3.382208e-07 | 2.444412e-09 | 2000 | [no, no, no, no] | [0, 0, 0, 0] |
| [1.2, -5.521461, 0.1, 0.1] | [0.999318, 0.00199868, 1.076855, 1.073607] | 3.382782e-07 | 8.763031e-09 | 2000 | [no, no, no, no] | [0, 0, 5, 30] |
| [0.8, -5.521461, 1, 1] | [0.999318, 0.00200299, 1.076855, 1.028401] | 3.389331e-07 | 3.090878e-08 | 2000 | [no, no, no, no] | [0, 0, 0, 0] |
| [1.2, -6.907755, 1, 1] | [0.999318, 0.001996463, 1.076854, 1.096933] | 3.382195e-07 | 2.125504e-09 | 2000 | [no, no, no, no] | [0, 0, 0, 0] |
| [0.8, -6.907755, 1.8, 1.8] | [0.999318, 0.00199649, 1.076854, 1.096644] | 3.382191e-07 | 2.006141e-09 | 2000 | [no, no, no, no] | [0, 0, 3, 21] |
| [1.2, -5.521461, 1.8, 1.8] | [0.999318, 0.001998057, 1.076855, 1.080153] | 3.382427e-07 | 6.024992e-09 | 2000 | [no, no, no, no] | [0, 0, 0, 0] |
| [0.8, -5.521461, 0.2, 1.8] | [0.999318, 0.001997897, 1.076855, 1.081839] | 3.382359e-07 | 5.188775e-09 | 2000 | [no, no, no, no] | [0, 0, 0, 0] |
| [1.2, -6.907755, 1.8, 0.2] | [0.999318, 0.001997251, 1.076854, 1.088638] | 3.382188e-07 | 1.830411e-09 | 2000 | [no, no, no, no] | [0, 0, 0, 0] |

Hessian in [v,q,c3,c6], symmetrized numerically:

|  | v | q | c3 | c6 |
| --- | --- | --- | --- | --- |
| v | 4.969555 | 4.325643e-05 | 0.1034974 | 3.467655e-06 |
| q | 4.325643e-05 | 0.002717391 | 3.468643e-06 | 0.0001289204 |
| c3 | 0.1034974 | 3.468643e-06 | 0.003423902 | 2.141167e-07 |
| c6 | 3.467655e-06 | 0.0001289204 | 2.141167e-07 | 6.463025e-06 |

Eigenvalues: [3.459145e-07, 0.001267879, 0.002723512, 4.971711]. Positive-definiteness threshold: 1.103942e-13; positive definite: yes; condition number: 1.437266e+07. Eigenvectors below are columns, in ascending eigenvalue order.

|  | e1 | e2 | e3 | e4 |
| --- | --- | --- | --- | --- |
| v | 4.46433e-08 | -0.02082705 | -4.552254e-05 | -0.9997831 |
| q | -0.0473954 | -0.001764887 | 0.9988746 | -8.717938e-06 |
| c3 | -1.580193e-05 | 0.9997815 | 0.001765557 | -0.0208271 |
| c6 | 0.9988762 | -6.792443e-05 | 0.04739536 | -6.984499e-07 |

Normalized Hessian couplings Hij/√(Hii Hjj), which are not statistical correlations:

|  | v | q | c3 | c6 |
| --- | --- | --- | --- | --- |
| v | 1 | 0.0003722338 | 0.7934329 | 0.0006118701 |
| q | 0.0003722338 | 1 | 0.001137163 | 0.972809 |
| c3 | 0.7934329 | 0.001137163 | 1 | 0.001439368 |
| c6 | 0.0006118701 | 0.972809 | 0.001439368 | 1 |

## Conditional local loss geometry

Two 41×41 grids keep the other coordinates fixed: q–c6 spans q±.15,c6±.6; v–c3 spans v±.02,c3±.6, clipped to bounds. Full axes and losses are stored in JSON. These are conditional loss slices, not profile likelihoods; nuisance coordinates are not reoptimized. Strong phase/damping couplings and a small Hessian eigenvalue indicate tilted compensation valleys even if a minimum is mathematically isolated. Finite-grid curvature alone does not establish practical identifiability. Optional profile curves and an exact-symbol model are omitted to keep this study focused.

| slice | axis1 extent | axis2 extent | min loss | max loss |
| --- | --- | --- | --- | --- |
| q_c6 | [-6.366168, -6.066168] | [0.4925142, 1.692514] | 3.382162e-07 | 4.989064e-05 |
| v_c3 | [0.979318, 1.019318] | [0.4768544, 1.676854] | 3.382162e-07 | 0.002895607 |

## Clean transfer with frozen N32 discrepancy

T1=A,(1,.002); T2=C,(1,.002); T3=A,(1.05,.0018); T4=B,(.9,.0015). All use full nine-time fields and the original N32 calibration. No held-out field or parameter pair recalibrates M5.

| case/model | v | ν4 | c3 | c6 | rel v error | rel ν4 error | NMSE | final L2 | KKT | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continuum_A/M0 | 1.021861 | 0.002201215 | 0 | 0 | 0.02186117 | 0.1006075 | 0.0007666544 | 0.04499283 | 6.112537e-16 | 2000 |
| continuum_A/M1 | 1.00092 | 0.002006258 | 1 | 1 | 0.0009196993 | 0.003128766 | 4.098049e-06 | 0.003258612 | 4.503791e-16 | 2000 |
| continuum_A/M5 | 0.9997471 | 0.001999859 | 1.056254 | 1.061871 | 0.0002528619 | 7.053685e-05 | 6.077613e-07 | 0.001252235 | 1.199401e-15 | 2000 |
| continuum_C/M0 | 1.045726 | 0.002286809 | 0 | 0 | 0.04572593 | 0.1434046 | 0.003385441 | 0.09067406 | 2.184373e-15 | 2000 |
| continuum_C/M1 | 1.003494 | 0.00201941 | 1 | 1 | 0.003494391 | 0.009704924 | 4.776982e-05 | 0.009410412 | 2.513356e-15 | 2000 |
| continuum_C/M5 | 1.001154 | 0.002008796 | 1.056254 | 1.061871 | 0.00115437 | 0.004398206 | 2.076827e-05 | 0.005585569 | 2.43256e-15 | 2000 |
| continuum_offnominal/M0 | 1.073241 | 0.002000464 | 0 | 0 | 0.02213383 | 0.1113691 | 0.0008701024 | 0.04809205 | 2.408386e-16 | 2000 |
| continuum_offnominal/M1 | 1.050988 | 0.00180578 | 1 | 1 | 0.0009412499 | 0.003211174 | 4.687586e-06 | 0.003502384 | 4.759849e-16 | 2000 |
| continuum_offnominal/M5 | 1.049742 | 0.001799901 | 1.056254 | 1.061871 | 0.0002457702 | 5.481511e-05 | 6.962469e-07 | 0.001347209 | 8.899059e-16 | 2000 |
| continuum_B/M0 | 0.9339173 | 0.00183494 | 0 | 0 | 0.03768588 | 0.2232935 | 0.003629041 | 0.09007519 | 3.880107e-11 | 2000 |
| continuum_B/M1 | 0.9033002 | 0.00151757 | 1 | 1 | 0.003666936 | 0.01171357 | 6.006352e-05 | 0.01174738 | 2.488059e-10 | 4000 |
| continuum_B/M5 | 0.9015079 | 0.001507914 | 1.056254 | 1.061871 | 0.001675497 | 0.005276099 | 2.418714e-05 | 0.007451547 | 1.565059e-09 | 4000 |

Cross-grid transfer: M5 retains N32 coefficients, not each grid’s diagnostic calibration.

| case/model | v | ν4 | c3 | c6 | rel v error | rel ν4 error | NMSE | final L2 | KKT | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N16/M0 | 1.081186 | 0.004348315 | 0 | 0 | 0.0811857 | 1.174157 | 0.01152735 | 0.1700883 | 2.220871e-16 | 2000 |
| N16/M1 | 1.013086 | 0.002206329 | 1 | 1 | 0.01308578 | 0.1031646 | 0.0008277186 | 0.04619361 | 5.445838e-16 | 2000 |
| N16/M5 | 1.009133 | 0.002155783 | 1.056254 | 1.061871 | 0.009132745 | 0.07789152 | 0.0006114559 | 0.03965051 | 1.289516e-15 | 2000 |
| N24/M0 | 1.038418 | 0.002510612 | 0 | 0 | 0.03841777 | 0.2553061 | 0.002409057 | 0.07946221 | 1.776919e-15 | 2000 |
| N24/M1 | 1.002828 | 0.002022921 | 1 | 1 | 0.002828235 | 0.0114606 | 3.846249e-05 | 0.009982673 | 1.921729e-15 | 2000 |
| N24/M5 | 1.000831 | 0.002009912 | 1.056254 | 1.061871 | 0.0008305363 | 0.004955997 | 1.615884e-05 | 0.00644272 | 1.366816e-15 | 2000 |
| N32/M0 | 1.021861 | 0.002201215 | 0 | 0 | 0.02186117 | 0.1006075 | 0.0007666544 | 0.04499283 | 6.112537e-16 | 2000 |
| N32/M1 | 1.00092 | 0.002006258 | 1 | 1 | 0.0009196993 | 0.003128766 | 4.098049e-06 | 0.003258612 | 4.503791e-16 | 2000 |
| N32/M5 | 0.9997471 | 0.001999859 | 1.056254 | 1.061871 | 0.0002528619 | 7.053685e-05 | 6.077613e-07 | 0.001252235 | 1.199401e-15 | 2000 |
| N48/M0 | 1.009764 | 0.002064518 | 0 | 0 | 0.009763813 | 0.03225902 | 0.0001514184 | 0.0200246 | 6.139771e-15 | 2000 |
| N48/M1 | 1.000185 | 0.002001172 | 1 | 1 | 0.000185121 | 0.0005858632 | 1.673577e-07 | 0.0006584738 | 1.519698e-15 | 2000 |
| N48/M5 | 0.999648 | 0.001998418 | 1.056254 | 1.061871 | 0.0003520393 | 0.0007911184 | 8.593506e-08 | 0.0004875556 | 7.386966e-16 | 2000 |
| N64/M0 | 1.005498 | 0.002031626 | 0 | 0 | 0.005498006 | 0.01581321 | 4.786871e-05 | 0.01126219 | 2.492726e-14 | 2000 |
| N64/M1 | 1.000059 | 0.002000369 | 1 | 1 | 5.895364e-05 | 0.0001844649 | 1.702857e-08 | 0.0002100345 | 3.29512e-15 | 2000 |
| N64/M5 | 0.9997536 | 0.00199881 | 1.056254 | 1.061871 | 0.0002464184 | 0.0005950451 | 6.600794e-08 | 0.0004214902 | 1.156317e-14 | 2000 |

## Frozen sparse designs and common-noise protocol

| budget | sensors | physical times |
| --- | --- | --- |
| 8 | [9, 11, 13, 19, 22, 27, 29, 31] | [0, 1.75, 2] |
| 4 | [10, 21, 28, 31] | [0, 1.75, 2] |

Times are remapped exactly onto 1024 Stage 9 steps. Moderate: 8 sensors, 2% noise; severe: 4 sensors, 5%. Original-model and continuum-A experiments use seeds 700–719. Held-out continuum C uses 720–739 at the moderate budget. Noise is homoscedastic iid Gaussian, scaled by selected positive-time clean RMS, with t=0 exact. Each truth/design/seed observation array is generated once, saved with its hash, then reused verbatim across models. This preserves the Stage 7/8 recovery convention; no new design search or model-specific noise occurs.

## Zero-noise sparse references

Each reference was computed before its corresponding noisy group. If stationarity fails, it remains an approximate endpoint and the associated “clean optimum” interpretation must be qualified.

| case/model | v | ν4 | c3 | c6 | rel v error | rel ν4 error | NMSE | final L2 | KKT | updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_A/8/M0 | 1 | 0.002 | 0 | 0 | 3.108624e-15 | 8.795048e-13 | 2.415553e-27 | 4.70953e-14 | 9.174723e-15 | 2000 |
| matched_A/8/M1 | 0.9853049 | 0.002025861 | 1 | 1 | 0.01469511 | 0.01293042 | 0.002096733 | 0.04384201 | 2.910661e-14 | 2000 |
| matched_A/8/M4 | 1 | 0.002 | 2.744542e-13 | 0 | 9.992007e-16 | 5.438358e-13 | 1.783176e-27 | 4.26749e-14 | 1.020549e-14 | 2000 |
| matched_A/8/M5 | 0.9844663 | 0.00203713 | 1.056254 | 1.061871 | 0.01553365 | 0.01856514 | 0.002331859 | 0.04621557 | 1.094106e-14 | 2000 |
| matched_A/4/M0 | 1 | 0.002 | 0 | 0 | 4.884981e-15 | 9.256918e-13 | 3.802669e-27 | 4.807035e-14 | 5.553321e-14 | 2000 |
| matched_A/4/M1 | 0.9832236 | 0.002058063 | 1 | 1 | 0.01677636 | 0.0290315 | 0.002539812 | 0.04235955 | 4.189354e-14 | 2000 |
| matched_A/4/M4 | 0.9999977 | 0.001996757 | 0.0002418535 | 0.03288978 | 2.277576e-06 | 0.001621381 | 3.052325e-10 | 2.706944e-05 | 2.749471e-08 | 2000 |
| matched_A/4/M5 | 0.9823033 | 0.00206485 | 1.056254 | 1.061871 | 0.0176967 | 0.03242487 | 0.002818278 | 0.04467567 | 4.393364e-14 | 2000 |
| continuum_A/8/M0 | 1.014292 | 0.002348289 | 0 | 0 | 0.01429244 | 0.1741443 | 0.002387102 | 0.04818214 | 1.813224e-14 | 2000 |
| continuum_A/8/M1 | 1.000376 | 0.002009357 | 1 | 1 | 0.0003764286 | 0.004678527 | 1.259927e-05 | 0.003497525 | 3.785269e-14 | 2000 |
| continuum_A/8/M2 | 0.9992098 | 0.002110404 | 1.076433 | 0 | 0.00079015 | 0.05520213 | 1.196309e-06 | 0.00129677 | 1.98504e-14 | 2000 |
| continuum_A/8/M3 | 1.014292 | 0.002348289 | 0 | 0 | 0.01429244 | 0.1741443 | 0.002387102 | 0.04818214 | 2.728306e-14 | 2000 |
| continuum_A/8/M4 | 0.9992457 | 0.001985941 | 1.077398 | 1.234922 | 0.0007543242 | 0.007029294 | 5.12344e-07 | 0.0009863729 | 1.920182e-12 | 2000 |
| continuum_A/8/M5 | 0.9995511 | 0.002002532 | 1.056254 | 1.061871 | 0.0004489359 | 0.001266022 | 1.41599e-06 | 0.001337614 | 1.281958e-14 | 2000 |
| continuum_A/4/M0 | 1.017389 | 0.00221147 | 0 | 0 | 0.01738862 | 0.1057352 | 0.002934244 | 0.04592078 | 1.261801e-14 | 2000 |
| continuum_A/4/M1 | 1.000542 | 0.002003529 | 1 | 1 | 0.0005420545 | 0.001764691 | 1.311941e-05 | 0.003362933 | 6.916902e-14 | 2000 |
| continuum_A/4/M2 | 0.9994164 | 0.00210641 | 1.06459 | 0 | 0.0005836192 | 0.05320495 | 4.450286e-07 | 0.001385556 | 2.491451e-14 | 2000 |
| continuum_A/4/M3 | 1.017389 | 0.00221147 | 0 | 0 | 0.01738862 | 0.1057352 | 0.002934244 | 0.04592078 | 1.641303e-13 | 2000 |
| continuum_A/4/M4 | 0.9993423 | 0.00202684 | 1.071013 | 0.7615559 | 0.0006577409 | 0.01342006 | 2.427865e-07 | 0.001038464 | 8.698222e-11 | 2000 |
| continuum_A/4/M5 | 0.9996019 | 0.001997085 | 1.056254 | 1.061871 | 0.0003980874 | 0.001457387 | 9.803161e-07 | 0.0012972 | 6.649198e-15 | 2000 |
| continuum_C/8/M0 | 1.038411 | 0.002263741 | 0 | 0 | 0.038411 | 0.1318703 | 0.008281445 | 0.09027325 | 5.812265e-14 | 2000 |
| continuum_C/8/M1 | 1.00255 | 0.002006803 | 1 | 1 | 0.002549547 | 0.003401417 | 8.24738e-05 | 0.009394101 | 3.405521e-15 | 2000 |
| continuum_C/8/M5 | 1.000577 | 0.00200044 | 1.056254 | 1.061871 | 0.0005771842 | 0.0002200694 | 3.089964e-05 | 0.005565839 | 6.950619e-14 | 2000 |

## Sparse/noisy matched-model control

Errors below are relative fractions, not percentages. SD uses ddof=1. Rates count final bound contact, continuation, and failed final KKT respectively.

| budget/model | mean v err | median v err | mean ν err | median ν err | p90 ν err | SD v | SD ν | mean final L2 | clean NMSE | noisy loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8/M0 | 0.0009075474 | 0.0007710267 | 0.03860582 | 0.02925517 | 0.0933096 | 0.001044288 | 0.0001029958 | 0.003454069 | 1.458092e-05 | 0.000161535 |
| 8/M1 | 0.01497066 | 0.01517804 | 0.04117228 | 0.03324158 | 0.1057616 | 0.001006134 | 0.0001041727 | 0.04377779 | 0.002111616 | 0.002194124 |
| 8/M4 | 0.001035448 | 0.000823186 | 0.04681217 | 0.0217644 | 0.1240266 | 0.001180006 | 0.0001252639 | 0.004119093 | 2.052342e-05 | 0.0001535145 |
| 8/M5 | 0.01580764 | 0.01601701 | 0.04311524 | 0.03407175 | 0.111178 | 0.001004285 | 0.0001043193 | 0.04614515 | 0.00234676 | 0.00242501 |
| 4/M0 | 0.002500749 | 0.002268739 | 0.126447 | 0.114056 | 0.2141119 | 0.003020722 | 0.0003067583 | 0.01025351 | 0.0001892331 | 0.000494762 |
| 4/M1 | 0.01647021 | 0.01670643 | 0.1197349 | 0.1081486 | 0.2207027 | 0.003099412 | 0.0002898773 | 0.04392437 | 0.00273382 | 0.00348698 |
| 4/M4 | 0.002759625 | 0.002653779 | 0.1211777 | 0.102243 | 0.215086 | 0.003346445 | 0.0002786596 | 0.01068424 | 0.0001996349 | 0.0004764825 |
| 4/M5 | 0.01739034 | 0.01763389 | 0.1197879 | 0.1079234 | 0.2240082 | 0.003103164 | 0.0002890208 | 0.04617009 | 0.003012494 | 0.003789587 |

| budget/model | mean c3 | SD c3 | mean c6 | SD c6 | bound rate | continuation rate | unconverged rate | max KKT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8/M0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.663243e-13 |
| 8/M1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2.273781e-13 |
| 8/M4 | 0.02975614 | 0.04040551 | 0.9336379 | 0.9878541 | 0.95 | 0 | 0 | 2.179637e-13 |
| 8/M5 | 1.056254 | 0 | 1.061871 | 2.27813e-16 | 0 | 0 | 0 | 2.889032e-13 |
| 4/M0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.718941e-13 |
| 4/M1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1.921204e-13 |
| 4/M4 | 0.02913415 | 0.0532279 | 1.2 | 1.005249 | 1 | 0 | 0 | 3.642083e-13 |
| 4/M5 | 1.056254 | 0 | 1.061871 | 2.27813e-16 | 0 | 0 | 0 | 1.607975e-13 |

## Sparse/noisy continuum comparison

Errors below are relative fractions, not percentages. SD uses ddof=1. Rates count final bound contact, continuation, and failed final KKT respectively.

| budget/model | mean v err | median v err | mean ν err | median ν err | p90 ν err | SD v | SD ν | mean final L2 | clean NMSE | noisy loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8/M0 | 0.01395103 | 0.01381786 | 0.1850941 | 0.1797297 | 0.2695207 | 0.001103655 | 0.00011742 | 0.04868067 | 0.002402713 | 0.002602022 |
| 8/M1 | 0.0008524963 | 0.0008161299 | 0.04417758 | 0.03295815 | 0.104615 | 0.001038331 | 0.0001127558 | 0.005431805 | 2.837055e-05 | 0.0001810801 |
| 8/M2 | 0.001494171 | 0.001087445 | 0.07047183 | 0.05309679 | 0.159003 | 0.001393494 | 0.0001181213 | 0.004793953 | 2.402289e-05 | 0.0001583476 |
| 8/M3 | 0.01399321 | 0.01390162 | 0.1380019 | 0.1562343 | 0.2143046 | 0.001110645 | 0.0001480714 | 0.04860608 | 0.002403709 | 0.002598961 |
| 8/M4 | 0.001476142 | 0.001087445 | 0.05754863 | 0.05541132 | 0.1021974 | 0.001394292 | 0.0001374175 | 0.004756263 | 2.37949e-05 | 0.0001543507 |
| 8/M5 | 0.001019994 | 0.0009186619 | 0.04407519 | 0.03316866 | 0.1012901 | 0.001034793 | 0.0001125758 | 0.004143741 | 1.71951e-05 | 0.0001664233 |
| 4/M0 | 0.01771118 | 0.01765988 | 0.1460892 | 0.117898 | 0.3062284 | 0.003009654 | 0.0003172713 | 0.04716904 | 0.003125817 | 0.002924217 |
| 4/M1 | 0.002568949 | 0.002243711 | 0.1213934 | 0.1136415 | 0.1976764 | 0.00304722 | 0.0002886593 | 0.01094561 | 0.0002079171 | 0.0004722865 |
| 4/M2 | 0.003296048 | 0.002489353 | 0.1279082 | 0.09396354 | 0.2528995 | 0.004161212 | 0.0003048572 | 0.01265229 | 0.0002697096 | 0.0004167533 |
| 4/M3 | 0.01771118 | 0.01765988 | 0.1460892 | 0.117898 | 0.3062284 | 0.003009654 | 0.0003172713 | 0.04716904 | 0.003125817 | 0.002924217 |
| 4/M4 | 0.003255423 | 0.002669095 | 0.1098308 | 0.07894564 | 0.2498478 | 0.00414798 | 0.0002880373 | 0.01254807 | 0.0002697714 | 0.0004117729 |
| 4/M5 | 0.002512111 | 0.002471993 | 0.1211209 | 0.1156276 | 0.1976852 | 0.00304919 | 0.0002872927 | 0.01036442 | 0.0001959299 | 0.0004856739 |

| budget/model | mean c3 | SD c3 | mean c6 | SD c6 | bound rate | continuation rate | unconverged rate | max KKT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8/M0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.348909e-13 |
| 8/M1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1.219253e-13 |
| 8/M2 | 1.091932 | 0.05911174 | 0 | 0 | 0 | 0 | 0 | 1.637282e-13 |
| 8/M3 | 0 | 0 | 0.9224049 | 0.9695721 | 0.85 | 0 | 0 | 1.720236e-10 |
| 8/M4 | 1.092762 | 0.0591438 | 1.061417 | 0.9816496 | 0.9 | 0 | 0 | 4.097496e-12 |
| 8/M5 | 1.056254 | 0 | 1.061871 | 2.27813e-16 | 0 | 0 | 0 | 4.752312e-14 |
| 4/M0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.136274e-13 |
| 4/M1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1.340324e-13 |
| 4/M2 | 0.9708351 | 0.1501039 | 0 | 0 | 0 | 0 | 0 | 1.605137e-13 |
| 4/M3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1.230795e-13 |
| 4/M4 | 0.9790576 | 0.1499653 | 0.9952469 | 1.021311 | 0.95 | 0 | 0 | 1.144812e-09 |
| 4/M5 | 1.056254 | 0 | 1.061871 | 2.27813e-16 | 0 | 0 | 0 | 7.726539e-14 |

## Held-out field-C noisy transfer

Errors below are relative fractions, not percentages. SD uses ddof=1. Rates count final bound contact, continuation, and failed final KKT respectively.

| budget/model | mean v err | median v err | mean ν err | median ν err | p90 ν err | SD v | SD ν | mean final L2 | clean NMSE | noisy loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8/M0 | 0.03835626 | 0.03864364 | 0.1322071 | 0.1323677 | 0.1535231 | 0.001800748 | 3.709591e-05 | 0.09045565 | 0.008308009 | 0.008521411 |
| 8/M1 | 0.002578283 | 0.002829791 | 0.0131048 | 0.01193753 | 0.02683508 | 0.001668966 | 3.297498e-05 | 0.01083265 | 0.0001072629 | 0.0002531902 |
| 8/M5 | 0.001458114 | 0.001312156 | 0.01258222 | 0.009729124 | 0.02628772 | 0.001663173 | 3.28115e-05 | 0.007646796 | 5.559991e-05 | 0.00019832 |

| budget/model | mean c3 | SD c3 | mean c6 | SD c6 | bound rate | continuation rate | unconverged rate | max KKT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8/M0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.070617e-13 |
| 8/M1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2.63023e-13 |
| 8/M5 | 1.056254 | 0 | 1.061871 | 2.27813e-16 | 0 | 0 | 0 | 6.914994e-13 |

## Physical error = clean model/design bias + noise-induced shift

The decomposition is in physical coordinates [v,ν4], not q. It is exact for every paired run; the full per-seed vectors and all endpoints are retained. Means below are signed, so they differ from mean absolute relative recovery errors above.

| truth/budget/model | mean physical error [v,ν] | clean bias [v,ν] | mean noise shift [v,ν] |
| --- | --- | --- | --- |
| matched_A/8/M0 | [-0.0003071578, 1.772532e-05] | [3.108624e-15, 1.75901e-15] | [-0.0003071578, 1.772532e-05] |
| matched_A/8/M1 | [-0.01497066, 3.926962e-05] | [-0.01469511, 2.586084e-05] | [-0.0002755445, 1.340878e-05] |
| matched_A/8/M4 | [-0.0006847726, -7.087262e-05] | [-9.992007e-16, 1.087672e-15] | [-0.0006847726, -7.087262e-05] |
| matched_A/8/M5 | [-0.01580764, 5.029718e-05] | [-0.01553365, 3.713029e-05] | [-0.0002739886, 1.316689e-05] |
| matched_A/4/M0 | [0.0003156661, -7.543994e-06] | [4.884981e-15, 1.851384e-15] | [0.0003156661, -7.543994e-06] |
| matched_A/4/M1 | [-0.01647021, 6.684498e-05] | [-0.01677636, 5.806301e-05] | [0.0003061495, 8.781977e-06] |
| matched_A/4/M4 | [-0.000121675, -0.0001157782] | [-2.277576e-06, -3.242763e-06] | [-0.0001193974, -0.0001125354] |
| matched_A/4/M5 | [-0.01739034, 7.444604e-05] | [-0.0176967, 6.484974e-05] | [0.0003063588, 9.596302e-06] |
| continuum_A/8/M0 | [0.01395103, 0.0003701883] | [0.01429244, 0.0003482887] | [-0.0003414117, 2.18996e-05] |
| continuum_A/8/M1 | [7.866964e-05, 2.530487e-05] | [0.0003764286, 9.357054e-06] | [-0.000297759, 1.594781e-05] |
| continuum_A/8/M2 | [-0.001313315, 0.000126038] | [-0.00079015, 0.0001104043] | [-0.0005231648, 1.563376e-05] |
| continuum_A/8/M3 | [0.01399321, 0.0002700643] | [0.01429244, 0.0003482887] | [-0.0002992299, -7.822442e-05] |
| continuum_A/8/M4 | [-0.0012851, 2.219561e-05] | [-0.0007543242, -1.405859e-05] | [-0.0005307753, 3.62542e-05] |
| continuum_A/8/M5 | [-0.0007444058, 1.815905e-05] | [-0.0004489359, 2.532043e-06] | [-0.0002954699, 1.5627e-05] |
| continuum_A/4/M0 | [0.01771118, 0.0002098208] | [0.01738862, 0.0002114704] | [0.0003225664, -1.649644e-06] |
| continuum_A/4/M1 | [0.00083156, 1.844796e-05] | [0.0005420545, 3.529382e-06] | [0.0002895055, 1.491858e-05] |
| continuum_A/4/M2 | [0.001281907, 0.0001198578] | [-0.0005836192, 0.0001064099] | [0.001865526, 1.344795e-05] |
| continuum_A/4/M3 | [0.01771118, 0.0002098208] | [0.01738862, 0.0002114704] | [0.0003225664, -1.649644e-06] |
| continuum_A/4/M4 | [0.001186255, 1.974972e-05] | [-0.0006577409, 2.684012e-05] | [0.001843996, -7.090396e-06] |
| continuum_A/4/M5 | [-0.0001095453, 1.280171e-05] | [-0.0003980874, -2.914773e-06] | [0.0002885421, 1.571649e-05] |
| continuum_C/8/M0 | [0.03835626, 0.0002644142] | [0.038411, 0.0002637406] | [-5.474934e-05, 6.735604e-07] |
| continuum_C/8/M1 | [0.002487694, 5.945768e-06] | [0.002549547, 6.802834e-06] | [-6.185271e-05, -8.570667e-07] |
| continuum_C/8/M5 | [0.0005145513, -4.595822e-07] | [0.0005771842, 4.401389e-07] | [-6.263298e-05, -8.997211e-07] |

For M4, discrepancy shifts are measured from its own clean reference:

| group | mean Δc3 | SD Δc3 | mean Δc6 | SD Δc6 |
| --- | --- | --- | --- | --- |
| matched_A/8/M4 | 0.02975614 | 0.04040551 | 0.9336379 | 0.9878541 |
| matched_A/4/M4 | 0.02889229 | 0.0532279 | 1.16711 | 1.005249 |
| continuum_A/8/M4 | 0.01536431 | 0.0591438 | -0.1735048 | 0.9816496 |
| continuum_A/4/M4 | -0.09195519 | 0.1499653 | 0.233691 | 1.021311 |

## Fixed theory, calibrated, and free discrepancy: interpretation

Fixed theory already removes most of the clean continuum bias: full-field ν4 error falls from 10.0608% (M0) to 0.3129% (M1). Known-physics calibration further reduces it to 0.00705% (M5), but that is not blind calibration of unknown physics. M5 improves clean recovery over M1 for every tested N32 transfer field/parameter pair, including C and B; it does not appear confined to field A on those controls. Under moderate continuum-A noise, M1 and M5 are nearly tied in mean ν4 error (4.4178% and 4.4075%), so the large clean calibration benefit does not translate into a comparably large noisy advantage. On held-out noisy C, ν4 error is 13.2207% for M0, 1.3105% for M1, and 1.2582% for M5. These are descriptive results from 20 paired seeds, not significance claims.

Cross-grid calibration multipliers decrease toward one: (1.2394,1.3380) at N16 to (1.0138,1.0149) at N64. Frozen N32 M5 remains much better than M0 on every tested grid, but M1 beats M5 at N48 and N64. Therefore calibrated correction transfers imperfectly across resolution; its finite-dx refinement is not a universal coefficient.

Phase-only M2 reduces clean v error to 0.0682% but leaves 5.2867% ν4 error. Damping-only M3 leaves 2.1847% v error and reaches c6=2 while reducing ν4 error to 0.1139%; that boundary compensation is not a physically meaningful calibration. M4 is complementary but not additive: it reaches ν4 error 0.1559%, slightly worse than this particular M3 endpoint, while removing most phase bias. The full-grid Hessian has weak cross phase/damping couplings, but strong v–c3 coupling 0.7934 and q–c6 coupling 0.9728.

M4 gives the smallest clean continuum field loss, 3.3822e−7, with a positive-definite but ill-conditioned Hessian (condition number 1.4373e7). All eight starts pass the declared KKT threshold, yet c6 spans about 0.0685 and ν4 spans 6.53e−6 at almost equal losses (3.3822e−7 to 3.3893e−7). This supports weak practical identifiability at the declared tolerance; it does not prove an exactly nonunique structural optimum. The conditional slices show local compensation, and the Hessian resolves positive curvature rather than an exact zero mode.

Under moderate continuum-A noise, M4 has lower noisy loss than M1/M5 but greater ν4 error (5.7549%), higher ν4 SD (1.3742e−4 versus about 1.126e−4), and greater v variance. Its c6 SD is 0.982 and 90% of fits hit a discrepancy bound. At the severe budget, M4 improves mean ν4 error to 10.9831% versus about 12.1% for M1/M5, yet worsens velocity error and final-field prediction relative to those fixed corrections; 95% hit bounds. Thus there is no universal winner, and free flexibility is not justified solely by residual reduction. For matched truth, free discrepancy also frequently reaches bounds (95%/100% at moderate/severe), while fixed nonzero discrepancy is deliberately misspecified and introduces clear velocity bias. Physical accuracy and field prediction remain distinct.

All 527 production fits pass the original stationarity threshold. The two needed continuations are full-field B transfer fits for M1 and M5, both resolved by the common 4000 cap. No noisy fit was extended, and no hyperparameter was changed after production started.

## Continuation and unresolved stationarity

Production comparisons, multistarts, grid transfers, and noisy fits: 527 total; 2 continued; 0 remain above 1e−7. Calibration and optimizer pilots are reported separately and not included in this count. Original 2000-update states, gradients, moments, losses, and final states are retained for every production fit; continuation changes and total counts are explicit. No failed run is removed from summaries.

| group | fits | continued | unconverged | max KKT |
| --- | --- | --- | --- | --- |
| matched_A/8/M0 | 20 | 0 | 0 | 2.663243e-13 |
| matched_A/8/M1 | 20 | 0 | 0 | 2.273781e-13 |
| matched_A/8/M4 | 20 | 0 | 0 | 2.179637e-13 |
| matched_A/8/M5 | 20 | 0 | 0 | 2.889032e-13 |
| matched_A/4/M0 | 20 | 0 | 0 | 1.718941e-13 |
| matched_A/4/M1 | 20 | 0 | 0 | 1.921204e-13 |
| matched_A/4/M4 | 20 | 0 | 0 | 3.642083e-13 |
| matched_A/4/M5 | 20 | 0 | 0 | 1.607975e-13 |
| continuum_A/8/M0 | 20 | 0 | 0 | 1.348909e-13 |
| continuum_A/8/M1 | 20 | 0 | 0 | 1.219253e-13 |
| continuum_A/8/M2 | 20 | 0 | 0 | 1.637282e-13 |
| continuum_A/8/M3 | 20 | 0 | 0 | 1.720236e-10 |
| continuum_A/8/M4 | 20 | 0 | 0 | 4.097496e-12 |
| continuum_A/8/M5 | 20 | 0 | 0 | 4.752312e-14 |
| continuum_A/4/M0 | 20 | 0 | 0 | 2.136274e-13 |
| continuum_A/4/M1 | 20 | 0 | 0 | 1.340324e-13 |
| continuum_A/4/M2 | 20 | 0 | 0 | 1.605137e-13 |
| continuum_A/4/M3 | 20 | 0 | 0 | 1.230795e-13 |
| continuum_A/4/M4 | 20 | 0 | 0 | 1.144812e-09 |
| continuum_A/4/M5 | 20 | 0 | 0 | 7.726539e-14 |
| continuum_C/8/M0 | 20 | 0 | 0 | 2.070617e-13 |
| continuum_C/8/M1 | 20 | 0 | 0 | 2.63023e-13 |
| continuum_C/8/M5 | 20 | 0 | 0 | 6.914994e-13 |

| continued fit | initial parameters | initial KKT | final parameters | final KKT | updates |
| --- | --- | --- | --- | --- | --- |
| continuum_B/full/M1 | [0.9028789, 0.001518059] | 0.002271564 | [0.9033002, 0.00151757, 1, 1] | 2.488059e-10 | 4000 |
| continuum_B/full/M5 | [0.8960749, 0.001502899] | 0.02946868 | [0.9015079, 0.001507914, 1.056254, 1.061871] | 1.565059e-09 | 4000 |

## Tests and acceptance

```text
........................................................................ [ 81%]
................                                                         [100%]
88 passed in 30.29s
```

| check | pass |
| --- | --- |
| all_tests_pass | yes |
| previous_behavior_preserved | yes |
| D3_D6_verified | yes |
| zero_correction_and_scan_equivalence | yes |
| correction_signs_verified | yes |
| whole_box_sampled_stability | yes |
| all_four_gradient_checks | yes |
| known_physics_calibration_separate | yes |
| matched_controls_and_M0_M5_comparison | yes |
| multistart_and_Hessian_reported | yes |
| frozen_designs | yes |
| exact_common_observations | yes |
| clean_bias_noise_decomposition | yes |
| heldout_transfer_without_recalibration | yes |
| uniform_hardening | yes |
| all_production_stationarity_resolved | yes |
| physical_and_field_errors_separate | yes |

**PASS**. PASS does not require c=1, perfect physical recovery, a well-conditioned Hessian, universal correction benefit, or perfect transfer. Unresolved numerical failures are distinguished from valid unfavorable scientific findings.

## Scientific caveats

The correction is only a low-order modified-equation basis. Its dimensionless multipliers are scheme-dependent, not physical constants, and calibration at one dx need not transfer. Joint discrepancy/physics inference can create structural or practical non-identifiability; Hessian conditioning depends on parameter coordinates and scaling. Continuum truth is idealized, initial conditions are exactly known, and noise is synthetic iid Gaussian. Lower state error does not imply more accurate physics. Fixed theory uses numerical-analysis knowledge unavailable for generic unknown model error. No arbitrary functional discrepancy, neural closure, or Bayesian uncertainty model is learned here.

## Recommended Stage 10 (not implemented)

Option A: profile-based uncertainty and practical identifiability. The resolved but very small q–c6 curvature, tolerance-dependent multistart spread, and frequent noisy discrepancy bounds make constrained profile-loss studies and explicit uncertainty calibration more informative next steps than adding a neural closure. Preserve fixed-theory correction as a reference and study coordinate/bound sensitivity without treating inverse Hessians as automatic covariances. IMEX integration is a secondary scalability direction because the stability policy grows from 1024 steps at N32 to 16384 at N64. Stage 10 is not implemented.

## Reproduction and checkpoints

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.stage9_model_discrepancy --forward-validation
.venv/bin/python -m experiments.stage9_model_discrepancy --calibrate
.venv/bin/python -m experiments.stage9_model_discrepancy --clean-inference
.venv/bin/python -m experiments.stage9_model_discrepancy --noisy-inference
.venv/bin/python -m experiments.stage9_model_discrepancy --finalize
```

Atomic JSON checkpoints retain completed groups; exceptions are recorded and raised. Resume validates noisy group counts and the finalizer requires all 23 groups and 44 clean references. Nonstationary endpoints are retained, not silently rerun or discarded. JSON serialization rejects NaN/Infinity. Git status: this directory is not a Git repository; no commit or push was attempted.
