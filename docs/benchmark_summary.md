# Time-integration engineering summary

Source: [Stage 11 report](stage11_validation.md) and [JSON](stage11_results.json), `benchmarks`, `stability_scaling`, `storage`, `accuracy_cost`.

Environment: {'platform': 'macOS-14.7.4-arm64-arm-64bit', 'jax': '0.6.2', 'devices': ['TFRT_CPU_0']}

| N | Method | Task | First call ms | Warm median ms | Schedule steps |
| --- | --- | --- | --- | --- | --- |
| 32 | scan_rk4 | observations | 194.664 | 4.780 | 1024 |
| 32 | scan_rk4 | gradient | 1486.208 | 55.866 | 1024 |
| 32 | rk4 | observations | 45.004 | 0.052 | 1024 |
| 32 | rk4 | gradient | 109.103 | 0.076 | 1024 |
| 32 | exact | observations | 35.962 | 0.044 | 0 |
| 32 | exact | gradient | 66.870 | 0.064 | 0 |
| 32 | cn | observations | 80.509 | 0.096 | 512 |
| 32 | cn | gradient | 139.651 | 0.604 | 512 |
| 32 | cnab2 | observations | 107.463 | 0.211 | 512 |
| 32 | cnab2 | gradient | 189.106 | 1.168 | 512 |
| 64 | scan_rk4 | observations | 325.342 | 123.055 | 16384 |
| 64 | scan_rk4 | gradient | 2652.991 | 1182.775 | 16384 |
| 64 | rk4 | observations | 44.633 | 0.074 | 16384 |
| 64 | rk4 | gradient | 119.613 | 0.120 | 16384 |
| 64 | exact | observations | 36.901 | 0.057 | 0 |
| 64 | exact | gradient | 75.717 | 0.068 | 0 |
| 64 | cn | observations | 52.430 | 0.335 | 1024 |
| 64 | cn | gradient | 174.078 | 1.955 | 1024 |
| 64 | cnab2 | observations | 63.784 | 0.721 | 1024 |
| 64 | cnab2 | gradient | 221.580 | 3.657 | 1024 |
| 128 | rk4 | observations | 45.842 | 0.110 | 262144 |
| 128 | rk4 | gradient | 125.339 | 0.190 | 262144 |
| 128 | exact | observations | 37.416 | 0.075 | 0 |
| 128 | exact | gradient | 81.041 | 0.106 | 0 |
| 128 | cn | observations | 50.503 | 1.208 | 2048 |
| 128 | cn | gradient | 149.671 | 6.507 | 2048 |
| 128 | cnab2 | observations | 63.777 | 2.732 | 2048 |
| 128 | cnab2 | gradient | 204.394 | 12.149 | 2048 |
| 256 | rk4 | observations | 47.648 | 0.181 | 4194304 |
| 256 | rk4 | gradient | 122.379 | 0.330 | 4194304 |
| 256 | exact | observations | 36.832 | 0.122 | 0 |
| 256 | exact | gradient | 78.514 | 0.122 | 0 |
| 256 | cn | observations | 54.878 | 4.269 | 4096 |
| 256 | cn | gradient | 169.014 | 25.015 | 4096 |
| 256 | cnab2 | observations | 73.116 | 10.295 | 4096 |
| 256 | cnab2 | gradient | 291.234 | 50.233 | 4096 |

First call includes JIT compilation and execution. Warm medians use ten synchronized calls; raw repetitions and min/max are archived. Nine-observation output shapes and four-coordinate gradient objectives are comparable; schedules differ in accuracy. CN here is iterative, although an equivalent direct-power evaluator is used in inverse fitting. No universal speedup is claimed.

RK4 dtmax scales as dx⁴; CNAB2 measured dx^0.23051 over N16–128. This is dense numerical sampling over the declared box, not a continuous stability proof. The stability advantage is resolution-dependent and CNAB2 retains an explicit phase restriction. CN is A-stable but not L-stable.

Fine-grid scan RK4 schedules of 262144 (N128) and 4194304 (N256) steps are estimated, not measured timings. They exceed the predeclared cap. Exact final/observation/gradient tasks were measured through N1024. No resource failure was observed; intentional workload omissions remain labeled.

## Storage and use cases

- Scan RK4: a real state plus stage temporaries; the archived API allocates the full step trajectory. Reverse-mode can retain history.
- FFT RK4-power: initial spectrum and requested-time output workspace; no step scan, but RK4 stability/time error remain.
- Exact exponential: similar requested-time storage, no internal steps or temporal truncation error. Preferred for this linear constant-coefficient periodic FD system.
- CN: actual iteration or direct powers; stable stiff modes may alternate and decay too slowly.
- CNAB2: two Fourier states plus requested outputs; reverse-mode can store/reconstruct history. Transferable to future semilinear work, subject to stability and accuracy checks.

Output sizes are N, 9N, or (steps+1)N for final, observations, or full trajectory. Precise peak device memory was not measured. See Stage 11 for fixed-target accuracy-at-cost tables; full-trajectory comparisons have differing output counts and are not used for runtime exponent claims.
