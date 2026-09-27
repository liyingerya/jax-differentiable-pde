# Final scientific consistency audit

All 63 archived source/test/experiment/document files match their pre-synthesis SHA-256 hashes in [preservation manifest](stage12_preservation.json). No scientific computation was rerun for these claims.

| Claim | Exact JSON field | Verified value | Tolerance | Status |
| --- | --- | --- | --- | --- |
| Matched scalar recovery | [Stage 3](stage3_results.json): `config.nu4_true; matched.runs[0].recovered_nu4` | [0.002, 0.0019999999999999953] | 1e-15 | PASS |
| Continuum refinement, known v | [Stage 4](stage4_results.json): `spatial_refinement[N=32,48].relative_bias` | [18.21540715560934, 4.848956025350644] | 5e-07 | PASS |
| Severe scalar sparse/noisy failure | [Stage 5](stage5_results.json): `studies.combined_3.summary.{mean_absolute_relative_error,mean_final_clean_field_relative_l2}` | [44.161614964084706, 2.5070673036742264] | 5e-07 | PASS |
| Joint clean Hessian | [Stage 6](stage6_results.json): `matched_geometry.hessian.condition_number` | 1903.8485153080633 | 1e-06 | PASS |
| Matched eight-sensor design | [Stage 7](stage7_results.json): `hardening.recovery.matched_8_{baseline,joint_E}.by_design.*.mean_relative_nu4_error` | [11.94648403342283, 5.797915698649437] | 5e-07 | PASS |
| Worst-family diffusion error, 8 sensors | [Stage 8](stage8_results.json): `worst_case_recovery.8.{baseline,joint_E,robust_noise_weighted}.mean_relative_nu4_error` | [38.06516294373509, 51.07512977378541, 17.53430512580198] | 5e-07 | PASS |
| Worst-family diffusion error, 4 sensors | [Stage 8](stage8_results.json): `worst_case_recovery.4.{baseline,joint_E,robust_noise_weighted}.mean_relative_nu4_error` | [42.24257294717586, 66.65403488007414, 22.027957922603566] | 5e-07 | PASS |
| Joint clean continuum discrepancy | [Stage 9](stage9_results.json): `clean.continuum_A/full/{M0,M1,M5}.final.relative_physical_errors[1]` | [10.06075316775947, 0.312876588019176, 0.007053685093633653] | 5e-07 | PASS |
| Free discrepancy compensation | [Stage 9](stage9_results.json): `geometry.condition_number; normalized_hessian_couplings[1][3]` | [14372656.445654092, 0.9728089863890997] | 1e-06 | PASS |
| Full-clean nominal 95% profile width | [Stage 10](stage10_results.json): `profiles.continuum_A/full/{M1,M4,M5}/1.intervals.95.nu4.total_width` | [0.00025092770551819895, 0.00043854148774345107, 0.0002501972041103638] | 1e-13 | PASS |
| Moderate paired-noise diffusion SD | [Stage 10](stage10_results.json): `bootstrap.continuum_A/8/{M1,M4,M5}.summary.parameters.sample_sd[1]` | [0.0001456786866483964, 0.00017826785414235783, 0.00014514793270987975] | 1e-13 | PASS |
| M4 discrepancy-bound contact | [Stage 10](stage10_results.json): `bootstrap.continuum_A/8/M4.summary.discrepancy_bound_fraction` | 0.86 | 1e-15 | PASS |
| Timestep stability scaling | [Stage 11](stage11_results.json): `stability_scaling.{rk4,cnab2}` | [4.0000000024961615, 0.23051092486911215] | 1e-10 | PASS |
| RK4 temporal contribution to clean fits | [Stage 11](stage11_results.json): `inverse.full/{M0,M1,M5}/rk4.rows[0].exact_shift` | 2.26973995154367e-12 | 1e-20 | PASS |
| Hardened score/error association | [Stage 7](stage7_results.json): `hardening.association_rank_correlation` | -0.7592880978865407 | 1e-15 | PASS |
| Severe joint relative errors | [Stage 6](stage6_results.json): `studies.severe.summary.{mean_relative_v_error,mean_relative_nu4_error}` | [0.005690794346168837, 0.4693455315015343] | 1e-15 | PASS |
| Exact warmed observation timing, N32 | [Stage 11](stage11_results.json): `benchmarks[key=32/exact/observations].median` | 4.4312968384474516e-05 | 1e-06 | PASS |
| Exponential semigroup precision | [Stage 11](stage11_results.json): `symbols.exponential.semigroup[0].max_abs` | 4.440892098500626e-16 | 1e-25 | PASS |

Values were checked against the rounded historical report values before authoring the synthesis. Report comparisons use the indicated absolute rounding tolerances. No unresolved numerical discrepancy was found. Stage 4 scalar and Stage 9 joint biases are intentionally distinct; Stage 7 hardened values are used; Stage 8 worst-case means are not seed maxima; Stage 10 hypothetical precision is explicit.
