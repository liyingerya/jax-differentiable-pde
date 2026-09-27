# Final figure provenance

Generated with `python -m experiments.final_figures`. Only archived files are read. SVG and PNG versions share the same data. Deterministic metadata and a fixed SVG hash salt are used.

## 01_architecture

Source stage(s): 1–11. Source: [Stage 1 report](stage1_validation.md), [Stage 11 JSON](stage11_results.json)

Exact fields: `Source modules and archived report progression`.

Transformation: Conceptual schematic; no numerical data.

Outputs: [SVG](../figures/01_architecture.svg), [PNG](../figures/01_architecture.png).

Architecture: validated operators feed differentiable time integration and inverse loss; design, discrepancy and uncertainty are connected analyses.

## 02_refinement

Source stage(s): 4. Source: [Stage 4 JSON](stage4_results.json)

Exact fields: `spatial_refinement[].{N,relative_bias}`.

Transformation: relative_bias × 100.

Outputs: [SVG](../figures/02_refinement.svg), [PNG](../figures/02_refinement.png).

Known velocity, continuum A, zero noise; each archived fit uses its declared stability-safe RK4 schedule.

## 03_parameter_vs_field

Source stage(s): 5,6. Source: [Stage 5 JSON](stage5_results.json), [Stage 6 JSON](stage6_results.json)

Exact fields: `Stage5 studies.combined_3.run_ids → runs[].{final_clean_field_relative_l2,absolute_relative_error}; Stage6 studies.severe.run_ids → runs[].{final_clean_field_relative_l2,relative_nu4_error}`.

Transformation: Both errors × 100; all five seeds per severe study.

Outputs: [SVG](../figures/03_parameter_vs_field.svg), [PNG](../figures/03_parameter_vs_field.png).

Severe matched sparse/noisy conditions; field and coefficient errors are distinct metrics.

## 04_design_association

Source stage(s): 7. Source: [Stage 7 JSON](stage7_results.json)

Exact fields: `hardening.association_pairs[].{tier,score,median_nu4_error}; hardening.association_rank_correlation`.

Transformation: Error × 100; use hardened results; no reselection or trend refit.

Outputs: [SVG](../figures/04_design_association.svg), [PNG](../figures/04_design_association.png).

Thirty frozen top/middle/bottom layouts; negative rank association is descriptive, not causal proof.

## 05_robust_design

Source stage(s): 8. Source: [Stage 8 JSON](stage8_results.json)

Exact fields: `worst_case_recovery.{8,4}.{baseline,joint_E,robust_unweighted,robust_noise_weighted}.mean_relative_nu4_error`.

Transformation: Stored worst truth-family mean × 100; not maximum individual-seed error.

Outputs: [SVG](../figures/05_robust_design.svg), [PNG](../figures/05_robust_design.png).

Eight sensors use 2% noise; four use 5%. Robustness is limited to declared truth families and finite candidate pools.

## 06_discrepancy_bias

Source stage(s): 9. Source: [Stage 9 JSON](stage9_results.json)

Exact fields: `clean.continuum_A/full/M0–M5.final.parameters[0:2]`.

Transformation: 100 × (estimate / physical truth − 1); truth=(1,.002).

Outputs: [SVG](../figures/06_discrepancy_bias.svg), [PNG](../figures/06_discrepancy_bias.png).

Full clean continuum A observations. M0 uncorrected; M1 theory; M2/M3 one free correction; M4 both free; M5 fixed calibrated correction.

## 07_profile_valley

Source stage(s): 10. Source: [Stage 10 JSON](stage10_results.json)

Exact fields: `surface.{q_axis,c6_axis,NLL,reference_NLL,valid,valley}`.

Transformation: 2*(NLL-reference_NLL); transpose q×c6 grid for plotting; mask invalid points.

Outputs: [SVG](../figures/07_profile_valley.svg), [PNG](../figures/07_profile_valley.png).

Full-clean M4 surface profiled over v,c₃, using the archived hypothetical 2%-RMS precision. Finite local domain; contour levels are diagnostics, not joint coverage claims.

## 08_uncertainty

Source stage(s): 10. Source: [Stage 10 JSON](stage10_results.json)

Exact fields: `bootstrap.continuum_A/8/{M1,M4,M5}.runs[].parameters[1]; profiles.continuum_A/full/{M1,M4,M5}/1.intervals.95.nu4.components`.

Transformation: ν₄ × 1000; boxplot median/IQR, whiskers 1.5 IQR; full-clean profile intervals separate panel.

Outputs: [SVG](../figures/08_uncertainty.svg), [PNG](../figures/08_uncertainty.png).

Left: eight sensors, 2% noise, 100 paired seeds; right: full-clean hypothetical precision, not zero-noise confidence. Different observation protocols must not be conflated. M4 has 86% discrepancy-bound contact in the left ensemble.

## 09_stability

Source stage(s): 11. Source: [Stage 11 JSON](stage11_results.json)

Exact fields: `stability[].{N,rk4_dtmax,imex_dtmax}; stability_scaling`.

Transformation: Log–log axes; no new stability computation.

Outputs: [SVG](../figures/09_stability.svg), [PNG](../figures/09_stability.png).

625 parameter tuples and ≥2049 angles per grid; fitted dt-vs-dx exponents 4.00 and 0.23. Sampling is not proof over the continuous box.

## 10_runtime

Source stage(s): 11. Source: [Stage 11 JSON](stage11_results.json)

Exact fields: `benchmarks filtered status=measured, task∈{observations,gradient}, N≤256: {N,method,median}`.

Transformation: seconds × 1000; separate tasks; no compile times or estimated points.

Outputs: [SVG](../figures/10_runtime.svg), [PNG](../figures/10_runtime.png).

Ten synchronized warmed repetitions. CN is iterative; RK4-power and exact use direct Fourier propagation. Schedules differ in accuracy. Missing fine-grid scan measurements exceed the predeclared cap; lines do not extrapolate.
