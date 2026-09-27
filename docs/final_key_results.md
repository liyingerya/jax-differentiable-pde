# Final key results

| Result | Verified finding | Archive |
| --- | --- | --- |
| Matched scalar recovery | ν₄ true/recovered = .002 / .002 | [Stage 3](stage3_validation.md) / [JSON](stage3_results.json) |
| Continuum refinement, known v | ν₄ bias: +18.2154% (N32) → +4.8490% (N48) | [Stage 4](stage4_validation.md) / [JSON](stage4_results.json) |
| Severe scalar sparse/noisy failure | Mean ν₄ error 44.16%; field error 2.51% | [Stage 5](stage5_validation.md) / [JSON](stage5_results.json) |
| Joint clean Hessian | Condition number 1903.85 in (v, log ν₄); weak diffusion direction | [Stage 6](stage6_validation.md) / [JSON](stage6_results.json) |
| Matched eight-sensor design | Mean ν₄ error 11.95% → 5.80%; frozen held-out seeds | [Stage 7](stage7_validation.md) / [JSON](stage7_results.json) |
| Worst-family diffusion error, 8 sensors | Baseline / joint E / robust weighted: 38.07% / 51.08% / 17.53% | [Stage 8](stage8_validation.md) / [JSON](stage8_results.json) |
| Worst-family diffusion error, 4 sensors | Baseline / joint E / robust weighted: 42.24% / 66.65% / 22.03% | [Stage 8](stage8_validation.md) / [JSON](stage8_results.json) |
| Joint clean continuum discrepancy | Absolute ν₄ error M0 / M1 / M5: 10.0608% / 0.3129% / 0.0071% | [Stage 9](stage9_validation.md) / [JSON](stage9_results.json) |
| Free discrepancy compensation | M4 Hessian condition 1.44e7; normalized q–c₆ coupling .972809 | [Stage 9](stage9_validation.md) / [JSON](stage9_results.json) |
| Full-clean nominal 95% profile width | M1 / M4 / M5: .000250928 / .000438541 / .000250197 | [Stage 10](stage10_validation.md) / [JSON](stage10_results.json) |
| Moderate paired-noise diffusion SD | M1 / M4 / M5: .000145679 / .000178268 / .000145148 | [Stage 10](stage10_validation.md) / [JSON](stage10_results.json) |
| M4 discrepancy-bound contact | 86% of 100 moderate repeated-noise fits | [Stage 10](stage10_validation.md) / [JSON](stage10_results.json) |
| Timestep stability scaling | dtmax ∝ dxᵖ: RK4 p=4.00, CNAB2 p=.23 | [Stage 11](stage11_validation.md) / [JSON](stage11_results.json) |
| RK4 temporal contribution to clean fits | Largest absolute coordinate shift: 2.27e−12; spatial/model bias dominates | [Stage 11](stage11_validation.md) / [JSON](stage11_results.json) |

Percent errors are not pooled across incompatible protocols. Stage 4 fixes velocity; Stage 9 jointly fits velocity and diffusion. Stage 8 reports the maximum of truth-family mean errors, not the maximum seed error. Stage 10 full-clean profiles use hypothetical 2%-RMS measurement precision, not zero-noise confidence. All detailed definitions are in the cited archives.
