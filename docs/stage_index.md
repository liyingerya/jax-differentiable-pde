# Stage index

Each stage addresses a limitation exposed by the preceding work. Stages 1–2 archive their numeric tables in reports rather than separate result JSON files. Stage 7 summaries use its convergence-hardened results; original 2000-update records remain intact.

| Stage | Theme | Main question | Key result | Report | Results |
| --- | --- | --- | --- | --- | --- |
| 1 | Spatial operators | Do the periodic stencils converge? | Validated derivative and consistency checks | [Report](stage1_validation.md) | No JSON archived; tables in report |
| 2 | Differentiable forward solver | Does RK4 match Fourier dynamics and derivatives? | Forward and gradient validation | [Report](stage2_validation.md) | No JSON archived; tables in report |
| 3 | Scalar inverse recovery | Does matched recovery imply physical recovery? | Matched truth recovered; continuum fit biased | [Report](stage3_validation.md) | [JSON](stage3_results.json) |
| 4 | Discretization bias | Does refinement reduce inverse bias? | Spatial refinement reduces effective ν₄ bias | [Report](stage4_validation.md) | [JSON](stage4_results.json) |
| 5 | Observation robustness | Can a good field fit hide coefficient failure? | Sparse noisy diffusion recovery is fragile | [Report](stage5_validation.md) | [JSON](stage5_results.json) |
| 6 | Joint inference | Which physical direction is weak? | Diffusion is much less constrained than velocity | [Report](stage6_validation.md) | [JSON](stage6_results.json) |
| 7 | Sensitivity design | Can measurements constrain the weak direction? | Matched recovery improves; transfer can deteriorate | [Report](stage7_validation.md) | [JSON](stage7_results.json) |
| 8 | Robust design | Can design withstand model mismatch? | Worst-family error reduced in declared scenarios | [Report](stage8_validation.md) | [JSON](stage8_results.json) |
| 9 | Model discrepancy | Can numerical-analysis corrections reduce bias? | Fixed corrections help; free corrections compensate | [Report](stage9_validation.md) | [JSON](stage9_results.json) |
| 10 | Practical uncertainty | What does local curvature miss? | Profiles expose bounds and nonlinear compensation | [Report](stage10_validation.md) | [JSON](stage10_results.json) |
| 11 | Scalable integration | Are conclusions time-discretization artifacts? | Exact semi-discrete propagation confirms spatial/model origin | [Report](stage11_validation.md) | [JSON](stage11_results.json) |
