# Release preparation: v1.0-ready

Provisional label only; no repository initialization, commit, push, release or Git tag.

Scope: the completed 1D periodic JAX simulation-to-inference study, Stages 1–11, with a final synthesis in Stage 12. Validated capabilities include FD stencils, differentiable RK4/exponential/CN/CNAB2 propagation, physical inverse estimation, observation design, robust design, explicit discrepancy and profile/repeated-noise uncertainty.

Final tests: **124 passed in 35.22s**. Scientific preservation: **63 files unchanged**. Ten figures have exact data provenance. See [acceptance and link audit](stage12_validation.md), [technical report](final_project_report.md), and [reproduction](reproduction.md).

## Limitations and non-goals

The model is one-dimensional, periodic, linear and constant coefficient. Observations are synthetic; no real experimental dataset validates physical inference. The initial condition is known exactly and noise uses simplified Gaussian models. The discrepancy basis is derived from this particular finite-difference scheme. M5 calibration uncertainty is not propagated. Exact exponential propagation relies on Fourier diagonalizability and does not establish generic nonlinear PDE scalability. Observation-design optimality is confined to finite candidate pools; robustness is only over the declared scenarios. Profile and coverage studies have finite Monte Carlo sample sizes, and nominal likelihood thresholds are not guaranteed finite-sample coverage. Discrepancy bounds affect uncertainty. Timings depend on hardware, JAX version, output requirements and compilation state.

No new physics, inverse model, integrator, observation algorithm or scientific experiment was introduced. This is research software preparation, not a production deployment. No PDF, slides, publication submission or external release is claimed.

## Prioritized extensions

Prioritized extensions are: (1) a 2D PDE, (2) semilinear/nonlinear dynamics using IMEX, (3) uncertain initial-condition inference, (4) richer regularized discrepancy models, (5) neural-operator or surrogate hybridization, and (6) real observational data. None is implemented here.
