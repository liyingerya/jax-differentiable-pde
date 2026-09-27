# Test matrix

Final full-suite result: **124 passed in 35.22s**. Parametrization makes collected cases differ from function counts. Test files are unchanged in Stage 12.

| Subsystem / file | Test functions | Actual named coverage |
| --- | --- | --- |
| [test_design.py](../tests/test_design.py) | 9 | jacobian shape and initial zero; sensitivity finite difference; information and singularity; temporal enumeration; layout reproducibility and budget; rank correlation ties; batch kernel matches public optimizer; batched pde fit matches individual; candidate pool scoring matches direct jacobian |
| [test_discrepancy.py](../tests/test_discrepancy.py) | 5 | derivatives; zero correction and fourier equivalence; gradients and projection; symbols; continuation state four dimensions |
| [test_identifiability.py](../tests/test_identifiability.py) | 8 | nll excludes deterministic initial data; fixer and profile optimum and active bound; quadratic intervals; coordinate hessian and profile invariance; noise pairing and bias decomposition; serialization rejects nonfinite; likelihood scaling preserves stage9 gradient units; paired record generation is once per seed |
| [test_integrators.py](../tests/test_integrators.py) | 18 | symbol stencil; zero exact; shape; semigroup; mean; diffusion; imaginary; parameter gradient; initial derivative; rk4 scan; convergence; startup shape and selected; cn power iterative; roots recurrence; split dissipation; original fd symbol; differs continuum; invalid steps |
| [test_inverse.py](../tests/test_inverse.py) | 6 | positive transform and roundtrip; observation shape and selection; loss scalar finite and truth minimum; inverse gradient; short recovery; optimizer bounds are enforced |
| [test_joint_inverse.py](../tests/test_joint_inverse.py) | 7 | unpack; prediction shape and loss; joint gradient; vector bounds; short joint fit; hessian finite and shape; nonpositive hessian is not given condition number |
| [test_observations.py](../tests/test_observations.py) | 8 | zero noise exact; noise reproducibility and initial state; noise scale uses positive observed entries; time and sensor shapes; full selection matches old loss; masked loss ignores unobserved entries; sensor layout reproducibility; noisy masked gradient and short fit |
| [test_operators.py](../tests/test_operators.py) | 4 | output shape; constant derivative; sine derivative; refinement decreases error |
| [test_refinement.py](../tests/test_refinement.py) | 6 | continuum shape and initial; timestep and observation alignment; bias metrics; pure diffusion has no phase translation; modified wavenumbers converge; two grid recovery smoke |
| [test_robust_design.py](../tests/test_robust_design.py) | 9 | continuum initial and shape; continuum derivatives; declared ensemble; noise information and relative efficiency; exhaustive indexing; pool score matches direct weighted subset; frozen hash and recovery independence; design specific bias identity; stage8 batched continuation preserves state for alternate initial |
| [test_solver.py](../tests/test_solver.py) | 11 | rhs shape; rhs constant; step shape; trajectory shape and initial; advection direction and amplitude; hyperdiffusion amplitude and energy; mean conservation; temporal convergence; stability diagnostic; diffusion gradient; initial state and velocity gradients |
| [test_stage7_hardening.py](../tests/test_stage7_hardening.py) | 3 | split adam matches uninterrupted history; policy uses original strict threshold and cap; outward boundary gradient does not trigger continuation |

Runtime scalability is validated by archived Stage 11 experiments, not asserted as a pytest benchmark winner. The quickstart and deterministic figure/document generators are executed separately. Numerical integrator tests cover values, gradients, recurrence, semigroup and convergence; they do not establish universal stability or hardware performance.
