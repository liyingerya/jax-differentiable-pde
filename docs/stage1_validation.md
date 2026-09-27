# Stage 1 validation report

## Conclusion: PASS

Periodic first- and fourth-derivative operators agree with the analytic
derivatives of sin(3x). Errors decrease at every requested refinement and
measured convergence orders approach 2. Only Stage 1 is implemented.

## Files created

- `src/__init__.py`: package description.
- `src/operators.py`: reusable JAX periodic finite differences, float64 enabled.
- `experiments/stage1_derivative_validation.py`: error and convergence table.
- `tests/test_operators.py`: eight parametrized numerical test cases.
- `requirements.txt`: JAX and pytest only.
- `README.md`: PDE motivation, scope, installation and run commands.
- `.gitignore`: virtual environment and Python/pytest caches.
- `docs/stage1_validation.md`: this report.

A local `.venv` was created for execution. The directory was initially empty;
no existing files were overwritten and no Git history was modified.

## Formulas

For uniform spacing dx and periodic indices:

    d_dx(f)[i] = (f[i+1] - f[i-1]) / (2 dx)
    d4_dx4(f)[i] = (f[i-2] - 4 f[i-1] + 6 f[i]
                    - 4 f[i+1] + f[i+2]) / dx^4

`jnp.roll(f, -1)[i]` supplies f[i+1]; positive shifts supply backward
neighbors. The fourth derivative has a positive sign for sin(3x).
The eventual PDE applies the separate negative coefficient -ν4 for damping.

## Measured errors and orders

Grid: `jnp.linspace(0.0, 2*jnp.pi, N, endpoint=False)`, dx = 2π/N.
Input: sin(3x). Exact derivatives: 3 cos(3x) and 81 sin(3x).
Errors are maximum absolute errors over all grid points, including the seam.
Each order is log(error at N/2 / error at N) / log(2).

| N | dx | d1 max error | d4 max error | p(d1) | p(d4) |
|---:|---:|---:|---:|---:|---:|
| 32 | 1.963495e-01 | 1.70504038e-01 | 4.56419023e+00 | — | — |
| 64 | 9.817477e-02 | 4.31845575e-02 | 1.16346266e+00 | 1.98122 | 1.97193 |
| 128 | 4.908739e-02 | 1.08313201e-02 | 2.92287220e-01 | 1.99531 | 1.99297 |
| 256 | 2.454369e-02 | 2.71003308e-03 | 7.31609779e-02 | 1.99883 | 1.99824 |
| 512 | 1.227185e-02 | 6.77646027e-04 | 1.82958289e-02 | 1.99971 | 1.99956 |

## Execution and pytest

Executed with Python 3.10, JAX/jaxlib 0.6.2 and pytest 9.1.1, with JAX
float64 enabled:

```text
.venv/bin/python -m experiments.stage1_derivative_validation
# Completed successfully; results above.

.venv/bin/python -m pytest -q
........                                                                 [100%]
8 passed in 7.86s
```

Tests cover output shape, constant derivatives, analytic agreement at N=256,
and decreasing errors over all five grids, for each operator. Analytic error
tolerances are 0.003 for d1 and 0.1 for d4, allowing second-order truncation
error at N=256. The non-integer constant 1.3 uses an absolute tolerance of
1e-8 to accommodate fourth-stencil cancellation roundoff.

## Numerical interpretation and caveats

Both errors fall by approximately a factor of four per doubling. Measured
orders, rather than an assumed order, support second-order convergence.
Agreement with both analytic derivatives across the periodic seam gives no
evidence of indexing or sign errors.

The fourth stencil subtracts nearly equal terms and divides by dx^4,
amplifying floating-point roundoff as the grid becomes finer. At N=512,
p(d4)=1.99956 shows no obvious roundoff-driven deterioration; this does not
imply roundoff is absent or that refinement will help indefinitely. Results
at substantially finer grids may plateau or worsen. No tolerance was loosened
in response to a failure. These functions assume a 1D array and positive,
uniform grid spacing.

## Git status and next step

`git status --short` reports:

```text
fatal: not a git repository (or any of the parent directories): .git
```

No Git repository was initialized and no commits or pushes were made.

Recommended next step: design and validate a periodic explicit time integrator,
including its advection/hyperdiffusion stability restriction and Fourier-mode
damping checks. Stage 2 has not been implemented.
