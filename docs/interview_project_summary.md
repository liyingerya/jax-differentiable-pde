# Interview and portfolio summary

## 30-second explanation

This is a JAX study of how numerical solvers affect physical parameter inference. It begins with a verified periodic PDE solver and follows what breaks when observations become sparse, noisy or inconsistent with the numerical model. The central result is that good optimization and good field predictions can still give poor physics. The project then investigates sensor design, numerical discrepancy, uncertainty and faster integration.

## 90-second explanation

The equation combines advection with fourth-order diffusion. Automatic differentiation makes it straightforward to fit velocity and diffusion through the solver, but derivative correctness is only the first check. Matched synthetic data recover the planted parameters; continuum observations expose finite-grid bias. Refinement reduces that bias, while sparse observations reveal weak diffusion sensitivity. Sensitivity-based designs improve matched recovery but can overfit the assumed model, motivating robust design across scenarios. Fixed modified-equation corrections improve physical inference; fitting those corrections freely produces compensation and bound-dependent uncertainty. Profiles and repeated-noise ensembles make that tradeoff visible. Finally, exact semi-discrete Fourier propagation shows that the earlier bias was spatial/model driven, not an RK4 timestep artifact.

## Three-minute technical explanation

The forward discretization uses centered periodic first- and fourth-derivative stencils and classical RK4 in JAX float64. Fourier-mode and derivative checks establish the baseline. Inference uses a log diffusion coordinate and projected Adam with explicit bounds and stationarity checks. The first conceptual failure is model consistency: a finite-difference solver fitting continuum data can converge to an effective coefficient. The scalar N32 bias is about 18.22%, and refinement reduces it.

The next failure is information. Diffusion primarily affects damped higher modes, so sparse/noisy measurements can leave it weakly constrained even when velocity and field predictions look good. Scaled sensitivities guide sensor/time choices. Their matched-model success does not guarantee transfer; robust design uses a declared ensemble and held-out tests rather than changing layouts after seeing recovery results.

Discrepancy corrections use the finite-difference modified equation. Fixing theoretically or separately calibrated coefficients reduces bias while keeping only two physical coordinates free. Free correction adds flexibility but creates a q–c6 compensation valley. Profile likelihood, repeated-noise recovery and bound sensitivity are therefore needed alongside Hessians. Physical truth and model/design pseudo-truth are reported separately.

The last engineering issue is stiffness. RK4 stability scales with dx to the fourth power. The archived FFT RK4-power shortcut already avoids step scanning for this linear model; exact exponential propagation also removes temporal truncation error. CNAB2 adds a differentiable stepping method that can transfer to semilinear extensions. Benchmarks distinguish first-call compilation, warmed synchronized execution and estimated omitted workloads. All conclusions remain limited to this synthetic linear periodic setting.

## Problem statement and why JAX

The problem is to recover physical coefficients reliably when discretization and observation choices alter the inverse problem. JAX supplies composable automatic differentiation, JIT compilation and scan/vmap transformations. It does not supply identifiability or eliminate numerical bias.

## Hardest numerical issue

Hyperdiffusion makes explicit timesteps shrink rapidly with grid refinement. Stability schedules must remain safe across the search box, not just at the final fitted coefficient. The exact Fourier reference is possible only because the system is linear, periodic and constant coefficient.

## Hardest inverse issue

Distinguishing physical bias, noise-induced deviation and discrepancy compensation. A low residual or small projected gradient certifies neither correct physics nor narrow uncertainty.

## Biggest surprise

In the severe scalar study, mean diffusion error is 44.16% while final-field error is only 2.51%. Another important qualification is that constrained profile intervals need not be wider than an unconstrained inverse-Hessian approximation.

## What failed and what was learned

A matched-model observation design transferred poorly to continuum data. Free discrepancy created weak and boundary-concentrated directions. One Stage 7 fit needed uniform same-state continuation, retained transparently alongside its historical result. Stage 10 retained unsuccessful optimizer attempts and selected converged profile points. These failures motivated the next stage instead of being removed from the record.

## How the project evolved

Operator validation → differentiable simulation → scalar inverse fitting → discretization bias → observation robustness → joint inference → design → robust design → discrepancy → uncertainty → scalable integration. See the [stage index](stage_index.md).

## Likely questions and concise answers

- **Why not only use continuum Fourier evolution?** The inverse study deliberately examines the same FD model; continuum propagation would change the model and hide discretization bias.
- **Does exact propagation solve the inference problem?** It removes temporal error, not spatial bias, noise, discrepancy or weak identifiability.
- **Is robust design globally optimal?** Only within the finite candidate pool and declared scenario ensemble.
- **Why are bounds important?** They limit compensation between physical and discrepancy parameters and can materially change profile intervals.
- **Are nominal intervals calibrated?** Coverage studies are finite and separate physical from pseudo-truth; precise universal coverage is not established.
- **Is M5 uncertainty complete?** No. Calibration is fixed, so reported inference uncertainty is conditional on it.
- **Why keep RK4?** The stencil solver is an independent generic validation baseline; Fourier powers are a special shortcut.
- **What would come next?** A carefully verified 2D or semilinear extension, then uncertain initial conditions and real data.

## Possible resume bullets

- Developed a JAX differentiable PDE and inverse-modeling study with **124 passing numerical tests**, covering spatial operators, time integration, gradients and inference diagnostics.
- Quantified finite-grid inverse bias: known-velocity continuum diffusion bias decreased from **18.22% at N32 to 4.85% at N48** under spatial refinement.
- Evaluated robust observation design that reduced **worst truth-family mean diffusion error from 38.07% to 17.53%** in the declared eight-sensor synthetic protocol.
- Investigated numerical discrepancy and practical identifiability using profile likelihood and paired repeated-noise ensembles, exposing physical/discrepancy compensation and bound dependence.
- Implemented differentiable exact semi-discrete, CN and CNAB2 propagation and benchmarked synchronized forward/gradient execution, including exact propagation through **N1024**.

All quantitative statements are linked to their protocol and archived evidence in the [key results](final_key_results.md) and [consistency audit](final_consistency_audit.md). These are suggested descriptions of the repository; use authorship language appropriate to the author’s actual contribution.
