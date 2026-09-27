"""Build final documentation and audit it against immutable archived evidence."""
import ast
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs'

def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def write(name,text): (ROOT/name).write_text(text.strip()+'\n')
def source(i):return f'[Stage {i}](stage{i}_validation.md) / [JSON](stage{i}_results.json)'

LIMITATIONS='''The model is one-dimensional, periodic, linear and constant coefficient. Observations are synthetic; no real experimental dataset validates physical inference. The initial condition is known exactly and noise uses simplified Gaussian models. The discrepancy basis is derived from this particular finite-difference scheme. M5 calibration uncertainty is not propagated. Exact exponential propagation relies on Fourier diagonalizability and does not establish generic nonlinear PDE scalability. Observation-design optimality is confined to finite candidate pools; robustness is only over the declared scenarios. Profile and coverage studies have finite Monte Carlo sample sizes, and nominal likelihood thresholds are not guaranteed finite-sample coverage. Discrepancy bounds affect uncertainty. Timings depend on hardware, JAX version, output requirements and compilation state.'''
FUTURE='''Prioritized extensions are: (1) a 2D PDE, (2) semilinear/nonlinear dynamics using IMEX, (3) uncertain initial-condition inference, (4) richer regularized discrepancy models, (5) neural-operator or surrogate hybridization, and (6) real observational data. None is implemented here.'''
ABSTRACT='''This project develops a validated JAX workflow for differentiable simulation and inverse modeling of a periodic advection–hyperdiffusion equation. Centered finite differences and differentiable Runge–Kutta integration first establish forward accuracy and matched-data coefficient recovery. Continuum-generated observations then expose a central limitation: a converged numerical inverse problem can identify an effective discretized coefficient rather than the physical parameter. Spatial refinement, noisy and sparse observations, and joint velocity–diffusion inference separate discretization bias from weak sensitivity and stochastic error. Sensitivity-aware observation design improves matched-model recovery but can deteriorate under model mismatch, motivating a robust design across a declared ensemble. Modified-equation discrepancy corrections remove much of the clean continuum bias, while freely fitted corrections create compensation directions and bound-dependent uncertainty. Profile likelihood and paired repeated-noise ensembles complement local Hessian diagnostics without assuming that quadratic uncertainty is always too narrow. Finally, exact semi-discrete Fourier propagation verifies that the earlier parameter-bias findings were not Runge–Kutta temporal artifacts. Differentiable Crank–Nicolson and IMEX CNAB2 are evaluated for stability, accuracy, gradients and computational cost. The resulting repository connects numerical verification to inference diagnostics and reproducible evidence. Its conclusions apply to the tested synthetic, one-dimensional linear setting; they do not establish general nonlinear scalability or validation against real measurements.'''

def main():
    preserved=json.loads((D/'stage12_preservation.json').read_text())
    changed=[p for p,h in preserved.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('Archive discrepancy: '+str(changed))
    r={i:json.loads((D/f'stage{i}_results.json').read_text()) for i in range(3,12)}
    tests=(D/'stage12_pytest.txt').read_text().strip().splitlines()[-1]
    if not re.match(r'124 passed in [\d.]+s',tests):raise RuntimeError('Tests incomplete or failed: '+tests)
    artifacts=json.loads((D/'stage12_artifact_audit.json').read_text())
    assert artifacts['figure_byte_reproducibility'] and artifacts['exported_stage11_resume'].startswith('PASS')
    assert len(artifacts['sha256'])==20
    for path,digest in artifacts['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    claims=[]
    def claim(label,i,field,value,expected,tol,display):
        ok=bool(np.allclose(value,expected,rtol=0,atol=tol))
        claims.append(dict(claim=label,stage=i,field=field,value=value,expected=expected,tolerance=tol,status='PASS' if ok else 'DISCREPANCY'))
        if not ok:raise RuntimeError('Scientific discrepancy: '+label)
        return [label,display,source(i)]
    rows=[]
    rows.append(claim('Matched scalar recovery',3,'config.nu4_true; matched.runs[0].recovered_nu4',[r[3]['config']['nu4_true'],r[3]['matched']['runs'][0]['recovered_nu4']],[.002,.002],1e-15,'ν₄ true/recovered = .002 / .002'))
    a={z['N']:z for z in r[4]['spatial_refinement']}
    rows.append(claim('Continuum refinement, known v',4,'spatial_refinement[N=32,48].relative_bias',[a[32]['relative_bias']*100,a[48]['relative_bias']*100],[18.215407,4.848956],5e-7,'ν₄ bias: +18.2154% (N32) → +4.8490% (N48)'))
    s=r[5]['studies']['combined_3']['summary']
    rows.append(claim('Severe scalar sparse/noisy failure',5,'studies.combined_3.summary.{mean_absolute_relative_error,mean_final_clean_field_relative_l2}',[100*s['mean_absolute_relative_error'],100*s['mean_final_clean_field_relative_l2']],[44.161615,2.507067],5e-7,'Mean ν₄ error 44.16%; field error 2.51%'))
    h=r[6]['matched_geometry']['hessian']['condition_number']
    rows.append(claim('Joint clean Hessian',6,'matched_geometry.hessian.condition_number',h,1903.848515,1e-6,'Condition number 1903.85 in (v, log ν₄); weak diffusion direction'))
    vals=[100*r[7]['hardening']['recovery'][f'matched_8_{m}']['by_design'][m]['mean_relative_nu4_error'] for m in ['baseline','joint_E']]
    rows.append(claim('Matched eight-sensor design',7,'hardening.recovery.matched_8_{baseline,joint_E}.by_design.*.mean_relative_nu4_error',vals,[11.946484,5.797916],5e-7,'Mean ν₄ error 11.95% → 5.80%; frozen held-out seeds'))
    for b,expected in [('8',[38.065163,51.075130,17.534305]),('4',[42.242573,66.654035,22.027958])]:
        vals=[100*r[8]['worst_case_recovery'][b][m]['mean_relative_nu4_error'] for m in ['baseline','joint_E','robust_noise_weighted']]
        rows.append(claim(f'Worst-family diffusion error, {b} sensors',8,f'worst_case_recovery.{b}.{{baseline,joint_E,robust_noise_weighted}}.mean_relative_nu4_error',vals,expected,5e-7,'Baseline / joint E / robust weighted: '+' / '.join(f'{v:.2f}%' for v in vals)))
    vals=[100*r[9]['clean'][f'continuum_A/full/{m}']['final']['relative_physical_errors'][1] for m in ['M0','M1','M5']]
    rows.append(claim('Joint clean continuum discrepancy',9,'clean.continuum_A/full/{M0,M1,M5}.final.relative_physical_errors[1]',vals,[10.060753,.312877,.007054],5e-7,'Absolute ν₄ error M0 / M1 / M5: '+ ' / '.join(f'{v:.4f}%' for v in vals)))
    g=r[9]['geometry'];rows.append(claim('Free discrepancy compensation',9,'geometry.condition_number; normalized_hessian_couplings[1][3]',[g['condition_number'],g['normalized_hessian_couplings'][1][3]],[14372656.445654,.972808986389],1e-6,'M4 Hessian condition 1.44e7; normalized q–c₆ coupling .972809'))
    vals=[r[10]['profiles'][f'continuum_A/full/{m}/1']['intervals']['95']['nu4']['total_width'] for m in ['M1','M4','M5']]
    rows.append(claim('Full-clean nominal 95% profile width',10,'profiles.continuum_A/full/{M1,M4,M5}/1.intervals.95.nu4.total_width',vals,[.0002509277055,.0004385414877,.0002501972041],1e-13,'M1 / M4 / M5: .000250928 / .000438541 / .000250197'))
    vals=[r[10]['bootstrap'][f'continuum_A/8/{m}']['summary']['parameters']['sample_sd'][1] for m in ['M1','M4','M5']]
    rows.append(claim('Moderate paired-noise diffusion SD',10,'bootstrap.continuum_A/8/{M1,M4,M5}.summary.parameters.sample_sd[1]',vals,[.00014567868665,.00017826785414,.00014514793271],1e-13,'M1 / M4 / M5: .000145679 / .000178268 / .000145148'))
    rows.append(claim('M4 discrepancy-bound contact',10,'bootstrap.continuum_A/8/M4.summary.discrepancy_bound_fraction',r[10]['bootstrap']['continuum_A/8/M4']['summary']['discrepancy_bound_fraction'],.86,1e-15,'86% of 100 moderate repeated-noise fits'))
    vals=list(r[11]['stability_scaling'].values());rows.append(claim('Timestep stability scaling',11,'stability_scaling.{rk4,cnab2}',vals,[4.000000002496,.230510924869],1e-10,'dtmax ∝ dxᵖ: RK4 p=4.00, CNAB2 p=.23'))
    shifts=[r[11]['inverse'][f'full/{m}/rk4']['rows'][0]['exact_shift'] for m in ['M0','M1','M5']]
    value=float(np.max(np.abs(shifts)));rows.append(claim('RK4 temporal contribution to clean fits',11,'inverse.full/{M0,M1,M5}/rk4.rows[0].exact_shift',value,2.26973995154367e-12,1e-20,'Largest absolute coordinate shift: 2.27e−12; spatial/model bias dominates'))
    claim('Hardened score/error association',7,'hardening.association_rank_correlation',r[7]['hardening']['association_rank_correlation'],-.7592880978865407,1e-15,'')
    claim('Severe joint relative errors',6,'studies.severe.summary.{mean_relative_v_error,mean_relative_nu4_error}',[r[6]['studies']['severe']['summary']['mean_relative_v_error'],r[6]['studies']['severe']['summary']['mean_relative_nu4_error']],[.005690794346168837,.4693455315015343],1e-15,'')
    bench=next(a for a in r[11]['benchmarks'] if a['key']=='32/exact/observations')
    claim('Exact warmed observation timing, N32',11,'benchmarks[key=32/exact/observations].median',bench['median'],.000044,1e-6,'')
    claim('Exponential semigroup precision',11,'symbols.exponential.semigroup[0].max_abs',r[11]['symbols']['exponential']['semigroup'][0]['max_abs'],4.440892098500626e-16,1e-25,'')
    results=table(['Result','Verified finding','Archive'],rows)
    write('docs/final_key_results.md','# Final key results\n\n'+results+'\n\nPercent errors are not pooled across incompatible protocols. Stage 4 fixes velocity; Stage 9 jointly fits velocity and diffusion. Stage 8 reports the maximum of truth-family mean errors, not the maximum seed error. Stage 10 full-clean profiles use hypothetical 2%-RMS measurement precision, not zero-noise confidence. All detailed definitions are in the cited archives.')
    themes=[('Spatial operators','Do the periodic stencils converge?','Validated derivative and consistency checks'),('Differentiable forward solver','Does RK4 match Fourier dynamics and derivatives?','Forward and gradient validation'),('Scalar inverse recovery','Does matched recovery imply physical recovery?','Matched truth recovered; continuum fit biased'),('Discretization bias','Does refinement reduce inverse bias?','Spatial refinement reduces effective ν₄ bias'),('Observation robustness','Can a good field fit hide coefficient failure?','Sparse noisy diffusion recovery is fragile'),('Joint inference','Which physical direction is weak?','Diffusion is much less constrained than velocity'),('Sensitivity design','Can measurements constrain the weak direction?','Matched recovery improves; transfer can deteriorate'),('Robust design','Can design withstand model mismatch?','Worst-family error reduced in declared scenarios'),('Model discrepancy','Can numerical-analysis corrections reduce bias?','Fixed corrections help; free corrections compensate'),('Practical uncertainty','What does local curvature miss?','Profiles expose bounds and nonlinear compensation'),('Scalable integration','Are conclusions time-discretization artifacts?','Exact semi-discrete propagation confirms spatial/model origin')]
    stage_rows=[]
    for i,(theme,q,result) in enumerate(themes,1):stage_rows.append([i,theme,q,result,f'[Report](stage{i}_validation.md)',f'[JSON](stage{i}_results.json)' if i>=3 else 'No JSON archived; tables in report'])
    write('docs/stage_index.md','# Stage index\n\nEach stage addresses a limitation exposed by the preceding work. Stages 1–2 archive their numeric tables in reports rather than separate result JSON files. Stage 7 summaries use its convergence-hardened results; original 2000-update records remain intact.\n\n'+table(['Stage','Theme','Main question','Key result','Report','Results'],stage_rows))
    figs=json.loads((D/'final_figure_data.json').read_text())
    gallery='\n\n'.join(f"![{p['filename']}](../figures/{p['filename']}.png)\n\n**Figure {i}.** {p['caption']} See [exact provenance](final_figure_provenance.md)." for i,p in enumerate(figs,1))
    report=f'''# Differentiable PDE Simulation, Inverse Modeling, and Robust Experimental Design with JAX

A validated end-to-end study of differentiable numerical simulation, physical parameter inference, discretization bias, model discrepancy, identifiability, observation design, uncertainty, and scalable integration.

## Abstract

{ABSTRACT}

## Motivation

Forward accuracy and optimizer convergence answer different questions from physical identifiability. This project starts with verified numerical differentiation, then follows failures that emerge when a differentiable solver is used to infer physics. It is a synthetic scientific-computing study, not production software or a neural PDE solver. The [stage index](stage_index.md) records the complete progression.

## Governing PDE

On x∈[0,2π) with periodic boundaries, ∂ₜψ = −v∂ₓψ − ν₄∂ₓ⁴ψ, with positive ν₄. The main initial field is sin(x)+0.5sin(2x)+0.25cos(3x). Velocity transports phase while hyperdiffusion damps short wavelengths. Later experiments use declared alternative fields and parameter points; they are not pooled with the nominal case.

## Numerical model

Centered periodic D₁ and D₄ stencils approximate the spatial derivatives. Stages [1](stage1_validation.md) and [2](stage2_validation.md) verify derivatives, Fourier-mode evolution and differentiability. The discrete symbols differ from continuum ik and k⁴. This difference remains even when time integration is exact.

## Differentiable simulation

JAX float64 computations differentiate through stencil/lax.scan RK4. Later direct Fourier propagation exploits the same linear periodic operator. Classical RK4-power evaluates its stability polynomial at integer step counts, while the Stage 11 exponential evolves the semi-discrete operator without time steps. These are separate numerical rules, not interchangeable definitions of exact continuum physics.

## Inverse parameter estimation

Synthetic observations are sampled in space and time. Earlier stages minimize normalized mean-square error; ν₄ is represented by q=log ν₄ and bounded projected Adam supplies reproducible updates. Later continuation policies preserve optimizer state and settings. Stage 10 uses positive-time Gaussian likelihood for profile inference while retaining the archived normalized stationarity criterion. Matched scalar data recover ν₄=.002 to floating-point precision, but this validates numerical consistency only. [Stage 3 report](stage3_validation.md).

## Numerical discretization bias

Continuum data fitted by a coarse FD model produce an effective coefficient. With velocity known, the N32 diffusion bias is +18.2154%, falling to +4.8490% at N48. The archived timestep controls leave this bias essentially unchanged. These scalar results must not be confused with Stage 9 joint velocity–diffusion estimates. A converged optimizer has accurately solved a biased numerical inverse problem. See [Stage 4](stage4_validation.md) and Figure 2.

## Observation robustness and identifiability

The severe scalar study has 44.16% mean diffusion error despite 2.51% mean clean-field error. Joint fitting adds a weak diffusion direction: the clean Hessian condition number is 1903.85 in (v,q). Under the severe joint protocol, mean velocity error is 0.569%, versus 46.935% for diffusion. Small state errors therefore do not establish accurate coefficient inference. These are finite synthetic ensembles, not universal error rates. See [Stage 5](stage5_validation.md), [Stage 6](stage6_validation.md), and Figure 3.

## Observation design

Scaled sensitivity scores guide finite-pool sensor/time selection. For eight sensors, matched mean diffusion error falls from 11.95% to 5.80%. Across frozen held-out layouts, the hardened score/error Spearman association is −.7593. One Stage 7 fit originally failed stationarity at 2000 updates; uniform same-state continuation resolved it without changing designs, seeds or conclusions. Better matched-model design did not guarantee transfer: the original continuum comparison worsened from about 14.38% to 32.47%. See [Stage 7](stage7_validation.md) and Figure 4.

## Robust design under model uncertainty

Stage 8 evaluates a declared multi-model, multi-field ensemble and freezes maximin designs before recovery. Worst truth-family mean diffusion error for eight sensors is 38.07% for baseline, 51.08% for Stage 7 joint E, and 17.53% for robust weighted design. Four-sensor values are 42.24%, 66.65%, and 22.03%. This reduces sampled cross-model failure rather than proving universal robustness or global continuous design optimality. See [Stage 8](stage8_validation.md) and Figure 5.

## Explicit model discrepancy

The corrected FD generator adds c₃v dx²D₃/6+c₆ν₄ dx²D₆/6. M0 has no correction; M1 fixes both coefficients to one; M2/M3 fit one correction; M4 fits both; M5 fixes coefficients from separate known-physics calibration. Joint full-clean continuum diffusion error falls from 10.0608% for M0 to .3129% for M1 and .0071% for M5. This is numerical-analysis-informed correction, not a neural model. M4 reduces residuals but its Hessian has condition number 1.44×10⁷ and normalized q–c₆ coupling .972809. More flexibility can exchange physical and discrepancy parameters. See [Stage 9](stage9_validation.md) and Figures 6–7.

## Practical uncertainty and profile inference

Stage 10 profiles nuisance coordinates, distinguishes physical truth from each model/design pseudo-truth, and uses paired repeated-noise ensembles. Full-clean nominal 95% diffusion profile widths are .000250928, .000438541 and .000250197 for M1/M4/M5. These use a hypothetical measurement-precision scale; they are not confidence statements for noiseless data. Moderate diffusion SDs are .000145679, .000178268 and .000145148. Fixed correction changes bias without a large conditional variance increase, but calibration uncertainty is excluded. M4 hits a discrepancy bound in 86% of moderate fits. The unconstrained Hessian can give wider intervals than constrained profiles; it is incorrect to claim quadratic uncertainty is always too small. Finite coverage samples and bound dependence limit certainty. See [Stage 10](stage10_validation.md) and Figures 7–8.

## Scalable time integration

Exact exponential propagation is preferred for this special linear periodic constant-coefficient system. It removes time error while retaining spatial/model bias. Stage 11 shifts clean Stage 9 parameters by at most 2.27×10⁻¹² when replacing RK4-power by the exact exponential. Sampled dtmax scaling is dx⁴ for RK4 and dx^0.23 for CNAB2 across the tested grids; no expected IMEX exponent was imposed. CN is A-stable, not L-stable. CNAB2 is a transferable semilinear stepping structure, with explicit phase restrictions and imperfect stiff damping. Timing comparisons retain the already-fast FFT RK4-power evaluator and distinguish compilation from synchronized warm calls. See [Stage 11](stage11_validation.md), [benchmark summary](benchmark_summary.md), and Figures 9–10.

## Main results

{results}

## Final figures

{gallery}

## Limitations

{LIMITATIONS}

## Lessons learned

Differentiability supplies derivatives; it does not establish identifiability. Accurate optimization can return biased effective physics. State prediction and parameter identification need separate diagnostics. Observation design can improve inference while overfitting an assumed forward model. Extra discrepancy flexibility can lower residuals while broadening uncertainty. Numerical-analysis priors can improve inference, but their calibration assumptions remain part of the uncertainty budget. Exact time integration cannot remove spatial or model bias.

## Reproduction

The [reproduction guide](reproduction.md) separates a quick demo/test/figure path from heavy historical studies. The final validation is **{tests}**. No expensive Stage 8 or Stage 10 study was rerun for synthesis. Figures use archived JSON only; [provenance](final_figure_provenance.md), [consistency audit](final_consistency_audit.md) and [test matrix](test_matrix.md) document checks. Historical files remain immutable.

## Future work

{FUTURE}

## Conclusion

The main contribution is a connected validation workflow: verified derivatives enable inverse fitting; biased inference motivates refinement and discrepancy; weak sensitivity motivates design; model mismatch motivates robust design; compensation motivates profile uncertainty; stiff integration motivates exact and IMEX propagation. Each conclusion remains conditional on the declared model, data and numerical protocol. [Release preparation](release_summary.md) uses the provisional label v1.0-ready, not a Git tag.
'''
    write('docs/final_project_report.md',report)
    readme=f'''# Differentiable PDE Simulation, Inverse Modeling, and Robust Experimental Design with JAX

A validated end-to-end study of differentiable numerical simulation, physical parameter inference, discretization bias, model discrepancy, identifiability, observation design, uncertainty, and scalable integration. Eleven connected scientific stages investigate a synthetic periodic advection–hyperdiffusion problem; the final synthesis makes their evidence reproducible.

## Why this matters

A differentiable solver can converge to the wrong physical parameters. This project separates optimizer accuracy, numerical bias, observation quality and identifiability rather than judging inference only by field fit.

## Model and architecture

**∂ₜψ = −v∂ₓψ − ν₄∂ₓ⁴ψ**, on periodic x∈[0,2π), with ν₄>0 and a known initial condition.

![Architecture](figures/01_architecture.png)

## Key findings

- Matched synthetic data recover the diffusion coefficient to numerical precision.
- Known-velocity continuum diffusion bias falls from **18.22% at N32 to 4.85% at N48** under refinement.
- Severe scalar observations yield **44.16% parameter error despite 2.51% field error**.
- Robust weighted design reduces worst-family mean diffusion error from **38.07% to 17.53%** for the eight-sensor protocol.
- Fixed discrepancy corrections reduce bias; free corrections introduce compensation and bound-dependent uncertainty.
- Exact semi-discrete propagation confirms that earlier bias was spatial/model driven, rather than an RK4 time artifact.

All values, definitions and archived sources appear in the [key-results table](docs/final_key_results.md) and [technical report](docs/final_project_report.md). These are protocol-specific findings, not universal performance claims.

## Repository structure

```text
src/          validated numerical and inference components
experiments/  historical stage drivers and final artifact generators
examples/     small runnable demonstration
tests/        numerical, derivative and inference validation
docs/         immutable stage archives and final synthesis
figures/      ten final figures, each in SVG and PNG
```

## Quick start

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m examples.quickstart
```

The example creates a periodic field, propagates the semi-discrete model exactly, samples matched observations, and evaluates an NMSE gradient. It does not launch an inverse study.

## Reproduce core validations and figures

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.final_figures
```

Final test result: **{tests}**. [Test matrix](docs/test_matrix.md).
See the [reproduction guide](docs/reproduction.md) before running historical drivers: some overwrite outputs, and Stages 8/10 are heavy. [Stage index](docs/stage_index.md).

## Final figures

[Figure gallery and captions](docs/final_project_report.md) · [Exact data provenance](docs/final_figure_provenance.md) · [SVG/PNG files](figures/)

## Limitations

This is a 1D linear periodic constant-coefficient model with synthetic Gaussian observations and a known initial condition. Exact Fourier propagation is a special-case method. Design and robustness claims are limited to declared pools/scenarios; profile coverage has finite sample sizes, and M5 calibration uncertainty is excluded. No real dataset or generic nonlinear scalability is established. [Full limitations](docs/final_project_report.md).

## Documentation

[Final report](docs/final_project_report.md) · [Benchmarks](docs/benchmark_summary.md) · [Consistency audit](docs/final_consistency_audit.md) · [Interview summary](docs/interview_project_summary.md) · [Release preparation](docs/release_summary.md)

Status: **v1.0-ready**, a provisional documentation label. No Git release or tag has been created.
'''
    write('README.md',readme)
    benchmark(r[11]);test_matrix(tests);reproduction();interview()
    audit_lines=['# Final scientific consistency audit',f'All {len(preserved)} archived source/test/experiment/document files match their pre-synthesis SHA-256 hashes in [preservation manifest](stage12_preservation.json). No scientific computation was rerun for these claims.',table(['Claim','Exact JSON field','Verified value','Tolerance','Status'],[[a['claim'],f"[Stage {a['stage']}](stage{a['stage']}_results.json): `{a['field']}`",str(a['value']),a['tolerance'],a['status']] for a in claims]),'Values were checked against the rounded historical report values before authoring the synthesis. Report comparisons use the indicated absolute rounding tolerances. No unresolved numerical discrepancy was found. Stage 4 scalar and Stage 9 joint biases are intentionally distinct; Stage 7 hardened values are used; Stage 8 worst-case means are not seed maxima; Stage 10 hypothetical precision is explicit.']
    write('docs/final_consistency_audit.md','\n\n'.join(audit_lines))
    write('docs/release_summary.md',f'''# Release preparation: v1.0-ready

Provisional label only; no repository initialization, commit, push, release or Git tag.

Scope: the completed 1D periodic JAX simulation-to-inference study, Stages 1–11, with a final synthesis in Stage 12. Validated capabilities include FD stencils, differentiable RK4/exponential/CN/CNAB2 propagation, physical inverse estimation, observation design, robust design, explicit discrepancy and profile/repeated-noise uncertainty.

Final tests: **{tests}**. Scientific preservation: **{len(preserved)} files unchanged**. Ten figures have exact data provenance. See [acceptance and link audit](stage12_validation.md), [technical report](final_project_report.md), and [reproduction](reproduction.md).

## Limitations and non-goals

{LIMITATIONS}

No new physics, inverse model, integrator, observation algorithm or scientific experiment was introduced. This is research software preparation, not a production deployment. No PDF, slides, publication submission or external release is claimed.

## Prioritized extensions

{FUTURE}
''')
    # Audit all final Markdown links, including figures; exclude external URLs.
    finaldocs=['README.md']+[str(p.relative_to(ROOT)) for p in D.glob('*.md') if p.name.startswith('final_') or p.name in ['stage_index.md','reproduction.md','test_matrix.md','benchmark_summary.md','interview_project_summary.md','release_summary.md']]
    links=[]
    for name in finaldocs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(ROOT/name).read_text()):
            if '://' in target or target.startswith('#'):continue
            path=target.split('#')[0]
            if path=='stage12_validation.md':continue # written below, then rechecked
            ok=(ROOT/name).parent.joinpath(path).exists();links.append(dict(file=name,target=target,exists=ok))
    if any(not a['exists'] for a in links):raise RuntimeError('Broken links '+str([a for a in links if not a['exists']]))
    acceptance=['All Stage 1–11 tests pass','Archived scientific files byte-preserved','Final technical report exists','Concise final README exists','Ten figures exist with exact provenance','Reproduction guide exists','Stage index exists','Test matrix exists','Benchmark summary exists','Interview summary exists','Scientific consistency audit passes','Final documentation links resolve','Cache/venv/OS artifacts excluded from release manifest','Limitations explicit','No new core physics or scientific study']
    write('docs/stage12_validation.md','# Stage 12: Final synthesis and release preparation\n\n**PASS — 15/15 checks.**\n\n'+table(['Acceptance','Status'],[[x,'PASS'] for x in acceptance])+f'\n\nFinal pytest: **{tests}**. {len(preserved)} archived files match their baseline hashes. {len(claims)} quantitative checks pass. {len(links)} local documentation/figure links resolve. Ten final figures are available in both SVG and PNG. The quickstart runs successfully; no heavy production study was rerun. All 20 SVG/PNG files are byte-identical across two rendering runs. The exported Stage 11 inventory passes its original resume guard. See [artifact audit](stage12_artifact_audit.json).\n\nThe release manifest lists intended files only, excluding the existing local virtual environment, caches and OS metadata. These local runtime directories are ignored, not packaged. Git status: not a Git repository; no initialization, commit, push or tagging.\n\nSee [audit](final_consistency_audit.md), [release summary](release_summary.md), and [provenance](final_figure_provenance.md).')
    final_links=[]
    for target in finaldocs+['docs/stage12_validation.md']:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(ROOT/target).read_text()):
            if '://' in link or link.startswith('#'):continue
            exists=(ROOT/target).parent.joinpath(link.split('#')[0]).exists()
            assert exists,(target,link)
            final_links.append(dict(file=target,target=link,exists=exists))
    validation=(D/'stage12_validation.md').read_text().replace(f'{len(links)} local documentation/figure links resolve',f'{len(final_links)} local documentation/figure links resolve')
    write('docs/stage12_validation.md',validation)
    links=final_links
    audit=dict(artifact_audit='stage12_artifact_audit.json',status='PASS',archive_files=len(preserved),changed_archives=changed,claims=claims,links=links,pytest=tests,acceptance={x:True for x in acceptance})
    write('docs/stage12_audit.json',json.dumps(audit,indent=2))
    files=sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and not any(part.startswith('.') and part not in ['.gitignore'] for part in p.relative_to(ROOT).parts) and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'])
    write('docs/release_manifest.txt','\n'.join(sorted(set(files+['docs/release_manifest.txt']))))
    print(f'PASS: {len(claims)} claims; {len(links)} links; {len(preserved)} archived files; {tests}')


def benchmark(r):
    rows=[]
    for a in r['benchmarks']:
        if a['status']=='measured' and a['task'] in ['observations','gradient'] and a['N']<=256:rows.append([a['N'],a['method'],a['task'],f"{a['first_call_seconds']*1000:.3f}",f"{a['median']*1000:.3f}",a['steps']])
    write('docs/benchmark_summary.md','# Time-integration engineering summary\n\nSource: [Stage 11 report](stage11_validation.md) and [JSON](stage11_results.json), `benchmarks`, `stability_scaling`, `storage`, `accuracy_cost`.\n\nEnvironment: '+str(r['environment'])+'\n\n'+table(['N','Method','Task','First call ms','Warm median ms','Schedule steps'],rows)+'''

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
''')


def test_matrix(summary):
    rows=[]
    for p in sorted((ROOT/'tests').glob('test_*.py')):
        names=[n.name.removeprefix('test_').replace('_',' ') for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
        rows.append([f'[{p.name}](../tests/{p.name})',len(names),'; '.join(names)])
    write('docs/test_matrix.md',f'# Test matrix\n\nFinal full-suite result: **{summary}**. Parametrization makes collected cases differ from function counts. Test files are unchanged in Stage 12.\n\n'+table(['Subsystem / file','Test functions','Actual named coverage'],rows)+'''\n\nRuntime scalability is validated by archived Stage 11 experiments, not asserted as a pytest benchmark winner. The quickstart and deterministic figure/document generators are executed separately. Numerical integrator tests cover values, gradients, recurrence, semigroup and convergence; they do not establish universal stability or hardware performance.''')


def reproduction():
    versions={k:importlib.metadata.version(k) for k in ['jax','jaxlib','numpy','pytest','matplotlib']}
    modules=['stage1_derivative_validation','stage2_time_integration_validation','stage3_parameter_inference','stage4_refinement_study','stage5_observation_robustness','stage6_joint_inference']
    write('docs/reproduction.md',f'''# Reproduction guide

## Validated environment

Python {sys.version.split()[0]}; packages: {versions}. CPU/macOS hardware details for archived timing are in [Stage 11](stage11_results.json), `environment`. This run uses the existing CPU environment. `requirements.txt` retains the project policy of unpinned direct dependencies: JAX, pytest, matplotlib. NumPy is supplied by JAX. No pandas, seaborn or additional ML framework was added. Exact future-version reproducibility is not promised; the versions here identify the validated environment.

## Quick reproduction — no heavy studies

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m examples.quickstart
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.final_figures
```

The quickstart evaluates a loss and gradient, not optimization. Its warmed execution is below a few seconds on the validated machine. Tests take about 36 seconds here; figure rendering takes seconds after font-cache setup. Figure data come only from archived JSON. To regenerate documentation/audits, capture pytest to `docs/stage12_pytest.txt`, then run `python -m experiments.final_synthesis`. The synthesis script checks the pre-existing `docs/stage12_preservation.json`; do not replace that baseline to hide a mismatch.

## Full reproduction — separate disposable workspace

Historical experiment drivers can overwrite their own outputs. Preserve the original repository and run them only in a separate copy. Stage 11's archive guard compares the entire earlier file inventory; adding final documents makes that old inventory guard intentionally reject the synthesis tree. Use `python -m experiments.reproduction_workspace --destination /tmp/jax-pde-stage11-reproduction` to export the exact archived Stage 11 file set, without venv/caches/final documents. Install requirements in that destination. This exporter copies files and performs no scientific work.

Existing checkpoints normally resume rather than rerun completed groups. To recompute a stage, use another disposable copy and explicitly move that stage's output/checkpoint aside after reading its driver. Keep dependency-stage archives intact. If upstream records are regenerated, strict downstream byte-hash guards can reject them even when numeric differences are only timing/roundoff; investigate and compare values rather than disabling a guard. This release does not claim a one-command fresh run of all heavy archives.

Historical commands, run from a suitable disposable workspace:

```sh
'''+ '\n'.join('python -m experiments.'+m for m in modules)+'''
python -m experiments.stage7_observation_design --design-only
python -m experiments.stage7_observation_design --recover-frozen
python -m experiments.stage7_observation_design --harden-existing
python -m experiments.stage8_robust_design --design-only
python -m experiments.stage8_robust_design --recover-frozen
```

Stage 9 phases:

```sh
for phase in forward-validation calibrate clean-inference noisy-inference finalize; do
  python -m experiments.stage9_model_discrepancy --"$phase" || break
done
```

Stage 10 phases (HEAVY):

```sh
for phase in clean-profiles profile-surfaces bootstrap-moderate bootstrap-severe heldout-transfer matched-controls representative-profiles finalize; do
  python -m experiments.stage10_identifiability --"$phase" || break
done
```

Stage 11 phases:

```sh
python -m pytest -q > docs/stage11_pytest.txt
for phase in symbols temporal stability gradients inverse benchmark accuracy finalize; do
  python -m experiments.stage11_scalable_integration --phase "$phase" || break
done
```

## Computational burden

Stages 1–2 are small validations; Stages 3–7 involve repeated differentiable solves and can take minutes or more depending on hardware. Stage 8 is **heavy**: 14056 candidate combinations per budget, scenario sensitivities and repeated recovery. Stage 9 has 527 archived production fits plus diagnostics. Stage 10 is **very heavy**: 48336 archived optimizer attempts including profile grids and repeated-noise studies; reserve a long checkpointed run rather than expecting a quick demo. These counts describe archived work, not newly measured wall-clock estimates. Stage 11 skips excessive fine-grid scan workloads by declared caps. Consult each [stage report](stage_index.md) before execution. Stage 12 reruns none of these production studies.

## Artifact policy

Figures are generated deterministically from saved data with fixed styles and SVG metadata. Same-environment byte reproducibility is audited; fonts/library changes can change rendering. `docs/release_manifest.txt` lists intended deliverables and excludes virtual environments, Python/pytest/JAX caches and OS metadata. Do not package the local `.venv`.
''')


def interview():
    write('docs/interview_project_summary.md','''# Interview and portfolio summary

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
''')

if __name__=='__main__':main()
