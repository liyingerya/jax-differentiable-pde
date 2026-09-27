"""Render Stage 9's complete numerical record without selecting favorable runs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.robust_design import frozen_hash

ROOT=Path(__file__).resolve().parents[1]
INTERPRETATION='Fixed theory already removes most of the clean continuum bias: full-field ν4 error falls from 10.0608% (M0) to 0.3129% (M1). Known-physics calibration further reduces it to 0.00705% (M5), but that is not blind calibration of unknown physics. M5 improves clean recovery over M1 for every tested N32 transfer field/parameter pair, including C and B; it does not appear confined to field A on those controls. Under moderate continuum-A noise, M1 and M5 are nearly tied in mean ν4 error (4.4178% and 4.4075%), so the large clean calibration benefit does not translate into a comparably large noisy advantage. On held-out noisy C, ν4 error is 13.2207% for M0, 1.3105% for M1, and 1.2582% for M5. These are descriptive results from 20 paired seeds, not significance claims.\n\nCross-grid calibration multipliers decrease toward one: (1.2394,1.3380) at N16 to (1.0138,1.0149) at N64. Frozen N32 M5 remains much better than M0 on every tested grid, but M1 beats M5 at N48 and N64. Therefore calibrated correction transfers imperfectly across resolution; its finite-dx refinement is not a universal coefficient.\n\nPhase-only M2 reduces clean v error to 0.0682% but leaves 5.2867% ν4 error. Damping-only M3 leaves 2.1847% v error and reaches c6=2 while reducing ν4 error to 0.1139%; that boundary compensation is not a physically meaningful calibration. M4 is complementary but not additive: it reaches ν4 error 0.1559%, slightly worse than this particular M3 endpoint, while removing most phase bias. The full-grid Hessian has weak cross phase/damping couplings, but strong v–c3 coupling 0.7934 and q–c6 coupling 0.9728.\n\nM4 gives the smallest clean continuum field loss, 3.3822e−7, with a positive-definite but ill-conditioned Hessian (condition number 1.4373e7). All eight starts pass the declared KKT threshold, yet c6 spans about 0.0685 and ν4 spans 6.53e−6 at almost equal losses (3.3822e−7 to 3.3893e−7). This supports weak practical identifiability at the declared tolerance; it does not prove an exactly nonunique structural optimum. The conditional slices show local compensation, and the Hessian resolves positive curvature rather than an exact zero mode.\n\nUnder moderate continuum-A noise, M4 has lower noisy loss than M1/M5 but greater ν4 error (5.7549%), higher ν4 SD (1.3742e−4 versus about 1.126e−4), and greater v variance. Its c6 SD is 0.982 and 90% of fits hit a discrepancy bound. At the severe budget, M4 improves mean ν4 error to 10.9831% versus about 12.1% for M1/M5, yet worsens velocity error and final-field prediction relative to those fixed corrections; 95% hit bounds. Thus there is no universal winner, and free flexibility is not justified solely by residual reduction. For matched truth, free discrepancy also frequently reaches bounds (95%/100% at moderate/severe), while fixed nonzero discrepancy is deliberately misspecified and introduces clear velocity bias. Physical accuracy and field prediction remain distinct.\n\nAll 527 production fits pass the original stationarity threshold. The two needed continuations are full-field B transfer fits for M1 and M5, both resolved by the common 4000 cap. No noisy fit was extended, and no hyperparameter was changed after production started.'
RECOMMENDATION='Option A: profile-based uncertainty and practical identifiability. The resolved but very small q–c6 curvature, tolerance-dependent multistart spread, and frequent noisy discrepancy bounds make constrained profile-loss studies and explicit uncertainty calibration more informative next steps than adding a neural closure. Preserve fixed-theory correction as a reference and study coordinate/bound sensitivity without treating inverse Hessians as automatic covariances. IMEX integration is a secondary scalability direction because the stability policy grows from 1024 steps at N32 to 16384 at N64. Stage 10 is not implemented.'


def fmt(v):
    if isinstance(v,(list,tuple,np.ndarray)):return '['+', '.join(fmt(x) for x in v)+']'
    if v is None:return 'not resolved'
    if isinstance(v,(bool,np.bool_)):return 'yes' if v else 'no'
    if isinstance(v,(int,np.integer)):return str(v)
    if isinstance(v,(float,np.floating)):return f'{v:.7g}'
    return str(v)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])


def finalize(r):
    if len(r.get('noisy',{}))!=23:raise RuntimeError('Expected all 23 noisy groups; no failed group may be skipped')
    if len(r.get('clean',{}))!=44:raise RuntimeError('Expected all 44 clean references/comparisons')
    if 'validation' not in r:
        import subprocess
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q'],cwd=ROOT,text=True,capture_output=True)
        r['validation']={'passed':run.returncode==0,'summary':run.stdout+run.stderr}
    r.setdefault('scientific_interpretation', INTERPRETATION)
    r.setdefault('stage10_recommendation', RECOMMENDATION)
    r['ablations']={m:r['clean']['continuum_A/full/'+m]['final'] for m in ['M0','M2','M3','M4']}
    r['transfer_experiments']={t:{m:r['clean'][t+'/full/'+m]['final'] for m in ['M0','M1','M5']} for t in ['continuum_A','continuum_C','continuum_offnominal','continuum_B']}
    old=json.loads((ROOT/'docs/stage8_results.json').read_text())
    designs={b:old['frozen']['selected'][b]['robust_noise_weighted'] for b in ['8','4']}
    frozen=frozen_hash(designs)==r['protocol']['design_hash']
    before=json.loads((ROOT/'docs/stage9_preservation.json').read_text())
    changed=[p for p,digest in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest]
    operators=(ROOT/'src/operators.py').read_text().split('\n\ndef d3_dx3')[0]
    joint=(ROOT/'src/joint_inverse.py').read_text().replace('jnp.zeros_like(theta), jnp.zeros_like(theta), 0','jnp.zeros(2), jnp.zeros(2), 0')
    exact_old=(hashlib.sha256(operators.encode()).hexdigest()==before['src/operators.py']
               and hashlib.sha256(joint.encode()).hexdigest()==before['src/joint_inverse.py'])
    r['preservation_audit']={'changed_previous_files':changed,'old_operators_and_optimizer_preserved':exact_old,
      'allowed_changes_only':set(changed)<=set(['src/operators.py','src/joint_inverse.py']) and exact_old}
    fits=[(k,v) for k,v in r['clean'].items()]+[(f'M4_start_{i}',v) for i,v in enumerate(r['M4_multistart'])]
    fits += [(f'grid{n}/{m}',v) for n,models in r['cross_grid_transfer'].items() for m,v in models.items()]
    fits += [(f'{k}/seed{a["seed"]}',a) for k,g in r['noisy'].items() for a in g['runs']]
    failed=[(k,a) for k,a in fits if a['final']['kkt']>=1e-7]
    continued=[(k,a) for k,a in fits if a['continuation_required']]
    r['optimizer_summary']={'production_fits':len(fits),'continued':len(continued),'unconverged':len(failed),
       'max_kkt':max(a['final']['kkt'] for _,a in fits),'unconverged_ids':[k for k,_ in failed]}
    paired=all(a['observation_hash']==next(x['hash'] for x in r['datasets']['/'.join(k.split('/')[:2])] if x['seed']==a['seed']) for k,g in r['noisy'].items() for a in g['runs'])
    decomp=all(np.allclose(np.array(a['decomposition']['physical_error']),np.array(a['decomposition']['clean_model_design_bias'])+a['decomposition']['noise_induced_shift'],rtol=0,atol=1e-15) for g in r['noisy'].values() for a in g['runs'])
    checks={'all_tests_pass':r['validation']['passed'],'previous_behavior_preserved':r['preservation_audit']['allowed_changes_only'],
      'D3_D6_verified':all(r['forward']['derivative_convergence'][0][f]>r['forward']['derivative_convergence'][-1][f] for f in ['D3_relative_error','D6_relative_error']),
      'zero_correction_and_scan_equivalence':r['forward']['scan_spectral_max_error']<1e-10,
      'correction_signs_verified':all(v['phase_error']>3.8 and v['damping_error']>3.8 for k,v in r['forward']['orders'].items() if k.endswith('c1')),
      'whole_box_sampled_stability':all(p['safety_fraction']<=.5 for p in r['forward']['stability'].values()),
      'all_four_gradient_checks':all(max(v['relative_error'])<1e-5 for v in r['forward']['gradient_checks'].values()),
      'known_physics_calibration_separate':r['calibration']['32']['common_optimum_supported'],
      'matched_controls_and_M0_M5_comparison':all(f'{t}/full/{m}' in r['clean'] for t in ['matched_A','continuum_A'] for m in ['M0','M1','M2','M3','M4','M5']),
      'multistart_and_Hessian_reported':len(r['M4_multistart'])>=8 and 'geometry' in r,
      'frozen_designs':frozen,'exact_common_observations':paired,'clean_bias_noise_decomposition':decomp,
      'heldout_transfer_without_recalibration':all(f'continuum_C/8/{m}' in r['noisy'] for m in ['M0','M1','M5']),
      'uniform_hardening':all(a['initial_update_budget']==2000 and a['final_total_updates'] in [2000,a['cap']] and a['continuation_required']==(a['initial']['kkt']>=1e-7) for _,a in fits),
      'all_production_stationarity_resolved':not failed,'physical_and_field_errors_separate':all('relative_physical_errors' in a['final'] and 'clean_observed_nmse' in a['final'] for _,a in fits)}
    r['checks']=checks;r['conclusion']='PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    out=[];add=out.append
    add('# Stage 9: Explicit Model Discrepancy and Physics-Parameter Separation')
    add(f'**{r["conclusion"]}**. All results are synthetic. Calibration uses known physical coefficients; inverse-physics fits do not. No Stage 10 implementation, commits, or pushes.')
    add('## Objective and preservation')
    add('Stage 8 improved worst-case recovery with robust observation design, but retained continuum-versus-discrete clean parameter bias. Stage 9 holds its masks and times fixed and asks whether explicit numerical discrepancy separates that bias from physical coefficients. New files: `src/discrepancy.py`, `experiments/stage9_model_discrepancy.py`, `experiments/stage9_report.py`, `tests/test_discrepancy.py`, and this report/results/preservation record. D3/D6 are appended to operators; the only old optimizer change is `zeros(2)` to `zeros_like(theta)` for moments. The Stage 1–8 numerical reports/results are byte-preserved. README is updated separately.')
    add('## Modified equation, stencils, and Fourier symbols')
    add('For smooth periodic fields, D1=∂x+(dx²/6)∂x³+O(dx⁴) and D4=∂x⁴+(dx²/6)∂x⁶+O(dx⁴). Thus the original negative advection and damping terms have negative leading defects. Adding the positive corrections below cancels them at c3=c6=1:')
    add('```text\nD3 f_i = (f_(i+2)-2 f_(i+1)+2 f_(i-1)-f_(i-2))/(2 dx³)\nD6 f_i = (f_(i-3)-6 f_(i-2)+15 f_(i-1)-20 f_i\n          +15 f_(i+1)-6 f_(i+2)+f_(i+3))/dx⁶\nRHS = -v D1 ψ - ν4 D4 ψ + c3 v dx² D3 ψ/6 + c6 ν4 dx² D6 ψ/6\n```')
    add('D3 sin(kx)=−k³ cos(kx), D6 sin(kx)=−k⁶ sin(kx). c3 and c6 are dimensionless discrepancy multipliers tied to the finite-difference scheme, not material coefficients. D3 controls phase and D6 damping. Substituting exp(i iθ) into each stencil yields:')
    add(table(['Operator','Symbol'],r['forward']['symbols'].items()))
    add('In particular, sin(2θ)−2sinθ=−4sinθ sin²(θ/2) gives D3, while the cube of −4sin²(θ/2) gives D6. Their small-angle limits are −ik³ and −k⁶. The corrected eigenvalue is the identical linear combination of these symbols; continuum λ=−ivk−ν4 k⁴. Classical RK4 uses R(z)=1+z+z²/2+z³/6+z⁴/24.')
    add('The public corrected trajectory uses the requested stencils, classical RK4, and lax.scan including t=0. Production uses FFT diagonalization of that same RK4 update, R(dtλ)^integer_step. This is algebraically the finite-difference RK4 model, not exact time evolution or an exact-continuum symbol correction. It makes repeated inverse fits practical even at fine grids. Float64 is retained throughout.')
    add(f'Maximum full-trajectory scan/evaluator difference: {r["forward"]["scan_spectral_max_error"]:.3e}; loss-gradient difference: {r["forward"]["scan_spectral_gradient_max_error"]:.3e}. Zero-correction RHS, step, trajectory, and differentiation through the initial field are covered by pytest.')
    add('## Independent D3/D6 convergence')
    add('Relative L2 errors for sin(3x), sampled on each grid:')
    add(table(['N','D3 error','D6 error'],[[a['N'],a['D3_relative_error'],a['D6_relative_error']] for a in r['forward']['derivative_convergence']]))
    add('Both second-order derivative stencils converge independently. Their use in dx²-weighted corrections cancels a lower-order defect without making either derivative stencil fourth order.')
    add('## Phase and damping cancellation')
    add('Absolute rate errors against continuum, at v=1, ν4=.002. These isolate spatial symbols, without RK4 temporal error:')
    add(table(['N','k','c3=c6','phase error','damping error'],[[a[k] for k in ['N','k','c','phase_error','damping_error']] for a in r['forward']['rate_convergence']]))
    add('Orders fitted over N=32,48,64:')
    add(table(['mode/model','phase order','damping order'],[[k,v['phase_error'],v['damping_error']] for k,v in r['forward']['orders'].items()]))
    add('These measured orders support approximately fourth-order phase and damping cancellation on the tested resolved modes; this is not a claim about arbitrary under-resolved fields.')
    add('## Stability and frozen graph')
    add('Bounds: v∈[.5,1.5], ν4∈[1e−4,.02], c3,c6∈[0,2]. Each grid samples 625 box combinations (five values per coordinate) and 8193 Fourier angles including all discrete angles. Bisection estimates the first RK4 boundary. This is a dense numerical estimate, not an analytic proof over a continuous box. The selected power-of-two step count is at least 256, reaches T=2 exactly, represents quarter times exactly, and uses at most half the estimated limit. No fit adapts dt to its parameters.')
    add(table(['N','dt_max','dt','steps','fraction'],[[n,p['dt_max'],p['dt'],p['num_steps'],p['safety_fraction']] for n,p in r['forward']['stability'].items()]))
    add('## Model families and optimizer calibration')
    add(table(['Model','Free coordinates','Fixed discrepancy'],[['M0','v,q','0,0'],['M1','v,q','1,1'],['M2','v,q,c3','c6=0'],['M3','v,q,c6','c3=0'],['M4','v,q,c3,c6','none'],['M5','v,q',r['calibrated_coefficients']]]))
    add('q=log(ν4); discrepancy uses direct bounded coordinates. Adam retains β1=.9, β2=.999, ε=1e−8 and box projection. Every fit begins with 2000 updates. Only KKT≥1e−7 triggers continuation of the same parameters, moments, and absolute bias-correction count. The KKT residual zeros outward-blocked gradient components at active bounds (tolerance 1e−8), then takes the Euclidean norm. No restart or per-seed tuning occurs.')
    add('Before any production inference, clean full matched, clean full continuum, and both sparse continuum controls were run at four common learning rates. The frozen selection rule first minimizes failures, then maximum residual; the original .05 is retained if it passes every control. M4 6000-update candidates are considered only when every tested 4000-update setting has failures. All pilot endpoints are retained in JSON; pilot observations contain no held-out recovery noise.')
    add(table(['Model','lr','cap','pilot failures','max pilot KKT'],[[m,c['learning_rate'],c['cap'],c['failures'],c['max_kkt']] for m,c in r['optimizer_calibration']['chosen'].items()]))
    add(table(['Model','pilot lr','cap','failures','max KKT'],[[m,t['learning_rate'],t['cap'],t['failures'],t['max_kkt']] for m,ts in r['optimizer_calibration']['trials'].items() for t in ts]))
    add('## Four-coordinate gradient verification')
    add('At [v,q,c3,c6]=[.9,log(.0015),.7,1.3], centered h=1e−5 differences use the identical fixed graph and observations. The sparse check uses frozen 8-sensor data and seed 699, separate from production seeds.')
    add(table(['dataset','coordinate','JAX','finite difference','absolute difference','relative difference'],[[name,coord,a,b,c,d] for name,g in r['forward']['gradient_checks'].items() for coord,a,b,c,d in zip(['v','q','c3','c6'],g['autodiff'],g['finite_difference'],g['absolute_error'],g['relative_error'])]))
    add('## Known-physics calibration, not inverse-physics recovery')
    add('Field A, all nine times, all spatial points, noiseless continuum data; v=1 and ν4=.002 are held at truth. Only c3,c6 are optimized, from six predeclared starts. Calibration uses lr=.05 and the same 2000→4000 rule. Every parameter/loss history is saved. A common optimum is accepted only if all residuals pass and coefficient spread is below 1e−4; the lowest-loss endpoint then defines frozen M5.')
    add(table(['N32 start','c3','c6','loss','KKT','updates'],[[a['start'],*a['final']['theta'],a['final']['loss'],a['final']['kkt'],a['final_total_updates']] for a in r['calibration']['32']['runs']]))
    add('Cross-grid calibration (the same six starts at each resolution):')
    add(table(['N','c3','c6','best loss','max KKT','coefficient spread','common'],[[n,*c['selected'],min(a['final']['loss'] for a in c['runs']),max(a['final']['kkt'] for a in c['runs']),c['spread'],c['common_optimum_supported']] for n,c in r['calibration'].items()]))
    add('The table quantifies both finite-grid departure from leading theory and approach toward one. Each grid uses its globally safe timestep above. M5 always retains the N=32 coefficients, including the independent cross-grid transfer below; calibration at other grids is diagnostic only.')
    def clean_table(items):
        return table(['case/model','v','ν4','c3','c6','rel v error','rel ν4 error','NMSE','final L2','KKT','updates'],[[k,*a['final']['parameters'],*a['final']['relative_physical_errors'],a['final']['clean_observed_nmse'],a['final']['full_final_l2'],a['final']['kkt'],a['final_total_updates']] for k,a in items])
    add('## Clean matched original-model control')
    add('Truth is generated by the original validated stencil/scan solver at the Stage 9 fixed dt. M1 is intentionally misspecified for these data. Free discrepancy may reach active zero bounds; projected stationarity is appropriate there.')
    add(clean_table([(m,r['clean'][f'matched_A/full/{m}']) for m in r['protocol']['models']]))
    add('## Clean continuum comparison and phase/damping ablations')
    add(clean_table([(m,r['clean'][f'continuum_A/full/{m}']) for m in r['protocol']['models']]))
    base=np.array(r['clean']['continuum_A/full/M0']['final']['parameters'][:2])-np.array([1.,.002])
    add('Signed clean physical biases and changes from M0 (these are fitted interactions, not an assumed additive decomposition):')
    add(table(['model','v bias','ν4 bias','v change from M0','ν4 change from M0'],[[m,*(np.array(r['clean'][f'continuum_A/full/{m}']['final']['parameters'][:2])-[1.,.002]),*(np.array(r['clean'][f'continuum_A/full/{m}']['final']['parameters'][:2])-[1.,.002]-base)] for m in r['protocol']['models']]))
    add('## Clean M4 multistart and identifiability')
    add('All eight starts were frozen before calibration/recovery. The main clean result uses the separately declared start [.9,log(.0015),1,1], not whichever estimate is closest to truth. Bound contacts count history iterates within each optimization segment; endpoint bound flags are also saved. All multistart histories remain available.')
    add(table(['start [v,q,c3,c6]','final [v,ν4,c3,c6]','loss','KKT','updates','endpoint bounds','final-segment contacts'],[[a['start'],a['final']['parameters'],a['final']['loss'],a['final']['kkt'],a['final_total_updates'],a['final']['at_bound'],a['final']['bound_contacts']] for a in r['M4_multistart']]))
    geo=r['geometry'];add('Hessian in [v,q,c3,c6], symmetrized numerically:')
    add(table(['','v','q','c3','c6'],[[name,*row] for name,row in zip(['v','q','c3','c6'],geo['matrix'])]))
    add(f'Eigenvalues: {fmt(geo["eigenvalues"])}. Positive-definiteness threshold: {fmt(geo["positive_definiteness_threshold"])}; positive definite: {fmt(geo["positive_definite"])}; condition number: {fmt(geo["condition_number"])}. Eigenvectors below are columns, in ascending eigenvalue order.')
    add(table(['','e1','e2','e3','e4'],[[name,*row] for name,row in zip(['v','q','c3','c6'],geo['eigenvectors_columns'])]))
    add('Normalized Hessian couplings Hij/√(Hii Hjj), which are not statistical correlations:')
    add(table(['','v','q','c3','c6'],[[name,*row] for name,row in zip(['v','q','c3','c6'],geo['normalized_hessian_couplings'])]))
    add('## Conditional local loss geometry')
    add('Two 41×41 grids keep the other coordinates fixed: q–c6 spans q±.15,c6±.6; v–c3 spans v±.02,c3±.6, clipped to bounds. Full axes and losses are stored in JSON. These are conditional loss slices, not profile likelihoods; nuisance coordinates are not reoptimized. Strong phase/damping couplings and a small Hessian eigenvalue indicate tilted compensation valleys even if a minimum is mathematically isolated. Finite-grid curvature alone does not establish practical identifiability. Optional profile curves and an exact-symbol model are omitted to keep this study focused.')
    add(table(['slice','axis1 extent','axis2 extent','min loss','max loss'],[[name,[s['axis1'][0],s['axis1'][-1]],[s['axis2'][0],s['axis2'][-1]],np.min(s['loss']),np.max(s['loss'])] for name,s in geo['slices'].items()]))
    add('## Clean transfer with frozen N32 discrepancy')
    add('T1=A,(1,.002); T2=C,(1,.002); T3=A,(1.05,.0018); T4=B,(.9,.0015). All use full nine-time fields and the original N32 calibration. No held-out field or parameter pair recalibrates M5.')
    add(clean_table([(f'{t}/{m}',r['clean'][f'{t}/full/{m}']) for t in ['continuum_A','continuum_C','continuum_offnominal','continuum_B'] for m in ['M0','M1','M5']]))
    add('Cross-grid transfer: M5 retains N32 coefficients, not each grid’s diagnostic calibration.')
    add(clean_table([(f'N{n}/{m}',a) for n,models in r['cross_grid_transfer'].items() for m,a in models.items()]))
    add('## Frozen sparse designs and common-noise protocol')
    add(table(['budget','sensors','physical times'],[[b,d['sensor_indices'],d['times']] for b,d in r['protocol']['designs'].items()]))
    add('Times are remapped exactly onto 1024 Stage 9 steps. Moderate: 8 sensors, 2% noise; severe: 4 sensors, 5%. Original-model and continuum-A experiments use seeds 700–719. Held-out continuum C uses 720–739 at the moderate budget. Noise is homoscedastic iid Gaussian, scaled by selected positive-time clean RMS, with t=0 exact. Each truth/design/seed observation array is generated once, saved with its hash, then reused verbatim across models. This preserves the Stage 7/8 recovery convention; no new design search or model-specific noise occurs.')
    add('## Zero-noise sparse references')
    add('Each reference was computed before its corresponding noisy group. If stationarity fails, it remains an approximate endpoint and the associated “clean optimum” interpretation must be qualified.')
    add(clean_table([(k,a) for k,a in r['clean'].items() if '/full/' not in k]))
    for truth,title in [('matched_A','Sparse/noisy matched-model control'),('continuum_A','Sparse/noisy continuum comparison'),('continuum_C','Held-out field-C noisy transfer')]:
        add('## '+title)
        add('Errors below are relative fractions, not percentages. SD uses ddof=1. Rates count final bound contact, continuation, and failed final KKT respectively.')
        selected=[(k,g['summary']) for k,g in r['noisy'].items() if k.startswith(truth+'/')]
        add(table(['budget/model','mean v err','median v err','mean ν err','median ν err','p90 ν err','SD v','SD ν','mean final L2','clean NMSE','noisy loss'],[[k.split('/',1)[1],s['mean_relative_errors'][0],s['median_relative_errors'][0],s['mean_relative_errors'][1],s['median_relative_errors'][1],s['p90_nu4_relative_error'],s['parameter_sd'][0],s['parameter_sd'][1],s['mean_final_field_l2'],s['mean_clean_observed_nmse'],s['mean_noisy_loss']] for k,s in selected]))
        add(table(['budget/model','mean c3','SD c3','mean c6','SD c6','bound rate','continuation rate','unconverged rate','max KKT'],[[k.split('/',1)[1],s['parameter_mean'][2],s['parameter_sd'][2],s['parameter_mean'][3],s['parameter_sd'][3],s['bound_rate'],s['continuation_rate'],s['unconverged_rate'],s['max_kkt']] for k,s in selected]))
    add('## Physical error = clean model/design bias + noise-induced shift')
    add('The decomposition is in physical coordinates [v,ν4], not q. It is exact for every paired run; the full per-seed vectors and all endpoints are retained. Means below are signed, so they differ from mean absolute relative recovery errors above.')
    add(table(['truth/budget/model','mean physical error [v,ν]','clean bias [v,ν]','mean noise shift [v,ν]'],[[k,*[g['summary']['mean_decomposition'][name] for name in ['physical_error','clean_model_design_bias','noise_induced_shift']]] for k,g in r['noisy'].items()]))
    add('For M4, discrepancy shifts are measured from its own clean reference:')
    add(table(['group','mean Δc3','SD Δc3','mean Δc6','SD Δc6'],[[k,np.mean([a['discrepancy_noise_shift'][0] for a in g['runs']]),np.std([a['discrepancy_noise_shift'][0] for a in g['runs']],ddof=1),np.mean([a['discrepancy_noise_shift'][1] for a in g['runs']]),np.std([a['discrepancy_noise_shift'][1] for a in g['runs']],ddof=1)] for k,g in r['noisy'].items() if k.endswith('/M4')]))
    add('## Fixed theory, calibrated, and free discrepancy: interpretation')
    add(r.get('scientific_interpretation','Interpretation pending; do not infer physical success from residuals alone.'))
    add('## Continuation and unresolved stationarity')
    add(f'Production comparisons, multistarts, grid transfers, and noisy fits: {len(fits)} total; {len(continued)} continued; {len(failed)} remain above 1e−7. Calibration and optimizer pilots are reported separately and not included in this count. Original 2000-update states, gradients, moments, losses, and final states are retained for every production fit; continuation changes and total counts are explicit. No failed run is removed from summaries.')
    add(table(['group','fits','continued','unconverged','max KKT'],[[k,len(g['runs']),sum(a['continuation_required'] for a in g['runs']),sum(a['final']['kkt']>=1e-7 for a in g['runs']),max(a['final']['kkt'] for a in g['runs'])] for k,g in r['noisy'].items()]))
    add(table(['continued fit','initial parameters','initial KKT','final parameters','final KKT','updates'],
      [[k,[a['initial']['theta'][0],float(np.exp(a['initial']['theta'][1]))]+a['initial']['theta'][2:],a['initial']['kkt'],a['final']['parameters'],a['final']['kkt'],a['final_total_updates']] for k,a in continued]))
    if failed:
        add('The following endpoints fail stationarity. Their field and physical estimates are descriptive finite-budget results; unresolved optimization must not be misreported as demonstrated non-identifiability or as a calibrated scientific winner.')
        add(table(['fit','initial KKT','final KKT','updates','continuation parameter change'],[[k,a['initial']['kkt'],a['final']['kkt'],a['final_total_updates'],a['parameter_changes_during_continuation']] for k,a in failed]))
    add('## Tests and acceptance')
    add('```text\n'+r['validation']['summary']+'\n```')
    add(table(['check','pass'],checks.items()))
    add('**'+r['conclusion']+'**. PASS does not require c=1, perfect physical recovery, a well-conditioned Hessian, universal correction benefit, or perfect transfer. Unresolved numerical failures are distinguished from valid unfavorable scientific findings.')
    add('## Scientific caveats')
    add('The correction is only a low-order modified-equation basis. Its dimensionless multipliers are scheme-dependent, not physical constants, and calibration at one dx need not transfer. Joint discrepancy/physics inference can create structural or practical non-identifiability; Hessian conditioning depends on parameter coordinates and scaling. Continuum truth is idealized, initial conditions are exactly known, and noise is synthetic iid Gaussian. Lower state error does not imply more accurate physics. Fixed theory uses numerical-analysis knowledge unavailable for generic unknown model error. No arbitrary functional discrepancy, neural closure, or Bayesian uncertainty model is learned here.')
    add('## Recommended Stage 10 (not implemented)')
    add(r.get('stage10_recommendation','Choose only after interpreting completed results.'))
    add('## Reproduction and checkpoints')
    add('```sh\n.venv/bin/python -m pytest -q\n.venv/bin/python -m experiments.stage9_model_discrepancy --forward-validation\n.venv/bin/python -m experiments.stage9_model_discrepancy --calibrate\n.venv/bin/python -m experiments.stage9_model_discrepancy --clean-inference\n.venv/bin/python -m experiments.stage9_model_discrepancy --noisy-inference\n.venv/bin/python -m experiments.stage9_model_discrepancy --finalize\n```')
    add('Atomic JSON checkpoints retain completed groups; exceptions are recorded and raised. Resume validates noisy group counts and the finalizer requires all 23 groups and 44 clean references. Nonstationary endpoints are retained, not silently rerun or discarded. JSON serialization rejects NaN/Infinity. Git status: this directory is not a Git repository; no commit or push was attempted.')
    (ROOT/'docs/stage9_validation.md').write_text('\n\n'.join(out)+'\n')
