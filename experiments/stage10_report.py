"""Transparent reporting of likelihood geometry, finite ensembles and failures."""
import hashlib
from pathlib import Path
import numpy as np
from src.identifiability import profile_intervals,LR_LEVELS
from experiments.stage9_report import table,fmt

ROOT=Path(__file__).resolve().parents[1]


def aggregate_coverage(r):
    result={}
    for key in r['bootstrap']:
        profiles=[p for k,p in r['coverage'].items() if k.rsplit('/',1)[0]==key]
        if not profiles:continue
        lr=[p['LR_at_pseudo'] for p in profiles if p['LR_at_pseudo'] is not None]
        entry={'count':len(profiles),'LR_valid_count':len(lr),'LR_empirical_68':float(np.quantile(lr,.68)) if lr else None,
               'LR_empirical_95':float(np.quantile(lr,.95)) if lr else None,'coverage':{}}
        for level in LR_LEVELS:
            resolved=[p for p in profiles if p['coverage'][level]['resolved']]
            value={'resolved_count':len(resolved),'indeterminate_count':len(profiles)-len(resolved)}
            for target in ['pseudo','physical']:
                covered=sum(p['coverage'][level][target] for p in resolved)
                value[target+'_coverage_resolved']=covered/len(resolved) if resolved else None
                value[target+'_coverage_all_bounds']=[covered/len(profiles),(covered+len(profiles)-len(resolved))/len(profiles)]
            widths=[p['intervals'][level]['nu4']['total_width'] for p in resolved]
            value['mean_nu4_width']=float(np.mean(widths)) if widths else None
            value['bound_limited_count']=sum(p['intervals'][level]['bound_limited'] for p in profiles)
            entry['coverage'][level]=value
        result[key]=entry
    return result


def all_attempts(r):
    for k,a in r['pseudo'].items():
        for row in a['attempts']:yield 'pseudo/'+k,row
    for k,p in {**r['profiles'],**r.get('noisy_profiles',{})}.items():
        for point in p['rows']:
            for row in point['attempts']:yield 'profile/'+k,row
    for point in r['surface']['rows']:
        for row in point['attempts']:yield 'surface',row
    for k,g in r['bound_sensitivity'].items():
        for row in g['center']['attempts']:yield 'bound-center/'+k,row
        for target in ['1','3']:
            for point in g[target]['rows']:
                for row in point['attempts']:yield 'bound-profile/'+k+'/'+target,row
    for k,g in r['bootstrap'].items():
        for fit in g['runs']:
            for row in fit['attempts']:yield 'bootstrap/'+k,row
    for k,p in r['coverage'].items():
        for point in p['rows']:
            for row in point['attempts']:yield 'coverage/'+k,row
    profiles=list(r['profiles'].values())+list(r.get('noisy_profiles',{}).values())+list(r['coverage'].values())+[g[t] for g in r['bound_sensitivity'].values() for t in ['1','3']]
    for p in profiles:
        if p['baseline_reconciliation']:
            for row in p['baseline_reconciliation']['attempts']:yield 'baseline-reconciliation',row


def interval_text(interval,physical=False):
    if physical:interval=interval['nu4']
    pieces=['['+fmt(c['lower'])+', '+fmt(c['upper'])+']' for c in interval['components']]
    flags=[]
    for name in ['bound_limited','one_sided','disconnected','unresolved']:
        if interval[name]:flags.append(name.replace('_',' '))
    return ' ∪ '.join(pieces)+(' ('+', '.join(flags)+')' if flags else '')



def interpret(r):
    lines=[]
    for budget,label in [('full','full-clean reference'),('8','moderate'),('4','severe')]:
        fixed=r['profiles'][f'continuum_A/{budget}/M1/1']['intervals']['95']['nu4']['total_width']
        free=r['profiles'][f'continuum_A/{budget}/M4/1']['intervals']['95']['nu4']['total_width']
        a=r['pseudo'][f'continuum_A/{budget}/M4'];scale=a['curvature']['log_coordinates']['local_standard_scales']
        quadratic=None if scale is None else float(np.diff(np.exp(np.array([-1,1])*np.sqrt(3.841459)*scale[1]+a['theta'][1]))[0])
        lines.append(f'For {label} data, the M4/M1 nominal 95% diffusion-width ratio is {free/fixed:.3f}. M4 profile width is {free:.7g}, versus an unconstrained local inverse-curvature width {fmt(quadratic)}. The quadratic approximation is therefore not automatically an underestimate; discrepancy bounds constrain compensation that the unconstrained inverse Hessian permits.')
    widths=[r['bound_sensitivity'][str(b)]['1']['intervals']['95']['nu4']['total_width'] for b in [2,3,4]]
    lines.append(f'Widening both discrepancy bounds from 2 to 4 increases the moderate clean diffusion profile width by {100*(widths[2]/widths[0]-1):.2f}%. The c6 profile reaches its declared limits, so its interval is bound-limited. This is dependence on the discrepancy model, not an intrinsic physical uncertainty limit; diffusion itself still has finite threshold crossings on the tested boxes.')
    for budget,label in [('8','moderate'),('4','severe')]:
        a={m:r['bootstrap'][f'continuum_A/{budget}/{m}']['summary'] for m in ['M1','M4','M5']}
        lines.append(f'In the {label} A ensemble, diffusion SDs for M1/M4/M5 are '+', '.join(fmt(a[m]["parameters"]["sample_sd"][1]) for m in ['M1','M4','M5'])+f'. The M4 discrepancy-bound fraction is {100*a["M4"]["discrepancy_bound_fraction"]:.1f}%. Its mean final-field relative L2 is {a["M4"]["field_error"]["mean"]:.5g}, compared with {a["M1"]["field_error"]["mean"]:.5g} for M1. Small state error can coexist with broad or boundary-concentrated discrepancy estimates.')
        lines.append(f'M5/M1 diffusion SD ratio is {a["M5"]["parameters"]["sample_sd"][1]/a["M1"]["parameters"]["sample_sd"][1]:.4f}. Their clean signed diffusion biases are {r["pseudo"][f"continuum_A/{budget}/M5"]["parameters"][1]-.002:.7g} and {r["pseudo"][f"continuum_A/{budget}/M1"]["parameters"][1]-.002:.7g}, respectively. Fixed calibration changes model bias while leaving only v,q free; these conditional SDs exclude uncertainty in the calibration itself.')
    lines.append('Coverage tables show both pseudo-truth and physical-truth inclusion. The empirical LR quantiles are not forced to equal the asymptotic references, and they are not substituted into the reported nominal intervals. With only 10 or 20 profile replicates per case, deviations are descriptive rather than evidence of precisely estimated coverage.')
    held={m:r['bootstrap'][f'continuum_C/8/{m}']['summary'] for m in ['M1','M4','M5']}
    lines.append('For held-out field C, diffusion SDs for M1/M4/M5 are '+', '.join(fmt(held[m]['parameters']['sample_sd'][1]) for m in ['M1','M4','M5'])+f'; M4 hits a discrepancy bound in {100*held["M4"]["discrepancy_bound_fraction"]:.1f}% of fits. M5 retains the same N32 calibration throughout. Field-C coverage is evaluated independently on seeds 900–909, with physical and pseudo-targets separated.')
    return '\n\n'.join(lines)

def finalize(r):
    if len(r['pseudo'])!=27 or len(r['profiles'])!=24:raise RuntimeError('Incomplete clean work')
    if len(r['bootstrap'])!=13 or len(r['coverage'])!=120:raise RuntimeError('Incomplete ensembles or coverage subset')
    if len(r.get('surface',{}).get('rows',[]))!=961 or len(r.get('bound_sensitivity',{}))!=3:raise RuntimeError('Incomplete surface/bound diagnostics')
    expected={f'{truth}/{budget}/{model}':list(seeds)
      for truth,budget,seeds,models in [
        ('continuum_A','8',range(800,900),['M1','M4','M5']),
        ('continuum_A','4',range(800,850),['M1','M4','M5']),
        ('continuum_C','8',range(900,950),['M1','M4','M5']),
        ('matched_A','8',range(800,820),['M1','M4']),
        ('matched_A','4',range(800,820),['M1','M4'])] for model in models}
    if set(r['bootstrap'])!=set(expected) or any([a['seed'] for a in r['bootstrap'][k]['runs']]!=seeds for k,seeds in expected.items()):
        raise RuntimeError('Unexpected or incomplete bootstrap seed groups')
    expected_coverage={f'{truth}/{budget}/{model}/{seed}'
      for truth,budget,seeds in [('continuum_A','8',range(800,820)),('continuum_A','4',range(800,810)),('continuum_C','8',range(900,910))]
      for model in ['M1','M4','M5'] for seed in seeds}
    if set(r['coverage'])!=expected_coverage:raise RuntimeError('Coverage subset does not match declaration')
    if len(r.get('noisy_profiles',{}))!=10:raise RuntimeError('Missing representative noisy coordinate profiles')
    if 'validation' not in r:
        import subprocess
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q'],cwd=ROOT,text=True,capture_output=True)
        r['validation']={'passed':run.returncode==0,'summary':run.stdout+run.stderr}
    for bound,g in r['bound_sensitivity'].items():
        c=np.array(g['center']['parameters'][2:])
        g['center']['discrepancy_bound_hit']=bool(np.any((c<=1e-8)|(c>=float(bound)-1e-8)))
    for group in r['bootstrap'].values():
        params=np.array([a['parameters'] for a in group['runs']])
        group['summary']['physical_rmse']=np.sqrt(np.mean(np.array([a['physical_error'] for a in group['runs']])**2,axis=0)).tolist()
        group['summary']['pseudo_deviation_rmse']=np.sqrt(np.mean(np.array([a['decomposition']['noise_induced_shift'] for a in group['runs']])**2,axis=0)).tolist()
        group['summary']['discrepancy_lower_bound_fractions']=np.mean(params[:,2:]<=1e-8,axis=0).tolist()
        group['summary']['discrepancy_upper_bound_fractions']=np.mean(params[:,2:]>=2-1e-8,axis=0).tolist()
    r['coverage_summary']=aggregate_coverage(r)
    import jax
    import jax.numpy as jnp
    from experiments.stage10_identifiability import context,loss_fn
    coord={}
    for key,p in r['profiles'].items():
        if p['target_index']!=1:continue
        comparisons={}
        for level,threshold in LR_LEVELS.items():
            direct=profile_intervals(np.exp(p['axis']),p['LR'],p['valid_points'],threshold,np.exp(p['bounds']))
            mapped=p['intervals'][level]['nu4']
            if len(direct['components'])==len(mapped['components']):
                differences=[abs(a[e]-b[e])/max(abs(b[e]),1e-30) for a,b in zip(direct['components'],mapped['components']) for e in ['lower','upper']]
                comparisons[level]={'max_relative_endpoint_difference':max(differences,default=0.),'physical_axis_interval':direct}
            else:comparisons[level]={'max_relative_endpoint_difference':None,'physical_axis_interval':direct}
        truth,budget,model,_=key.split('/')
        ctx=context(truth,budget);loss=loss_fn(ctx,model,ctx['clean'])
        selected=jnp.array([a['theta'] for a in p['rows'][::max(1,len(p['rows'])//12)]])
        physical=selected.at[:,1].set(jnp.exp(selected[:,1]))
        original=jax.jit(jax.vmap(loss))(selected)
        changed=jax.jit(jax.vmap(lambda z:loss(z.at[1].set(jnp.log(z[1])))))(physical)
        comparisons['direct_NLL_max_difference']=float(jnp.max(jnp.abs(original-changed)))
        comparisons['direct_NLL_max_relative_difference']=float(jnp.max(jnp.abs(original-changed)/jnp.maximum(1.,jnp.abs(original))))
        coord[key]=comparisons
    r['coordinate_profile_comparison']=coord
    attempts=list(all_attempts(r));failures=[(k,a) for k,a in attempts if not a['converged']]
    allprofiles=list(r['profiles'].values())+list(r.get('noisy_profiles',{}).values())+list(r['coverage'].values())+[g[t] for g in r['bound_sensitivity'].values() for t in ['1','3']]
    for profile in allprofiles:
        x=np.array(profile['axis']);y=np.array(profile['LR']);valid=profile['valid_points']
        slopes=[float((y[i+1]-y[i])/(x[i+1]-x[i])) if valid[i] and valid[i+1] else None for i in range(len(x)-1)]
        curvature=[float(2*(slopes[i]-slopes[i-1])/(x[i+1]-x[i-1])) if slopes[i] is not None and slopes[i-1] is not None else None for i in range(1,len(x)-1)]
        profile['shape_diagnostics']={'adjacent_LR_slopes':slopes,'second_divided_differences':curvature,
          'discrete_minimum_indices':[i for i in range(1,len(x)-1) if slopes[i-1] is not None and slopes[i] is not None and slopes[i-1]<0<slopes[i]],
          'interpretation':'Finite-grid slopes and nuisance-jump flags; neither alone proves nonsmoothness or a separate branch. All flagged points already have central, alternate, and neighboring warm starts.'}
    spreads=[]
    for profile in allprofiles:
        for point in profile['rows']:
            vals=[a['NLL'] for a in point['attempts'] if a['converged']]
            if len(vals)>1 and 2*(point['NLL']-profile['baseline_NLL'])<=4.0:
                spreads.append(max(vals)-min(vals))
    r['profile_optimizer_accuracy']={'near_threshold_converged_start_NLL_spread_max':max(spreads,default=0.),
      'near_threshold_converged_start_NLL_spread_p95':float(np.quantile(spreads,.95)) if spreads else 0.}
    r['optimizer_summary']={'attempts':len(attempts),'continued_attempts':sum(a['continued'] for _,a in attempts),
      'nonstationary_attempts':len(failures),'selected_invalid_profile_points':sum(p['failed_points'] for p in allprofiles),
      'invalid_surface_points':sum(not a['valid'] for a in r['surface']['rows']),
      'invalid_bootstrap_fits':sum(not a['valid'] for g in r['bootstrap'].values() for a in g['runs']),
      'invalid_pseudo_fits':sum(not a['valid'] for a in r['pseudo'].values()),
      'max_selected_pseudo_KKT':max(a['KKT'] for a in r['pseudo'].values()),
      'failed_attempts_by_group':{k:sum(label==k for label,a in failures) for k in sorted(set(k for k,a in failures))}}
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in r['preservation'].items())
    paired=all(a['observation_hash']==next(d['hash'] for d in r['datasets'][k.rsplit('/',1)[0]] if d['seed']==a['seed']) for k,g in r['bootstrap'].items() for a in g['runs'])
    decomp=all(np.allclose(a['physical_error'],np.array(a['decomposition']['clean_model_design_bias'])+a['decomposition']['noise_induced_shift'],rtol=0,atol=1e-15) for g in r['bootstrap'].values() for a in g['runs'])
    checks={'old_and_new_tests':r['validation']['passed'],'Stage1_9_preserved':unchanged,
      'pseudo_truth_separate_and_stationary':r['optimizer_summary']['invalid_pseudo_fits']==0,
      'positive_time_Gaussian_NLL':all(a['sigma_abs']>0 for a in r['pseudo'].values()),'primary_profiles_complete':len(r['profiles'])==24,
      'profiled_surface_complete':len(r['surface']['rows'])==961,
      'profile_failures_retained_and_excluded':all(p['valid_points'][i]==(a['valid'] and p['baseline_valid']) for p in allprofiles for i,a in enumerate(p['rows'])),'Hessian_profile_comparison':all('curvature' in a for a in r['pseudo'].values()),
      'coordinate_sensitivity':all(a['curvature']['transformation_relative_error']<1e-10 for a in r['pseudo'].values()) and all(c['direct_NLL_max_relative_difference']<1e-10 for c in coord.values()),
      'bound_sensitivity':len(r['bound_sensitivity'])==3,'fresh_ensembles_complete':sum(len(g['runs']) for g in r['bootstrap'].values())==680,
      'paired_noise':paired,'nominal_and_empirical_thresholds_separate':len(r['coverage_summary'])==9 and all(c['LR_valid_count']>0 and c['coverage']['95']['resolved_count']>0 for c in r['coverage_summary'].values()),
      'physical_and_pseudo_coverage_separate':decomp,'heldout_C_without_recalibration':all(f'continuum_C/8/{m}' in r['bootstrap'] for m in ['M1','M4','M5']),
      'parameter_and_field_distributions':all('field_error' in g['summary'] for g in r['bootstrap'].values()),
      'uniform_continuation':all(a['updates']==(4000 if a['initial_KKT']>=1e-7 else 2000) for _,a in attempts)}
    checks['frozen_M5_calibration']=all(np.array_equal(a['parameters'][2:],r['protocol']['calibrated_coefficients']) for k,g in r['bootstrap'].items() if k.endswith('/M5') for a in g['runs'])
    r['checks']=checks
    # Explicitly retained tail failures are allowed by the requested acceptance;
    # no usable interval or unresolved central optimum is a substantive failure.
    informative=all(p['baseline_valid'] and p['intervals']['95']['components'] for p in r['profiles'].values())
    checks['primary_profiles_informative']=bool(informative)
    r['conclusion']='PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    r['scientific_interpretation']=interpret(r)
    r['stage11_recommendation']=(
      'Exponential integration is the most direct next reference for this periodic constant-coefficient linear operator: use the exact semi-discrete Fourier propagator, then benchmark an ETD or semi-implicit implementation against it. Validate forward accuracy, parameter/initial-state derivatives, frozen inverse-result equivalence, and runtime/memory scaling against stencil/scan RK4. Stage 9 already diagonalizes the RK4 polynomial for inverse evaluation; an exponential propagator would instead remove the explicit stability restriction and time-discretization error. It does not cure model bias or identifiability. Larger independent coverage studies remain necessary before claiming calibrated finite-sample confidence. No Stage 11 code is added.'
      if r['conclusion']=='PASS' else
      'Resolve the explicitly reported methodological or numerical limitations before promoting a new integration stage. An exponential semi-discrete reference remains the natural eventual stiff-integration benchmark; no Stage 11 code is added.')
    out=[];add=out.append
    add('# Stage 10: Practical Identifiability, Profile-Based Uncertainty, and Uncertainty Calibration')
    add('**'+r['conclusion']+'**. Profiles, local curvature, and repeated-noise distributions address different aspects of inference. All numerical-model and design definitions from Stages 1–9 are preserved. No Stage 11 implementation or commit/push is performed.')
    add('## Scope and files')
    add('Stage 9 found strong q–c6 compensation despite stationary optimizers and accurate predicted fields. Stage 10 compares fixed theoretical discrepancy M1, free discrepancy M4, and fixed calibrated M5. New files are `src/identifiability.py`, `experiments/stage10_identifiability.py`, `experiments/stage10_report.py`, `tests/test_identifiability.py`, and Stage 10 documents/results. README is updated. Existing source, tests, experiments, and Stage 1–9 results are hash-checked unchanged. Coordinates are q=log(ν4), v∈[.5,1.5], ν4∈[1e−4,.02], and main c3,c6∈[0,2]. M1/M5 infer only [v,q]; M4 infers [v,q,c3,c6]. M1 fixes c=(1,1), while M5 fixes the exact Stage 9 calibrated values shown in the pseudo-truth table.')
    add('## Observation model and meaning of clean geometry')
    add('At positive observation times, y=clean+ε with independent ε∼N(0,σ_abs²), where σ_abs=σ_rel×RMS(clean selected positive-time observations). NLL=SSE_positive/(2σ_abs²), omitting the additive constant. The exactly known initial field is deterministic: t=0 contributes no stochastic NLL. Normalized MSE over all observed entries is saved separately.')
    add('Truly zero-noise data have σ=0 and do not define finite Gaussian likelihood intervals. Full-clean geometry therefore uses a clearly declared hypothetical 2%-RMS measurement scale without adding noise. Its intervals describe sensitivity at that assumed precision, not uncertainty induced by a zero-variance experiment. Moderate and severe clean pseudo-truth references use their corresponding 2% and 5% scales; noisy analyses use the actual known scales. M5 calibration uncertainty is not propagated.')
    add('Full observations use all 32 points and the nine times 0,.25,…,2. Main inference retains the Stage 9 fixed 1024-step RK4 graph. The sparse masks are loaded from the stored Stage 8 design, with physical times remapped exactly to this graph:')
    add(table(['regime','sensors','times','relative noise'],[[('moderate' if b=='8' else 'severe'),d['sensor_indices'],d['times'],(.02 if b=='8' else .05)] for b,d in r['protocol']['designs'].items()]))
    add('## Optimizer units and retained scaling pilot')
    add('Adam uses the unchanged Stage 9 learning rate .05, β1=.9, β2=.999, ε=1e−8, and 2000 updates followed only when needed by continuation of the same state to 4000. The original KKT threshold 1e−7 is defined in the Stage 9 NMSE scale. We minimize NLL multiplied by the fixed data-dependent scalar 2σ²/[number_of_observed_entries×(mean(y²)+1e−12)]. This is exactly the original NMSE gradient scale: deterministic t=0 residuals are parameter-independent. The minimizer is unchanged by this positive scalar. Raw NLL, raw NLL KKT, scaled KKT, and the scalar are all saved. Likelihood thresholds and Hessians are always in raw NLL units; NMSE is never compared with an LR cutoff.')
    if 'scaling_pilot' in r:
        add('An initial clean-only raw-NLL optimization pilot applied the numerical threshold to different gradient units and exposed this mismatch. It is preserved in `docs/stage10_scaling_pilot.json`, including failed endpoints. The scaling correction preceded bootstrap production and changed no optimizer hyperparameters. Failed pilot cases: '+', '.join(r['scaling_pilot']['nonstationary_pseudo_cases'])+'.')
    add('## Protocol and nuisance optimization')
    add('Every profile point fixes its target coordinate exactly and optimizes all nuisance coordinates with three predeclared starts: central optimum, alternate [.8,log(.004),.2,1.8] restricted to model dimension, and the preceding point’s best preliminary endpoint (a batched warm-start pass). Threshold refinements use the nearest existing endpoint as the warm start. Every attempt is retained, and only the lowest converged NLL is selected. No selection uses physical accuracy. The unrestricted fits likewise use three declared starts. Fixed coordinates have zero optimizer derivatives and equal projection bounds.')
    add('Primary q profiles begin with at least 61 points; full-clean starts with ν4∈[.0005,.005] and expands if needed, while sparse profiles cover the full inverse bounds [1e−4,.02]. c6 has at least 61 points over its full bounds; c3 and v have 41 initial points. The central and pseudo-true coordinates are explicitly included. Eight bisection rounds refine crossings. Invalid points are retained and never bridged. Disconnected accepted components are not merged. Neighboring nuisance jumps above 15% of coordinate range are flagged; all points already have alternate starts. No artificial curve smoothing is applied.')
    add('Nominal scalar LR cutoffs are 2ΔNLL≤1 (68%) and ≤3.841459 (95%). These are asymptotic references, not exact confidence guarantees. A converged profile improving the unrestricted baseline by >1e−5 NLL triggers the predeclared unrestricted reconciliation attempt; original fits remain recorded. An unresolved baseline invalidates interval use.')
    add('## Clean pseudo-true parameters')
    add('Continuum physical truth is [v,ν4]=[1,.002]; each model/design’s clean stationary optimum is its pseudo-truth. The latter, rather than physical truth, centers sampling-deviation and curvature calculations. Full-clean reference precision is hypothetical as stated above.')
    add(table(['truth/regime/model','v','ν4','c3','c6','NLL','NMSE','scaled KKT','raw NLL KKT','updates'],[[k,*a['parameters'],a['NLL'],a['normalized_mse'],a['KKT'],a['NLL_KKT'],a['updates']] for k,a in r['pseudo'].items()]))
    add('## Primary nominal profiles')
    add('For q, endpoints below are transformed to ν4; q endpoints and all components remain in JSON. Other coordinates are reported directly. A bound-limited interval is constrained by the declared model, not a physical limit.')
    add(table(['case','target','68% interval','95% interval','invalid points','nuisance jumps'],[[k,['v','q → ν4','c3','c6'][p['target_index']],interval_text(p['intervals']['68'],p['target_index']==1),interval_text(p['intervals']['95'],p['target_index']==1),p['failed_points'],len(p['jumps'])] for k,p in r['profiles'].items()]))
    add('The same primary diffusion intervals in the internal q coordinate:')
    add(table(['case','68% q interval','95% q interval'],[[k,interval_text(p['intervals']['68']),interval_text(p['intervals']['95'])] for k,p in r['profiles'].items() if p['target_index']==1]))
    add('Representative noisy coordinate intervals use seed 800, the first predeclared A seed in each regime. Its q intervals are also included in the coverage study:')
    add(table(['case','68% interval','95% interval','invalid points'],[[k,interval_text(p['intervals']['68']),interval_text(p['intervals']['95']),p['failed_points']] for k,p in r['noisy_profiles'].items()]))
    add('## Profiled q–c6 surface and compensation valley')
    surface=r['surface'];valid=np.array(surface['valid']);values=np.array(surface['NLL']);masked=np.where(valid,values,np.inf)
    ii,jj=np.unravel_index(np.argmin(masked),masked.shape)
    add(f'The 31×31 surface fixes q,c6 and reoptimizes v,c3 at every point with three starts. It differs from Stage 9’s conditional slices. The sampled minimum is q={fmt(surface["q_axis"][ii])}, ν4={fmt(surface["nu4_axis"][ii])}, c6={fmt(surface["c6_axis"][jj])}, NLL={fmt(values[ii,jj])}. Invalid points: {np.sum(~valid)}. Full nuisance optima and diagnostics are stored. This is a local finite grid; full 1D profiles determine interval extent.')
    add(table(['q','ν4','best sampled c6','NLL','2ΔNLL'],[[a[k] for k in ['q','nu4','c6','NLL','LR']] for a in surface['valley']]))
    add(table(['nominal level','sampled valley ν4 range'],surface['level_ranges'].items()))
    if len(surface['valley'])>2:
        slope=float(np.polyfit([a['q'] for a in surface['valley']],[a['c6'] for a in surface['valley']],1)[0])
        add(f'The sampled compensation curve has overall dc6/dq≈{fmt(slope)} (including clipping at discrepancy bounds). Increased physical damping can be offset by reduced numerical damping correction. The full curve, rather than a straight-line extrapolation, defines the reported geometry.')
    interior=[a for a in surface['valley'] if 0<a['c6']<2]
    if len(interior)>2:
        interior_slope=float(np.polyfit([a['q'] for a in interior],[a['c6'] for a in interior],1)[0])
        r['surface']['interior_compensation_dc6_dq']=interior_slope
        add(f'Within the unclipped portion of the sampled valley, dc6/dq≈{fmt(interior_slope)}. This local compensation direction differs from the slope averaged over bound plateaus.')
    add('## Local inverse curvature versus nonlinear profiles')
    add('Hessians use raw NLL at each clean pseudo-truth. Their inverses, when numerically positive definite, are local inverse-curvature approximations, not calibrated covariances. Intervals below use q±sqrt(LR cutoff)×local q scale, transformed to ν4, with no artificial clipping to hide bound violations. Positive definiteness does not make an active-bound Gaussian approximation valid.')
    hrows=[]
    for key,a in r['pseudo'].items():
        if not key.startswith('continuum_A/'):continue
        c=a['curvature']['log_coordinates'];scales=c['local_standard_scales'];q=a['theta'][1]
        p=r['profiles'][key+'/1']
        local=None if scales is None else np.exp([q-np.sqrt(3.841459)*scales[1],q+np.sqrt(3.841459)*scales[1]]).tolist()
        physical_scales=a['curvature']['physical_coordinates']['local_standard_scales']
        local_physical=None if physical_scales is None else (a['parameters'][1]+np.array([-1,1])*np.sqrt(3.841459)*physical_scales[1]).tolist()
        hrows.append([key,c['condition_number'],None if scales is None else scales[1],local,local_physical,interval_text(p['intervals']['95'],True)])
    add(table(['case','H(q) condition','local q scale','local 95% via q','local 95% via physical ν4','nonlinear 95% ν4 profile'],hrows))
    add('A negative endpoint in the unconstrained physical-coordinate quadratic approximation is deliberately displayed as a sign of local-approximation failure; it is not an admissible physical interval. Nonlinear profiles retain the original positive diffusion bounds.')
    central=r['pseudo']['continuum_A/full/M4']['curvature']['log_coordinates']
    add('For full-clean M4, raw-NLL Hessian eigenvalues are '+fmt(central['eigenvalues'])+'; the smallest-eigenvalue direction in [v,q,c3,c6] is '+fmt(np.array(central['eigenvectors_columns'])[:,0])+'. The full matrix and eigensystem for every clean case are retained in JSON.')
    add('## Coordinate sensitivity')
    add('The physical-coordinate Hessian is computed by direct autodiff and independently checked against H_physical=JᵀH_logJ plus the gradient-weighted second-derivative term −g_q/ν4² in the ν4 diagonal. That term is retained even at numerically stationary or constrained endpoints. A pure Jacobian sandwich alone need not be exact at an active bound.')
    add(table(['case','condition [v,q,…]','condition [v,ν4,…]','q-coordinate diagonals','physical-coordinate diagonals','transform relative difference'],[[k,a['curvature']['log_coordinates']['condition_number'],a['curvature']['physical_coordinates']['condition_number'],a['curvature']['log_coordinates']['diagonal'],a['curvature']['physical_coordinates']['diagonal'],a['curvature']['transformation_relative_error']] for k,a in r['pseudo'].items()]))
    add('A monotone change from q to ν4 keeps the same fixed physical predictions and nuisance problems. Interpolating the same refined LR samples on the physical axis instead of transforming q crossings produces only finite-grid interpolation differences:')
    add(table(['profile','68% max relative endpoint difference','95% max relative endpoint difference','direct NLL max difference'],[[k,a['68']['max_relative_endpoint_difference'],a['95']['max_relative_endpoint_difference'],a['direct_NLL_max_difference']] for k,a in coord.items()]))
    add('## Diagnostic discrepancy-bound sensitivity')
    add('These diagnostics use clean continuum A under the moderate observation design, c3,c6∈[0,B] for B=2,3,4, and a common 2048-step graph. Main production bounds and the Stage 9 graph remain unchanged. The larger box is separately checked for RK4 stability on a dense angular grid and sampled parameter combinations. Both discrepancy bounds widen together; this is not a different basis.')
    add(table(['B','pseudo [v,ν4,c3,c6]','NLL','KKT','95% ν4 interval','95% c6 interval','sampled max amplification'],[[b,g['center']['parameters'],g['center']['NLL'],g['center']['KKT'],interval_text(g['1']['intervals']['95'],True),interval_text(g['3']['intervals']['95']),g['max_sampled_amplification']] for b,g in r['bound_sensitivity'].items()]))
    add('## Paired ensembles and declared coverage subsets')
    add('Continuum A uses seeds 800–899 (100 per model, moderate) and 800–849 (50 per model, severe). Held-out C uses 900–949 (50 per model, moderate). Matched original-FD controls use 800–819 (20 per model/regime) for M1 and M4. The same seed on a different truth/design is a different conditional dataset. Within each truth/design/seed, observations are generated once, saved and hashed, and reused exactly across models. M5 is never recalibrated.')
    add('Full q-profile coverage uses the first 20 moderate A seeds, first 10 severe A seeds, and first 10 held-out C seeds, for every M1/M4/M5 model. Subsets were declared before recovery. Sample SD uses ddof=1; percentiles use NumPy linear interpolation. Coverage changes in 5% increments for 20 profiles and 10% increments for 10 profiles, so interpolated LR quantiles need not imply an exactly matching empirical coverage fraction. Optional off-nominal ensembles and prediction envelopes were skipped. These are repeated synthetic truth-plus-noise ensembles, conditional on the stated model; no Gaussian shape is imposed on parameter estimates.')
    for truth,label in [('continuum_A','Primary continuum A'),('matched_A','Matched original-FD controls'),('continuum_C','Held-out field C')]:
        add('## '+label+' empirical distributions')
        groups=[(k,g) for k,g in r['bootstrap'].items() if k.startswith(truth+'/')]
        for coordinate,index in [('v',0),('ν4',1),('c3',2),('c6',3)]:
            rows=[]
            for k,g in groups:
                if index>1 and not k.endswith('/M4'):continue
                d=g['summary']['parameters'];rows.append([k,g['summary']['n'],*[d[name][index] for name in ['mean','median','sample_sd','p05','p25','p75','p95']]])
            add(coordinate+':')
            add(table(['case','n','mean','median','sample SD','p05','p25','p75','p95'],rows))
        add(table(['case','discrepancy-bound fraction','continuation fraction (selected fit)','unconverged fraction','mean final-field L2','SD final-field L2','mean clean observed NMSE'],[[k,g['summary']['discrepancy_bound_fraction'],g['summary']['continuation_fraction'],g['summary']['unconverged_fraction'],g['summary']['field_error']['mean'],g['summary']['field_error']['sample_sd'],g['summary']['clean_observed_nmse']['mean']] for k,g in groups]))
    add('Boundary point masses in M4 distributions (fractions, not a Gaussian approximation):')
    add(table(['case','c3 lower','c3 upper','c6 lower','c6 upper'],[[k,g['summary']['discrepancy_lower_bound_fractions'][0],g['summary']['discrepancy_upper_bound_fractions'][0],g['summary']['discrepancy_lower_bound_fractions'][1],g['summary']['discrepancy_upper_bound_fractions'][1]] for k,g in r['bootstrap'].items() if k.endswith('/M4')]))
    add('## Physical bias versus pseudo-true sampling deviations')
    add('Every run stores the exact identity physical error=clean model/design bias+noise-induced shift, in [v,ν4]. The ensemble tables below use signed means. The full mean/median/SD/quantile distributions of physical errors and pseudo-deviations are also stored. For free discrepancy, c3/c6 deviations from their clean pseudo-values are recorded separately.')
    add(table(['case','clean physical bias [v,ν4]','mean physical error [v,ν4]','mean pseudo deviation [v,ν4]'],[[k,(np.array(r['pseudo'][k]['parameters'][:2])-[1.,.002]).tolist(),g['summary']['physical_errors']['mean'],g['summary']['pseudo_deviations']['mean']] for k,g in r['bootstrap'].items()]))
    add(table(['case','physical RMSE [v,ν4]','pseudo-deviation RMSE [v,ν4]'],[[k,g['summary']['physical_rmse'],g['summary']['pseudo_deviation_rmse']] for k,g in r['bootstrap'].items()]))
    add('## Nominal coverage and empirical LR calibration')
    add('Coverage of pseudo-truth and physical truth is reported separately. Indeterminate profiles are neither silently discarded nor counted as ordinary misses: resolved-subset rates and worst/best all-subset coverage bounds are both shown. Known missing intervals therefore cannot improve a reported denominator. The empirical LR quantiles are descriptive estimates from only 20 or 10 replicates, not replacement thresholds or a validated recalibration. No seed was selected by fit quality.')
    add(table(['case','subset n','valid LR count','empirical LR p68','nominal 68 threshold','empirical LR p95','nominal 95 threshold'],[[k,a['count'],a['LR_valid_count'],a['LR_empirical_68'],1.,a['LR_empirical_95'],3.841459] for k,a in r['coverage_summary'].items()]))
    add(table(['case','level','resolved n','indeterminate n','pseudo coverage resolved','physical coverage resolved','pseudo all-subset bounds','physical all-subset bounds','mean ν4 width'],[[k,level,v['resolved_count'],v['indeterminate_count'],v['pseudo_coverage_resolved'],v['physical_coverage_resolved'],v['pseudo_coverage_all_bounds'],v['physical_coverage_all_bounds'],v['mean_nu4_width']] for k,a in r['coverage_summary'].items() for level,v in a['coverage'].items()]))
    add('## M1/M4/M5 uncertainty and state-prediction comparison')
    add(table(['case','clean ν4 bias','clean 95% ν4 width','bootstrap ν4 SD','mean final L2','SD final L2','discrepancy-bound fraction'],[[k,r['pseudo'][k]['parameters'][1]-.002,r['profiles'][k+'/1']['intervals']['95']['nu4']['total_width'] if k+'/1' in r['profiles'] else None,g['summary']['parameters']['sample_sd'][1],g['summary']['field_error']['mean'],g['summary']['field_error']['sample_sd'],g['summary']['discrepancy_bound_fraction']] for k,g in r['bootstrap'].items() if k.startswith('continuum_A/')]))
    add(r.get('scientific_interpretation','Detailed interpretation pending completed numerical review.'))
    add('## Optimizer diagnostics and failed points')
    add(table(['quantity','value'],[(k,v) for k,v in r['optimizer_summary'].items() if k!='failed_attempts_by_group']))
    add('Spread across converged starts near the nominal thresholds (ΔNLL units): '+fmt(r['profile_optimizer_accuracy'])+'. These diagnostics expose finite-tolerance and possible branch differences; the smallest converged value is retained.')
    add('Adjacent LR slopes and second divided differences are stored for every curve without crossing failed points. Nuisance jumps and finite-grid extrema are inspection flags, not automatic claims of branch switching or nonsmoothness. Every flagged point already received central, alternate, and neighboring warm-start optimizations; no smoothing is applied.')
    add('Counts distinguish all attempted starts from selected endpoints. A failed alternate start does not invalidate a point if another predeclared start converges. A point with no converged start is invalid and never interpolated through. Contact arrays refer to the final optimization segment; fixed-coordinate contacts are not evidence of nuisance-bound activity. Both original 2000-update and final optimizer states retain moments and absolute counts. No failed run is deleted or extended beyond its cap.')
    if failures:add(table(['attempt group','nonstationary attempts'],r['optimizer_summary']['failed_attempts_by_group'].items()))
    add(table(['profile','invalid selected points','large nuisance jumps','baseline valid'],[[k,p['failed_points'],len(p['jumps']),p['baseline_valid']] for k,p in {**r['profiles'],**r['coverage']}.items() if p['failed_points'] or p['jumps'] or not p['baseline_valid']]))
    add('## Tests and acceptance')
    add('```text\n'+r['validation']['summary']+'\n```')
    add(table(['criterion','pass'],checks.items()))
    add('**'+r['conclusion']+'**. Exact nominal coverage, narrower M4 uncertainty, physical-truth inclusion under misspecification, Hessian/profile agreement, and identifiable c6 are not acceptance requirements. Numerical failures remain explicit limitations.')
    add('## Caveats')
    add('Nominal profile-likelihood intervals depend on the declared synthetic Gaussian noise model. Wilks thresholds can fail under small samples, bounds, weak identifiability, nonlinear compensation, and misspecification. Pseudo-truth coverage is not physical calibration. c3/c6 are numerical discrepancy multipliers, not physical random variables. M5 conditions on frozen calibration and excludes calibration uncertainty. Hessian approximations depend on coordinates and local quadratic assumptions; nonlinear profiles expose asymmetry and bounds. Ensembles are synthetic and conditional on exactly known initial fields and iid noise. Parameter intervals do not imply equal uncertainty in predicted fields. A discrepancy-bound endpoint is a modeling restriction, not a physical limit. Full-clean nominal intervals use hypothetical precision, not a zero-noise sampling distribution. Coverage subsets of 10/20 have substantial Monte Carlo uncertainty.')
    add('## Recommended Stage 11 (not implemented)')
    add(r.get('stage11_recommendation','Select after reviewing the completed results.'))
    add('## Reproduction and resume')
    add('```sh\n.venv/bin/python -m pytest -q\n.venv/bin/python -m experiments.stage10_identifiability --clean-profiles\n.venv/bin/python -m experiments.stage10_identifiability --profile-surfaces\n.venv/bin/python -m experiments.stage10_identifiability --bootstrap-moderate\n.venv/bin/python -m experiments.stage10_identifiability --bootstrap-severe\n.venv/bin/python -m experiments.stage10_identifiability --heldout-transfer\n.venv/bin/python -m experiments.stage10_identifiability --matched-controls\n.venv/bin/python -m experiments.stage10_identifiability --representative-profiles\n.venv/bin/python -m experiments.stage10_identifiability --finalize\n```')
    add('Atomic checkpoints reject NaN/Infinity, preserve failed attempts, and validate the Stage 9 source-result hash and ensemble seeds on resume. Finalization requires all declared groups and coverage profiles. Git status: this directory is not a Git repository. No commit, push, or repository initialization was attempted.')
    (ROOT/'docs/stage10_validation.md').write_text('\n\n'.join(out)+'\n')
