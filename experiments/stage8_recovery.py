"""Held-out Stage 8 fits; this module never searches or selects designs."""
import copy
import math
import statistics

import jax
import jax.numpy as jnp
import numpy as np

from src.joint_inverse import projected_adam_kernel, joint_inverse_loss, predict_joint_observations
from src.observations import add_measurement_noise
from src.robust_design import INITIAL_MODES, initial_field, continuum_predict, frozen_hash, bias_decomposition
from src.design import rank_correlation
from experiments.stage7_convergence_hardening import kkt_residual, THRESHOLD
from experiments.stage8_robust_design import N, DT, STEPS, TIMES, save, design_at

TRUTHS = {
    'matched_A': {'model':'fd','initial':'A','v':1.,'nu4':.002},
    'continuum_A': {'model':'continuum','initial':'A','v':1.,'nu4':.002},
    'continuum_C': {'model':'continuum','initial':'C','v':1.,'nu4':.002},
    'continuum_offnominal': {'model':'continuum','initial':'A','v':1.05,'nu4':.0018},
}


def make_fitter(continuation=False, updates=2000):
    """Same Stage 7 state-preserving Adam equations and fixed 2000-step blocks."""
    lower=jnp.array([.5,jnp.log(1e-4)]);upper=jnp.array([1.5,jnp.log(.02)])
    initial=jnp.array([.7,jnp.log(.0005)])
    def one(data,psi0,times,sensors,state):
        loss=lambda theta:joint_inverse_loss(theta,psi0,DT,STEPS,2*jnp.pi/N,times,data,sensors)
        fit=projected_adam_kernel(loss,initial,lower,upper,updates,.05,
                                  optimizer_state=state if continuation else None)
        theta=fit['theta']; gradient=fit['gradient_history'][-1]
        final=predict_joint_observations(theta,psi0,DT,STEPS,2*jnp.pi/N,jnp.array([256]))[0]
        return {'state':fit['optimizer_state'],'gradient':gradient,
                'kkt_residual':kkt_residual(theta,gradient,lower,upper),
                'loss':fit['loss_history'][-1],'final_field':final,
                'parameter_min':jnp.min(fit['theta_history'],axis=0),
                'parameter_max':jnp.max(fit['theta_history'],axis=0),
                'proposal_hits':fit['proposal_boundary_hits'],
                'finite':jnp.all(jnp.isfinite(fit['loss_history'])) & jnp.all(jnp.isfinite(fit['gradient_history']))}
    return jax.jit(jax.vmap(one))


def truth_data(definition):
    x=jnp.arange(N)*2*jnp.pi/N
    psi0=initial_field(x,definition['initial'])
    theta=jnp.array([definition['v'],jnp.log(definition['nu4'])])
    values=(predict_joint_observations(theta,psi0,DT,STEPS,2*jnp.pi/N,TIMES)
            if definition['model']=='fd' else continuum_predict(theta,x,TIMES*DT,INITIAL_MODES[definition['initial']]))
    return psi0,values


def fit_metrics(output, truth, clean_final):
    v,q=map(float,output['state']['theta']);nu=math.exp(q)
    return {'v':v,'q':q,'nu4':nu,'kkt_residual':float(output['kkt_residual']),
            'gradient':np.asarray(output['gradient']).tolist(),'loss':float(output['loss']),
            'relative_v_error':abs(v-truth['v'])/abs(truth['v']),
            'relative_nu4_error':abs(nu-truth['nu4'])/truth['nu4'],
            'clean_field_l2':float(np.linalg.norm(np.asarray(output['final_field'])-np.asarray(clean_final))/np.linalg.norm(np.asarray(clean_final))),
            'at_bound':bool(v<=.5+1e-8 or v>=1.5-1e-8 or nu<=1e-4*(1+1e-8) or nu>=.02*(1-1e-8)),
            'proposal_hits':np.asarray(output['proposal_hits']).tolist(),
            'parameter_min':np.asarray(output['parameter_min']).tolist(),
            'parameter_max':np.asarray(output['parameter_max']).tolist(),
            'optimizer_state':jax.tree_util.tree_map(lambda a:np.asarray(a).tolist(),output['state'])}


def summary(rows):
    fits=[a['final_fit'] for a in rows];s={'count':len(rows)}
    for parameter in ('v','nu4'):
        values=[a[parameter] for a in fits];errors=[a['relative_'+parameter+'_error'] for a in fits]
        s['mean_'+parameter]=statistics.mean(values);s['sd_'+parameter]=statistics.stdev(values) if len(values)>1 else 0.
        s['mean_relative_'+parameter+'_error']=statistics.mean(errors)
        s['median_relative_'+parameter+'_error']=statistics.median(errors)
        s['p90_relative_'+parameter+'_error']=float(np.quantile(errors,.9))
    s['mean_clean_field_l2']=statistics.mean(a['clean_field_l2'] for a in fits)
    s['bound_rate']=statistics.mean(a['at_bound'] for a in fits)
    s['trajectory_bound_hit_rate']=statistics.mean(any(a['proposal_hits']) for a in fits)
    s['continuation_rate']=statistics.mean(a['continuation']['required'] for a in rows)
    s['unconverged_rate']=statistics.mean(a['kkt_residual']>=THRESHOLD for a in fits)
    s['maximum_kkt_residual']=max(a['kkt_residual'] for a in fits)
    if 'physical_decomposition' in rows[0]:
        for coord in ('physical_decomposition','theta_decomposition'):
            s[coord]={key:np.mean([a[coord][key] for a in rows],axis=0).tolist()
                      for key in ('physical_error','clean_model_design_bias','noise_induced_shift')}
    return s


def run_recovery(r):
    assert frozen_hash(r['frozen'])==r['frozen_hash']
    assert frozen_hash(r['scores'])==r['rankings_hash']
    initial_fitter=make_fitter();continuation_fitter=make_fitter(True)
    r['truth_families']=TRUTHS
    r.setdefault('clean_optima',{});r.setdefault('diagnostic_clean_optima',{})
    truth_cache={key:truth_data(d) for key,d in TRUTHS.items()}
    def fit_group(definition,design,seeds,noise):
        psi0,values=truth_cache[definition]
        truth=TRUTHS[definition]
        times=jnp.array(design['time_indices']);mask=jnp.array(design['sensor_indices'])
        clean=values[times//32][:,mask]
        arrays=[];metadata=[]
        for seed in seeds:
            data,details=add_measurement_noise(clean,times,noise,seed)
            arrays.append(data);metadata.append({'noise_seed':seed,'noise_fraction':noise,**details})
        data=jnp.stack(arrays);initials=jnp.repeat(psi0[None],len(seeds),axis=0)
        ts=jnp.repeat(times[None],len(seeds),axis=0);masks=jnp.repeat(mask[None],len(seeds),axis=0)
        dummy={'theta':jnp.zeros((len(seeds),2)),'first_moment':jnp.zeros((len(seeds),2)),
               'second_moment':jnp.zeros((len(seeds),2)),'update_count':jnp.zeros(len(seeds),dtype=int)}
        outputs=initial_fitter(data,initials,ts,masks,dummy)
        outputs=jax.tree_util.tree_map(np.asarray,outputs)
        if not np.all(outputs['finite']):raise FloatingPointError('Nonfinite initial fit')
        failed=np.flatnonzero(outputs['kkt_residual']>=THRESHOLD)
        continued={}
        if len(failed):
            states=jax.tree_util.tree_map(lambda a:jnp.asarray(a[failed]),outputs['state'])
            next_outputs=continuation_fitter(data[failed],initials[failed],ts[failed],masks[failed],states)
            next_outputs=jax.tree_util.tree_map(np.asarray,next_outputs)
            if not np.all(next_outputs['finite']):raise FloatingPointError('Nonfinite continuation')
            for j,i in enumerate(failed):continued[int(i)]=jax.tree_util.tree_map(lambda a:a[j],next_outputs)
        rows=[]
        for i,meta in enumerate(metadata):
            original=jax.tree_util.tree_map(lambda a:a[i],outputs)
            final=continued.get(i,original)
            first=fit_metrics(original,truth,values[-1]);last=fit_metrics(final,truth,values[-1])
            if i in continued:
                last['proposal_hits']=(np.array(first['proposal_hits'])+np.array(last['proposal_hits'])).tolist()
                last['parameter_min']=np.minimum(first['parameter_min'],last['parameter_min']).tolist()
                last['parameter_max']=np.maximum(first['parameter_max'],last['parameter_max']).tolist()
            rows.append({**meta,'initial_fit':first,'final_fit':last,
                         'continuation':{'initial_budget':2000,'required':i in continued,
                            'total_updates':4000 if i in continued else 2000,
                            'final_kkt_residual':last['kkt_residual'],
                            'parameter_changes':{p:last[p]-first[p] for p in ('v','q','nu4')}}})
        return rows
    def decompose(rows,clean,definition):
        truth=TRUTHS[definition]
        for row in rows:
            f=row['final_fit']
            row['physical_decomposition']=bias_decomposition([f['v'],f['nu4']],[clean['v'],clean['nu4']],[truth['v'],truth['nu4']])
            row['theta_decomposition']=bias_decomposition([f['v'],f['q']],[clean['v'],clean['q']],[truth['v'],math.log(truth['nu4'])])
    # Compute every primary design-specific zero-noise optimum before noisy fits.
    for count,designs in r['frozen']['selected'].items():
        for truth in TRUTHS:
            for name,d in designs.items():
                key=f'{count}/{truth}/{name}'
                if key in r['clean_optima']:continue
                rows=fit_group(truth,d,[500],0.)
                r['clean_optima'][key]={'design':d,'truth':truth,'run':rows[0]}
                save(r);print('Clean',key,rows[0]['final_fit']['kkt_residual'],flush=True)
    for count,designs in r['frozen']['selected'].items():
        for truth in TRUTHS:
            for name,d in designs.items():
                key=f'{count}/{truth}/{name}'
                if key in r['recovery']:continue
                print('Starting noisy',key,flush=True)
                rows=fit_group(truth,d,r['frozen']['primary_seeds'],.02 if count=='8' else .05)
                decompose(rows,r['clean_optima'][key]['run']['final_fit'],truth)
                r['recovery'][key]={'design':d,'truth':truth,'runs':rows,'summary':summary(rows)}
                save(r);print('Finished',key,r['recovery'][key]['summary'],flush=True)
    # Predeclared nine-design diagnostic. Its clean references also precede noise.
    diagnostic={}
    for tier,indices in r['frozen']['diagnostic_selection'].items():
        for index in indices:
            diagnostic[str(index)]={'tier':tier,'design':design_at(index,r['frozen']['schedules'],r['frozen']['candidate_pools']['4'])}
    for index,item in diagnostic.items():
        for truth in ('matched_A','continuum_A'):
            key=f'{truth}/{index}'
            if key not in r['diagnostic_clean_optima']:
                rows=fit_group(truth,item['design'],[530],0.)
                r['diagnostic_clean_optima'][key]={'run':rows[0],**item}
                save(r);print('Diagnostic clean',key,flush=True)
    for index,item in diagnostic.items():
        for truth in ('matched_A','continuum_A'):
            key=f'{truth}/{index}'
            if key in r['diagnostic_recovery']:continue
            rows=fit_group(truth,item['design'],r['frozen']['diagnostic_seeds'],.05)
            decompose(rows,r['diagnostic_clean_optima'][key]['run']['final_fit'],truth)
            r['diagnostic_recovery'][key]={'runs':rows,'summary':summary(rows),**item}
            save(r);print('Diagnostic noisy',key,flush=True)
    finalize(r)
    save(r)


def finalize(r):
    r['worst_case_recovery']={}
    for count,designs in r['frozen']['selected'].items():
        r['worst_case_recovery'][count]={}
        for name in designs:
            groups=[r['recovery'][f'{count}/{truth}/{name}']['summary'] for truth in TRUTHS]
            r['worst_case_recovery'][count][name]={metric:max(g[metric] for g in groups) for metric in
                ('mean_relative_nu4_error','p90_relative_nu4_error','mean_relative_v_error','mean_clean_field_l2')}
    pairs=[]
    for tier,indices in r['frozen']['diagnostic_selection'].items():
        for index in indices:
            medians={truth:r['diagnostic_recovery'][f'{truth}/{index}']['summary']['median_relative_nu4_error']
                     for truth in ('matched_A','continuum_A')}
            pairs.append({'tier':tier,'candidate_index':index,'robust_E':r['scores']['4']['weighted']['robust_E'][index],
                          'median_errors':medians,'worst_median_error':max(medians.values())})
    r['score_recovery_association']={'pairs':pairs,'spearman':rank_correlation([a['robust_E'] for a in pairs],[a['worst_median_error'] for a in pairs])}
    rows=[a for key in ('recovery','diagnostic_recovery') for g in r[key].values() for a in g['runs']]
    clean=[g['run'] for key in ('clean_optima','diagnostic_clean_optima') for g in r[key].values()]
    decomposition=all(np.allclose(np.array(a[c]['physical_error']),np.array(a[c]['clean_model_design_bias'])+a[c]['noise_induced_shift'],atol=1e-14,rtol=1e-12)
                      for a in rows for c in ('physical_decomposition','theta_decomposition'))
    checks={'continuum_derivatives':all(a['relative_l2_error']<1e-5 for a in r['continuum_derivative_checks']),
            '12_scenarios':len(r['frozen']['scenario_definitions'])==12,
            'exhaustive_candidates':all(len(r['scores'][c][k]['ranking'])==14056 for c in ('8','4') for k in ('weighted','unweighted')),
            'candidate_pool_preserved':all(a['stage7']==a['stage8'] for a in r['candidate_pool_hashes'].values()),
            'frozen_integrity':frozen_hash(r['frozen'])==r['frozen_hash'] and frozen_hash(r['scores'])==r['rankings_hash'],
            'all_primary_groups':len(r['recovery'])==48,'all_clean_references':len(clean)==66,
            'new_noise_seeds':all(500<=a['noise_seed']<=539 for a in rows),
            'decomposition_identity':decomposition,
            'uniform_continuation_policy':all(
                a['initial_fit']['optimizer_state']['update_count']==2000
                and a['continuation']['required']==(a['initial_fit']['kkt_residual']>=THRESHOLD)
                and a['continuation']['total_updates']==(4000 if a['continuation']['required'] else 2000)
                and a['final_fit']['optimizer_state']['update_count']==a['continuation']['total_updates']
                for a in rows+clean),
            'within_update_cap':all(a['continuation']['total_updates']<=4000 for a in rows+clean),
            'clean_stationarity':all(a['final_fit']['kkt_residual']<THRESHOLD for a in clean),
            'noisy_stationarity':all(a['final_fit']['kkt_residual']<THRESHOLD for a in rows),
            'bounds':all(a['final_fit']['parameter_min'][0]>=.5-1e-12 and a['final_fit']['parameter_max'][0]<=1.5+1e-12 and a['final_fit']['parameter_min'][1]>=math.log(1e-4)-1e-12 and a['final_fit']['parameter_max'][1]<=math.log(.02)+1e-12 for a in rows+clean)}
    r['checks']=checks
    r['optimizer_summary']={'clean_fits':len(clean),'noisy_fits':len(rows),
        'clean_continued':sum(a['continuation']['required'] for a in clean),
        'noisy_continued':sum(a['continuation']['required'] for a in rows),
        'clean_unconverged':sum(a['final_fit']['kkt_residual']>=THRESHOLD for a in clean),
        'noisy_unconverged':sum(a['final_fit']['kkt_residual']>=THRESHOLD for a in rows)}
    r['conclusion']='PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    print('Final',r['checks'],r['optimizer_summary'],flush=True)
