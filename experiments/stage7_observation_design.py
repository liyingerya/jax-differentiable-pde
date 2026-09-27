"""Sensitivity-only design selection, followed by held-out recovery evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import jax
import jax.numpy as jnp

from src.design import (temporal_schedules,candidate_layouts,sensitivity_jacobian,
                        information_matrix,information_metrics,rank_designs,rank_correlation)
from src.joint_inverse import (predict_joint_observations,joint_inverse_loss,
                               projected_adam_kernel)
from src.observations import add_measurement_noise
from experiments.stage3_parameter_inference import continuum_observations,sampled_range_stability

N,DT,STEPS=32,2/256,256
CANDIDATE_TIMES=jnp.arange(0,257,32)
PRIMARY_SEEDS=tuple(range(300,320))
ASSOCIATION_SEEDS=tuple(range(320,325))
ITERATIONS=2000
RESULT=Path(__file__).resolve().parents[1]/'docs/stage7_results.json'


def setup():
    x=jnp.linspace(0.,2*jnp.pi,N,endpoint=False)
    return x,jnp.sin(x)+.5*jnp.sin(2*x)+.25*jnp.cos(3*x),jnp.array([1.,jnp.log(.002)])


def full_sensitivity(theta,psi0):
    prediction=lambda t:predict_joint_observations(t,psi0,DT,STEPS,2*jnp.pi/N,CANDIDATE_TIMES)
    return jax.jit(jax.jacfwd(prediction))(theta)


def pool_scores(full_jacobian,times,layouts,scaling):
    masks=jnp.array([a['sensors'] for a in layouts])
    times=jnp.array(times[1:])//32
    samples=full_jacobian[times][:,masks,:].transpose(1,0,2,3).reshape(len(layouts),-1,2)
    scaled=samples*jnp.array(scaling)
    grams=jnp.einsum('bmi,bmj->bij',scaled,scaled)
    eig=jnp.linalg.eigvalsh(grams)
    scores=[]
    for matrix,(lo,hi) in zip(grams.tolist(),eig.tolist()):
        pd=lo>100*2.220446049250313e-16*max(abs(hi),1e-30)
        scores.append({'matrix':matrix,'eigenvalues':[lo,hi],'lambda_min':lo,'lambda_max':hi,
                       'condition_number':hi/lo if pd else None,'determinant':lo*hi,
                       'log_determinant':math.log(lo)+math.log(hi) if pd else None,
                       'positive_definite':pd})
    return scores


def score_design(full_jacobian,design,scaling=(1.,1.)):
    return pool_scores(full_jacobian,design['time_indices'],[{'sensors':design['sensor_indices']}],scaling)[0]


def selected_design(times,layout):
    return {'time_indices':list(times),'times':[i*DT for i in times],
            'sensor_indices':layout['sensors'],'layout_id':layout['id'],'layout_seed':layout['seed']}


def freeze_hash(selected):
    return hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()


def build_designs():
    x,psi0,theta=setup()
    jac=full_sensitivity(theta,psi0)
    rms=jnp.sqrt(jnp.mean(jac[1:]**2,axis=(0,1)))
    normalized_scale=(1/rms).tolist()
    scales={'primary':[1.,1.],'column_normalized':normalized_scale}
    schedules=temporal_schedules()
    r={'configuration':{'N':N,'dt':DT,'num_steps':STEPS,'nominal_v':1.,'nominal_nu4':.002,
        'primary_scaling':[1.,1.],'full_observation_column_rms':rms.tolist(),
        'normalized_scaling':normalized_scale,'time_top_K':5,'random_candidates_per_budget':500,
        'pool_seed_starts':{'8':10000,'4':20000},'evaluation_noise_seeds':PRIMARY_SEEDS,
        'association_noise_seeds':ASSOCIATION_SEEDS,'association_ranks':'top 10, middle 10, bottom 10 in the 4-sensor baseline-time E ranking',
        'primary_metric':'maximize lambda_min of unweighted sensitivity Gram in (v,q)',
        'iterations':ITERATIONS,'learning_rate':.05,'v_bounds':[.5,1.5],'nu4_bounds':[1e-4,.02]},
       'temporal':[],'spatial':{},'selected':{},'neighborhood':[],
       'representative_sensitivities':{},'recovery':{},'experiment_conclusion':'DESIGNS FROZEN; RECOVERY PENDING'}
    all_sensors=[{'sensors':list(range(N))}]
    for times in schedules:
        r['temporal'].append({'time_indices':list(times),'times':[i*DT for i in times],
             **{name:pool_scores(jac,times,all_sensors,scale)[0] for name,scale in scales.items()}})
    temporal_rank=rank_designs([a['primary'] for a in r['temporal']])
    r['temporal_rankings']={name:{criterion:rank_designs([a[name] for a in r['temporal']],criterion)
                                 for criterion in ('E','D','condition')} for name in scales}
    for count,seed_start in ((8,10000),(4,20000)):
        layouts=candidate_layouts(count,seed_start=seed_start)
        entry={'layouts':layouts,'fixed_time_indices':[0,128,256],
               'fixed_scores':{name:pool_scores(jac,(0,128,256),layouts,scale) for name,scale in scales.items()},
               'joint_scores':[]}
        entry['fixed_rankings']={name:{criterion:rank_designs(entry['fixed_scores'][name],criterion)
                                      for criterion in ('E','D','condition')} for name in scales}
        for tid in temporal_rank[:5]:
            times=schedules[tid]
            entry['joint_scores'].append({'temporal_id':tid,'time_indices':list(times),
                **{name:pool_scores(jac,times,layouts,scale) for name,scale in scales.items()}})
        best_j,best_layout=max(((j,i) for j in range(5) for i in range(len(layouts))),
                              key=lambda pair:entry['joint_scores'][pair[0]]['primary'][pair[1]]['lambda_min'])
        selected={'baseline':selected_design((0,128,256),layouts[0]),
            'evenly_spaced':selected_design((0,128,256),layouts[1]),
            'spatial_E':selected_design((0,128,256),layouts[entry['fixed_rankings']['primary']['E'][0]]),
            'joint_E':selected_design(entry['joint_scores'][best_j]['time_indices'],layouts[best_layout])}
        entry['selected_joint_indices']=[best_j,best_layout]
        # Secondary metrics/scalings are comparisons only, never recovery selectors.
        entry['secondary_joint_winners']={}
        for scale in scales:
            for criterion in ('E','D','condition'):
                candidates=[]
                for j,table in enumerate(entry['joint_scores']):
                    i=rank_designs(table[scale],criterion)[0]
                    value=table[scale][i][{'E':'lambda_min','D':'log_determinant','condition':'condition_number'}[criterion]]
                    candidates.append((j,i,value if criterion!='condition' else -value))
                j,i,_=max(candidates,key=lambda a:a[2])
                entry['secondary_joint_winners'][scale+'_'+criterion]=selected_design(entry['joint_scores'][j]['time_indices'],layouts[i])
        for design in selected.values():
            design['scores']={name:score_design(jac,design,scale) for name,scale in scales.items()}
        r['selected'][str(count)]=selected; r['spatial'][str(count)]=entry
        representative={'good_joint':selected['joint_E'],'good_spatial':selected['spatial_E'],
            'baseline':selected['baseline'],'poor_spatial':selected_design((0,128,256),layouts[entry['fixed_rankings']['primary']['E'][-1]])}
        r['representative_sensitivities'][str(count)]={}
        for name,design in representative.items():
            times=jnp.array(design['time_indices'][1:])//32
            samples=jac[times][:,jnp.array(design['sensor_indices']),:]
            gram=samples.reshape(-1,2).T@samples.reshape(-1,2)
            r['representative_sensitivities'][str(count)][name]={'design':design,
                'jacobian':samples.reshape(-1,2).tolist(),
                'q_squared_sensitivity_by_time':jnp.sum(samples[:,:,1]**2,axis=1).tolist(),
                'v_squared_sensitivity':float(gram[0,0]),'q_squared_sensitivity':float(gram[1,1]),
                'column_cosine':float(gram[0,1]/jnp.sqrt(gram[0,0]*gram[1,1]))}
    # Freeze the association subset by score rank before any inverse recovery.
    order=r['spatial']['4']['fixed_rankings']['primary']['E']; middle=len(order)//2
    r['association_selection']={'top':order[:10],'middle':order[middle-5:middle+5],'bottom':order[-10:]}
    # Selected-Jacobian verification at the nominal design point.
    checks=[]
    for count in (8,4):
        d=r['selected'][str(count)]['joint_E']; times=jnp.array(d['time_indices'][1:]); sensors=jnp.array(d['sensor_indices'])
        prediction=lambda t:predict_joint_observations(t,psi0,DT,STEPS,2*jnp.pi/N,times,sensors).reshape(-1)
        j=sensitivity_jacobian(theta,psi0,DT,STEPS,2*jnp.pi/N,times,sensors)
        for col,eps in ((0,1e-6),(1,1e-5)):
            perturb=jnp.zeros(2).at[col].set(eps)
            fd=(prediction(theta+perturb)-prediction(theta-perturb))/(2*eps)
            checks.append({'sensors':count,'coordinate':['v','q'][col],'epsilon':eps,
                'max_absolute_difference':float(jnp.max(jnp.abs(j[:,col]-fd))),
                'relative_l2_difference':float(jnp.linalg.norm(j[:,col]-fd)/jnp.linalg.norm(fd))})
    r['sensitivity_checks']=checks
    for v in (.9,1.,1.1):
        for nu in (.0015,.002,.0025):
            nearby=full_sensitivity(jnp.array([v,jnp.log(nu)]),psi0)
            for count in (8,4):
                row={'v':v,'nu4':nu,'sensors':count}
                for name in ('baseline','joint_E'):
                    row[name]={scale:score_design(nearby,r['selected'][str(count)][name],factors) for scale,factors in scales.items()}
                r['neighborhood'].append(row)
    # Modal sensitivity norms interpret high-k contributions without another solver.
    r['modal_sensitivity']={}
    for k,initial in ((1,jnp.sin(x)),(2,.5*jnp.sin(2*x)),(3,.25*jnp.cos(3*x))):
        modal=full_sensitivity(theta,initial)
        r['modal_sensitivity'][str(k)]={}
        for count in (8,4):
            d=r['selected'][str(count)]['joint_E']
            samples=modal[jnp.array(d['time_indices'][1:])//32][:,jnp.array(d['sensor_indices']),:]
            r['modal_sensitivity'][str(k)][str(count)]={'q_column_l2':float(jnp.linalg.norm(samples[:,:,1]))}
    r['frozen_selection_hash']=freeze_hash(r['selected'])
    RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print('Designs frozen:',r['frozen_selection_hash'],flush=True)
    for count,designs in r['selected'].items():
        print(count,{name:(d['times'],d['sensor_indices'],d['scores']['primary']['lambda_min']) for name,d in designs.items()},flush=True)
    return r


def make_batch_fitter(iterations=ITERATIONS,learning_rate=.05):
    """Independent vmapped fits reuse the extracted, unchanged Stage 6 kernel."""
    _,psi0,_=setup()
    lower=jnp.array([.5,jnp.log(1e-4)]); upper=jnp.array([1.5,jnp.log(.02)])
    initial=jnp.array([.7,jnp.log(.0005)])
    def one(data,times,sensors):
        loss=lambda theta:joint_inverse_loss(theta,psi0,DT,STEPS,2*jnp.pi/N,times,data,sensors)
        fit=projected_adam_kernel(loss,initial,lower,upper,iterations,learning_rate)
        theta=fit['theta']; gradient=fit['gradient_history'][-1]
        residual=jnp.where((theta<=lower+1e-8)&(gradient>=0),0.,gradient)
        residual=jnp.where((theta>=upper-1e-8)&(gradient<=0),0.,residual)
        final=predict_joint_observations(theta,psi0,DT,STEPS,2*jnp.pi/N,jnp.array([256]))[0]
        return {'adam_first_moment':fit['optimizer_state']['first_moment'],
                'adam_second_moment':fit['optimizer_state']['second_moment'],
                'adam_update_count':fit['optimizer_state']['update_count'],
                'theta':theta,'gradient':gradient,'kkt_residual':jnp.linalg.norm(residual),
                'loss':fit['loss_history'][-1],'initial_loss':fit['loss_history'][0],
                'proposal_hits':fit['proposal_boundary_hits'],
                'bound_contacts':fit['parameter_bound_contacts'],'final_field':final,
                'finite':jnp.all(jnp.isfinite(fit['loss_history'])) & jnp.all(jnp.isfinite(fit['gradient_history'])),
                'parameter_min':jnp.min(fit['theta_history'],axis=0),'parameter_max':jnp.max(fit['theta_history'],axis=0)}
    return jax.jit(jax.vmap(one))


def summary(rows):
    r={'count':len(rows)}
    for parameter in ('v','nu4'):
        values=[a[parameter] for a in rows]; errors=[a['relative_'+parameter+'_error'] for a in rows]
        r['mean_'+parameter]=statistics.mean(values); r['sd_'+parameter]=statistics.stdev(values) if len(values)>1 else None
        r['mean_relative_'+parameter+'_error']=statistics.mean(errors)
        r['median_relative_'+parameter+'_error']=statistics.median(errors)
        r['p90_relative_'+parameter+'_error']=float(jnp.quantile(jnp.array(errors),.9))
        r['min_'+parameter]=min(values); r['max_'+parameter]=max(values)
    r['mean_clean_field_l2']=statistics.mean(a['clean_field_l2'] for a in rows)
    r['bound_rate']=statistics.mean(a['at_bound'] for a in rows)
    r['max_kkt_residual']=max(a['kkt_residual'] for a in rows)
    r['nonconverged_count']=sum(a['kkt_residual']>=1e-7 for a in rows)
    return r


def run_recovery(r):
    assert freeze_hash(r['selected'])==r['frozen_selection_hash']
    x,psi0,theta=setup()
    truth=predict_joint_observations(theta,psi0,DT,STEPS,2*jnp.pi/N,CANDIDATE_TIMES)
    continuum=continuum_observations(x,CANDIDATE_TIMES*DT,1.,.002)
    fit_batch=make_batch_fitter()
    def save(): RESULT.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    amplification=max(sampled_range_stability(float(v),2*jnp.pi/N,DT,(1e-4,.02)) for v in jnp.linspace(.5,1.5,21))
    r['stability_max_amplification']=amplification
    if amplification>1+1e-12: raise RuntimeError('Unstable parameter box')
    def evaluate(key,designs,seeds,noise,model='matched'):
        print('Starting',key,'fits=',len(designs)*len(seeds),flush=True)
        rows=[]; meta=[]; arrays=[]; times=[]; masks=[]
        clean_full=truth if model=='matched' else continuum
        for label,design in designs:
            indices=jnp.array(design['time_indices']); sensors=jnp.array(design['sensor_indices'])
            clean=clean_full[indices//32][:,sensors]
            for seed in seeds:
                data,diagnostics=add_measurement_noise(clean,indices,noise,seed)
                arrays.append(data); times.append(indices); masks.append(sensors)
                meta.append({'design':label,'noise_seed':seed,'sigma_rel':noise,'truth_model':model,
                             'time_indices':design['time_indices'],'sensor_indices':design['sensor_indices'],**diagnostics})
        # Bounded batch sizes limit compiled state/adjoint memory. Same kernel for all.
        for start in range(0,len(arrays),20):
            outputs=fit_batch(jnp.stack(arrays[start:start+20]),jnp.stack(times[start:start+20]),jnp.stack(masks[start:start+20]))
            outputs={key:value.tolist() for key,value in outputs.items()}
            for i,m in enumerate(meta[start:start+20]):
                if not outputs['finite'][i]: raise FloatingPointError('Nonfinite recovery')
                v,q=outputs['theta'][i]; nu=math.exp(q)
                field=jnp.array(outputs['final_field'][i]); exact=clean_full[-1]
                row={**m,'v':v,'q':q,'nu4':nu,'relative_v_error':abs(v-1),
                     'relative_nu4_error':abs(nu-.002)/.002,'signed_v_error':v-1,'signed_nu4_error':nu-.002,
                     'final_loss':outputs['loss'][i],'initial_loss':outputs['initial_loss'][i],
                     'gradient':outputs['gradient'][i],'kkt_residual':outputs['kkt_residual'][i],
                     'proposal_hits':outputs['proposal_hits'][i],'bound_contacts':outputs['bound_contacts'][i],
                     'at_bound':bool(v<=.5+1e-8 or v>=1.5-1e-8 or nu<=1e-4*(1+1e-8) or nu>=.02*(1-1e-8)),
                     'parameter_min':outputs['parameter_min'][i],'parameter_max':outputs['parameter_max'][i],
                     'clean_field_l2':float(jnp.linalg.norm(field-exact)/jnp.linalg.norm(exact))}
                row['optimizer_state']={'theta':outputs['theta'][i],
                    'first_moment':outputs['adam_first_moment'][i],
                    'second_moment':outputs['adam_second_moment'][i],
                    'update_count':outputs['adam_update_count'][i]}
                rows.append(row)
        result={'runs':rows,'by_design':{label:summary([a for a in rows if a['design']==label]) for label,_ in designs}}
        r['recovery'][key]=result; save()
        print(key,result['by_design'],flush=True)
        return result
    # Primary design families and paired standardized noise, frozen before recovery.
    for count,noise in ((8,.02),(4,.05)):
        for name,d in r['selected'][str(count)].items():
            evaluate(f'matched_{count}_{name}',[(name,d)],PRIMARY_SEEDS,noise)
    # Score/error association uses a separately declared pool and held-out seeds.
    entry=r['spatial']['4']
    for tier,indices in r['association_selection'].items():
        designs=[(f'candidate_{i}',selected_design((0,128,256),entry['layouts'][i])) for i in indices]
        evaluate('association_'+tier,designs,ASSOCIATION_SEEDS,.05)
    scores=[]; errors=[]; r['association_pairs']=[]
    for tier,indices in r['association_selection'].items():
        for i in indices:
            score=entry['fixed_scores']['primary'][i]['lambda_min']
            error=r['recovery']['association_'+tier]['by_design'][f'candidate_{i}']['median_relative_nu4_error']
            scores.append(score); errors.append(error)
            r['association_pairs'].append({'tier':tier,'candidate_index':i,'score':score,'median_nu4_error':error})
    r['association_rank_correlation']=rank_correlation(scores,errors)
    for name in ('baseline','joint_E'):
        evaluate('continuum_'+name,[(name,r['selected']['8'][name])],PRIMARY_SEEDS,.02,'continuum')
    previous=json.loads((RESULT.parent/'stage6_results.json').read_text())['runs']['continuum_clean']
    r['continuum_reference']={'v':previous['v'],'nu4':previous['nu4']}
    for name in ('baseline','joint_E'):
        group=r['recovery']['continuum_'+name]
        for row in group['runs']:
            row['observation_shift_v']=row['v']-previous['v']
            row['observation_shift_nu4']=row['nu4']-previous['nu4']
        group['mean_observation_shift_v']=statistics.mean(a['observation_shift_v'] for a in group['runs'])
        group['mean_observation_shift_nu4']=statistics.mean(a['observation_shift_nu4'] for a in group['runs'])
    all_runs=[a for group in r['recovery'].values() for a in group['runs']]
    checks={'sensitivity_checks':all(a['relative_l2_difference']<1e-5 for a in r['sensitivity_checks']),
        '28_schedules':len(r['temporal'])==28,'frozen_designs':freeze_hash(r['selected'])==r['frozen_selection_hash'],
        'exact_budgets':all(len(d['sensor_indices'])==int(count) and len(d['time_indices'])==3 for count,designs in r['selected'].items() for d in designs.values()),
        'held_out_noise':all(a['noise_seed']>=300 for a in all_runs),
        'stationary':all(a['kkt_residual']<1e-7 for a in all_runs),
        'stable_bounds':all(a['parameter_min'][0]>=.5-1e-12 and a['parameter_max'][0]<=1.5+1e-12 and a['parameter_min'][1]>=math.log(1e-4)-1e-12 and a['parameter_max'][1]<=math.log(.02)+1e-12 for a in all_runs)}
    r['checks']=checks; r['experiment_conclusion']='PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    from experiments.stage7_convergence_hardening import harden_results
    harden_results(r)
    save(); print('Final checks:',r['hardening']['checks'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--recover-frozen',action='store_true')
    parser.add_argument('--harden-existing',action='store_true')
    args=parser.parse_args()
    results=json.loads(RESULT.read_text()) if args.recover_frozen or args.harden_existing else build_designs()
    if args.harden_existing:
        from experiments.stage7_convergence_hardening import harden_results
        harden_results(results)
        RESULT.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    elif not args.design_only:
        run_recovery(results)
