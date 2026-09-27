"""Joint recovery, two-dimensional landscapes, and local identifiability."""

import json
import math
from pathlib import Path
import statistics

import jax
import jax.numpy as jnp

from src.joint_inverse import (joint_inverse_loss,predict_joint_observations,
                               optimize_joint_parameters,hessian_diagnostics)
from src.observations import sensor_layout
from experiments.stage2_time_integration_validation import stability_limit
from experiments.stage3_parameter_inference import sampled_range_stability
from experiments.stage5_observation_robustness import prepare_case,SCHEDULES

N,STEPS,DT = 32,256,2/256
V_BOUNDS,NU_BOUNDS = (.5,1.5),(1e-4,.02)
STARTS = ((.7,.0005),(.7,.01),(1.,.0005),(1.,.01),(1.3,.0005),(1.3,.01))
SEEDS = (100,101,102,103,104)
ITERATIONS = 2000
# Declared before production; no accuracy requirement on degraded-data fits.
CLEAN_RELATIVE_TOL = 1e-4
GRADIENT_RELATIVE_TOL = 1e-5
STATIONARITY_TOL = 1e-7


def dataset(times=SCHEDULES[9],sensors=tuple(range(N)),sigma_rel=0.,seed=100,model='matched'):
    psi0,clean,data,final_truth,_,diagnostics=prepare_case(times,sensors,sigma_rel,seed,model)
    indices,mask=jnp.array(times),jnp.array(sensors)
    loss=jax.jit(lambda theta:joint_inverse_loss(theta,psi0,DT,STEPS,2*jnp.pi/N,indices,data,mask))
    return psi0,clean,final_truth,loss,diagnostics


def gradient_check(loss):
    theta=jnp.array([.9,jnp.log(.001)])
    gradient=jax.grad(loss)(theta)
    rows=[]
    # Two perturbation scales check finite-difference stability.
    for coord,epsilon in ((0,1e-6),(1,1e-5)):
        row={'coordinate': ('v','q')[coord], 'jax':float(gradient[coord]), 'estimates':[]}
        for eps in (epsilon,epsilon/2):
            perturb=jnp.zeros(2).at[coord].set(eps)
            fd=float((loss(theta+perturb)-loss(theta-perturb))/(2*eps))
            row['estimates'].append({'epsilon':eps,'finite_difference':fd,
                'absolute_difference':abs(float(gradient[coord])-fd),
                'relative_difference':abs(float(gradient[coord])-fd)/max(abs(fd),1e-15)})
        rows.append(row)
    return {'theta':theta.tolist(),'physical_pair':[.9,.001],'components':rows}


def run_fit(identifier,initial=(.7,.0005),times=SCHEDULES[9],sensors=tuple(range(N)),
            sigma_rel=0.,seed=100,model='matched',layout_seed=None):
    psi0,clean,final_truth,loss,noise=dataset(times,sensors,sigma_rel,seed,model)
    fit=optimize_joint_parameters(loss,initial,V_BOUNDS,NU_BOUNDS,iterations=ITERATIONS)
    theta=fit['theta']; velocity=float(theta[0]); nu4=float(jnp.exp(theta[1]))
    gradient=fit['gradient_history'][-1]
    lower=jnp.array([V_BOUNDS[0],math.log(NU_BOUNDS[0])])
    upper=jnp.array([V_BOUNDS[1],math.log(NU_BOUNDS[1])])
    # KKT residual respects outward-pointing gradients at active box bounds.
    residual=jnp.where((theta<=lower+1e-8)&(gradient>=0),0.,gradient)
    residual=jnp.where((theta>=upper-1e-8)&(gradient<=0),0.,residual)
    final=predict_joint_observations(theta,psi0,DT,STEPS,2*jnp.pi/N,jnp.array([STEPS]))[0]
    observed=predict_joint_observations(theta,psi0,DT,STEPS,2*jnp.pi/N,jnp.array(times),jnp.array(sensors))
    hessian=hessian_diagnostics(loss,theta)
    return {'id':identifier,'initial_v':initial[0],'initial_nu4':initial[1],
        'v':velocity,'nu4':nu4,'q':float(theta[1]),'signed_v_error':velocity-1,
        'signed_nu4_error':nu4-.002,'relative_v_error':abs(velocity-1),
        'relative_nu4_error':abs(nu4-.002)/.002,'initial_loss':float(fit['loss_history'][0]),
        'final_loss':float(fit['loss_history'][-1]),'gradient':gradient.tolist(),
        'gradient_norm':float(jnp.linalg.norm(gradient)),'kkt_residual_norm':float(jnp.linalg.norm(residual)),
        'hessian':hessian,'proposal_boundary_hits':fit['proposal_boundary_hits'].tolist(),
        'parameter_bound_contacts':fit['parameter_bound_contacts'].tolist(),
        'history_v_range':[float(jnp.min(fit['v_history'])),float(jnp.max(fit['v_history']))],
        'history_nu4_range':[float(jnp.min(fit['nu4_history'])),float(jnp.max(fit['nu4_history']))],
        'iterations':ITERATIONS,'truth_model':model,'time_indices':list(times),
        'sensor_indices':list(sensors),'layout_seed':layout_seed,'noise_seed':seed,
        'sigma_rel':sigma_rel,**noise,
        'final_clean_field_relative_l2':float(jnp.linalg.norm(final-final_truth)/jnp.linalg.norm(final_truth)),
        'clean_observation_nmse':float(jnp.mean((observed-clean)**2)/(jnp.mean(clean**2)+1e-12)),
        'histories':{key:fit[key].tolist() for key in ('v_history','nu4_history','q_history','loss_history','gradient_history')}}


def landscape(loss,velocities,q_values):
    """Forward-only 31-point row batches; one compilation reused across rows."""
    batch=jax.jit(jax.vmap(loss))
    values=jnp.stack([batch(jnp.stack((jnp.full_like(q_values,v),q_values),axis=1)) for v in velocities])
    if not bool(jnp.all(jnp.isfinite(values))):
        raise FloatingPointError('Nonfinite landscape')
    losses=values.tolist(); i,j=divmod(int(jnp.argmin(values)),len(q_values))
    minima=[]
    for a in range(1,len(velocities)-1):
        for b in range(1,len(q_values)-1):
            neighbors=[losses[c][d] for c in (a-1,a,a+1) for d in (b-1,b,b+1) if (c,d)!=(a,b)]
            if losses[a][b]<min(neighbors):
                minima.append({'v':float(velocities[a]),'nu4':float(jnp.exp(q_values[b])),'loss':losses[a][b]})
    lo,hi=float(jnp.min(values)),float(jnp.max(values))
    return {'v_axis':velocities.tolist(),'q_axis':q_values.tolist(),'nu4_axis':jnp.exp(q_values).tolist(),
        'loss_rows_v_columns_q':losses,'sampled_minimum':{'v':float(velocities[i]),'nu4':float(jnp.exp(q_values[j])),'loss':losses[i][j]},
        'interior_sampled_minima':minima,'min_loss':lo,'max_loss':hi,
        'log10_dynamic_range':math.log10(hi/lo) if lo>0 else None}


def geometry(model,fit):
    *_,loss,_=dataset(model=model)
    theta=jnp.array([fit['v'],fit['q']])
    global_grid=landscape(loss,jnp.linspace(.7,1.3,31),jnp.linspace(jnp.log(.0005),jnp.log(.005),31))
    local_grid=landscape(loss,fit['v']*(1+jnp.linspace(-.02,.02,31)),fit['q']+jnp.linspace(-.05,.05,31))
    slices=[]
    for offset in (-.05,-.02,-.01,-.005,0.,.005,.01,.02,.05):
        velocity=fit['v']*(1+offset); q=fit['q']+offset
        slices.append({'relative_v_or_log_nu4_offset':offset,'v':velocity,'nu4':float(jnp.exp(q)),
            'loss_v_slice':float(loss(theta.at[0].set(velocity))),
            'loss_q_slice':float(loss(theta.at[1].set(q)))})
    return {'global':global_grid,'local':local_grid,'slices':slices,'hessian':fit['hessian']}


def summarize(rows):
    summary={'count':len(rows)}
    for key in ('v','nu4'):
        values=[a[key] for a in rows]
        summary['mean_'+key]=statistics.mean(values)
        summary['sd_'+key]=statistics.stdev(values) if len(values)>1 else None
        summary['mean_relative_'+key+'_error']=statistics.mean(a['relative_'+key+'_error'] for a in rows)
        summary['min_'+key]=min(values); summary['max_'+key]=max(values)
    summary['mean_clean_field_relative_l2']=statistics.mean(a['final_clean_field_relative_l2'] for a in rows)
    conditions=[a['hessian']['condition_number'] for a in rows]
    summary['mean_condition_number']=statistics.mean(conditions) if all(a is not None for a in conditions) else None
    summary['mean_normalized_hessian_coupling']=statistics.mean(a['hessian']['normalized_hessian_coupling'] for a in rows)
    return summary


def main():
    destination=Path(__file__).resolve().parents[1]/'docs/stage6_results.json'
    velocities=jnp.linspace(*V_BOUNDS,21)
    amplification=max(sampled_range_stability(float(v),2*jnp.pi/N,DT,NU_BOUNDS) for v in velocities)
    limit,_=stability_limit(V_BOUNDS[1],NU_BOUNDS[1],2*jnp.pi/N)
    if amplification>1+1e-12 or DT/limit>.65:
        raise RuntimeError('Fixed step failed box stability diagnostic')
    results={'configuration':{'N':N,'T':2.,'dt':DT,'num_steps':STEPS,'v_true':1.,'nu4_true':.002,
        'v_bounds':V_BOUNDS,'nu4_bounds':NU_BOUNDS,'coordinates':['v','q=log(nu4)'],
        'iterations':ITERATIONS,'learning_rate':.05,'beta1':.9,'beta2':.999,'epsilon':1e-8,
        'clean_starts':STARTS,'noise_seeds':SEEDS,'layout_seed':200,'global_grid_shape':[31,31],
        'local_grid_shape':[31,31],'clean_relative_tolerance':CLEAN_RELATIVE_TOL,
        'gradient_relative_tolerance':GRADIENT_RELATIVE_TOL,'stationarity_tolerance':STATIONARITY_TOL},
        'stability':{'sampled_velocities':velocities.tolist(),'nu4_samples':101,'angle_samples':8193,
                     'max_amplification':amplification,'dt_max_at_upper_corner':limit,'safety_fraction':DT/limit},
        'runs':{},'studies':{},'multi_start_checks':[],'experiment_conclusion':'IN PROGRESS'}
    results['optimizer_calibration'] = json.loads(
        (destination.parent/'stage6_optimizer_calibration.json').read_text())
    def save():
        destination.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    def execute(identifier,**kwargs):
        print('Starting',identifier,flush=True)
        row=run_fit(identifier,**kwargs); results['runs'][identifier]=row; save()
        print(identifier,{k:row[k] for k in ('v','nu4','final_loss','gradient_norm','final_clean_field_relative_l2')},flush=True)
        jax.clear_caches()
        return identifier
    def group(name,keys):
        results['studies'][name]={'run_ids':keys,'summary':summarize([results['runs'][k] for k in keys])}; save()
    # Verify clean and fixed masked/noisy gradients before recovery.
    mask8=sensor_layout(N,8,200).tolist(); mask4=sensor_layout(N,4,200).tolist()
    *_,clean_loss,_=dataset()
    *_,masked_loss,_=dataset(SCHEDULES[3],mask8,.02,100)
    results['gradient_checks']={'clean':gradient_check(clean_loss),'masked_noisy':gradient_check(masked_loss)}
    if any(e['relative_difference']>GRADIENT_RELATIVE_TOL for check in results['gradient_checks'].values() for c in check['components'] for e in c['estimates']):
        save(); raise RuntimeError('Joint gradient verification failed')
    group('matched_clean',[execute(f'clean_start_{i}',initial=initial) for i,initial in enumerate(STARTS)])
    print('Computing matched global/local landscapes',flush=True)
    results['matched_geometry']=geometry('matched',results['runs']['clean_start_0']); save(); jax.clear_caches()
    for level in (.01,.05):
        group(f'noise_{level}',[execute(f'noise_{level}_seed_{seed}',sigma_rel=level,seed=seed) for seed in SEEDS])
    for name,level,mask in (('moderate',.02,mask8),('severe',.05,mask4)):
        group(name,[execute(f'{name}_seed_{seed}',times=SCHEDULES[3],sensors=mask,sigma_rel=level,seed=seed,layout_seed=200) for seed in SEEDS])
    for name in ('moderate','severe'):
        for seed in (100,104):
            primary=f'{name}_seed_{seed}'; source=results['runs'][primary]
            alternate=execute(primary+'_alternate',initial=(1.3,.01),times=source['time_indices'],sensors=source['sensor_indices'],
                sigma_rel=source['sigma_rel'],seed=seed,layout_seed=200)
            other=results['runs'][alternate]
            results['multi_start_checks'].append({'primary':primary,'alternate':alternate,
                'v_difference':other['v']-source['v'],'nu4_difference':other['nu4']-source['nu4'],
                'loss_difference':other['final_loss']-source['final_loss']})
    group('continuum_clean',[execute('continuum_clean',model='continuum'),execute('continuum_clean_alternate',model='continuum',initial=(1.3,.01))])
    print('Computing continuum global/local landscapes',flush=True)
    reference=results['runs']['continuum_clean']
    results['continuum_geometry']=geometry('continuum',reference); save(); jax.clear_caches()
    group('continuum_stressed',[execute(f'continuum_stressed_seed_{seed}',model='continuum',sigma_rel=.02,seed=seed,
        times=SCHEDULES[3],sensors=mask8,layout_seed=200) for seed in SEEDS])
    for row in results['runs'].values():
        if row['truth_model']=='continuum':
            row['observation_shift_v']=row['v']-reference['v']
            row['observation_shift_nu4']=row['nu4']-reference['nu4']
    stage5=json.loads((destination.parent/'stage5_results.json').read_text())
    results['stage5_comparison']={name:{'scalar':stage5['studies'][key]['summary'],
        'joint':results['studies'][name]['summary']} for name,key in (('moderate','combined_2'),('severe','combined_3'))}
    results['stage5_scalar_continuum']=stage5['runs']['continuum_clean_full']['recovered_nu4']
    k=jnp.arange(1,4,dtype=jnp.float64); dx=2*jnp.pi/N
    k1=jnp.sin(k*dx)/dx; k4=16*jnp.sin(k*dx/2)**4/dx**4
    results['modified_wavenumbers']={'k':k.tolist(),'k1_discrete':k1.tolist(),'k4_discrete':k4.tolist(),
        'v_effective':(k/k1).tolist(),'nu4_effective':(.002*k**4/k4).tolist()}
    rows=list(results['runs'].values()); clean=[results['runs'][k] for k in results['studies']['matched_clean']['run_ids']]
    checks={'clean_recovery':all(a['relative_v_error']<CLEAN_RELATIVE_TOL and a['relative_nu4_error']<CLEAN_RELATIVE_TOL for a in clean),
        'stationary':all(a['kkt_residual_norm']<STATIONARITY_TOL for a in rows),
        'stable_bounds':all(a['history_v_range'][0]>=V_BOUNDS[0]-1e-12 and a['history_v_range'][1]<=V_BOUNDS[1]+1e-12
            and a['history_nu4_range'][0]>=NU_BOUNDS[0]*(1-1e-12) and a['history_nu4_range'][1]<=NU_BOUNDS[1]*(1+1e-12) for a in rows),
        'matched_local_minimum_consistent':abs(results['matched_geometry']['local']['sampled_minimum']['loss']-clean[0]['final_loss'])<1e-12,
        'clean_hessian_positive':results['matched_geometry']['hessian']['positive_definite']}
    results['checks']=checks
    results['experiment_conclusion']='PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    save(); print('Checks',checks,flush=True); print(results['experiment_conclusion'],'(pytest must also pass)',flush=True)
    if not all(checks.values()):
        raise RuntimeError('Joint study numerical checks need attention')


if __name__=='__main__':
    main()
