"""Checkpointed nonlinear likelihood profiles and repeated-noise calibration."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src.discrepancy import rk4_observations,unpack_parameters,parameter_bounds,discrepancy_symbol,rk4_amplification
from src.identifiability import (gaussian_nll,constrained_adam,profile_intervals,transform_q_interval,
    interval_contains,transform_hessian_to_physical,curvature_summary,atomic_json,LR_LEVELS,paired_observation_records)
from src.observations import add_measurement_noise
from src.robust_design import frozen_hash,bias_decomposition
from experiments.stage9_model_discrepancy import context as stage9_context,plain

PATH=Path('docs/stage10_results.json')
OLD=json.loads(Path('docs/stage9_results.json').read_text())
CAL=tuple(OLD['calibrated_coefficients'])
MODELS=['M1','M4','M5']
TOL=1e-7


def save(r):atomic_json(PATH,plain(r))


def load():
    if PATH.exists():
        r=json.loads(PATH.read_text())
        if r['protocol']['stage9_hash']!=hashlib.sha256(Path('docs/stage9_results.json').read_bytes()).hexdigest():raise RuntimeError('Stage 9 record changed')
        return r
    preserved={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['src','tests','experiments','docs'] for p in Path(directory).glob('*') if p.is_file() and p.suffix in ['.py','.json','.md'] and 'identifiability' not in p.name and 'stage10' not in p.name}
    r={'stage':10,'preservation':preserved,'protocol':{
      'stage9_hash':hashlib.sha256(Path('docs/stage9_results.json').read_bytes()).hexdigest(),
      'calibrated_coefficients':CAL,'designs':OLD['protocol']['designs'],'models':MODELS,
      'clean_reference_sigma_rel':.02,'clean_reference_interpretation':'Hypothetical positive-time measurement precision for noiseless profile geometry; not a zero-variance likelihood.',
      'moderate_seeds':list(range(800,900)),'severe_seeds':list(range(800,850)),
      'heldout_seeds':list(range(900,950)),'matched_control_seeds':list(range(800,820)),
      'coverage_moderate':list(range(800,820)),'coverage_severe':list(range(800,810)),'coverage_heldout':list(range(900,910)),
      'optimizer':{'learning_rate':.05,'beta1':.9,'beta2':.999,'epsilon':1e-8,'initial_updates':2000,'cap':4000,'KKT_threshold':TOL,'objective':'Gaussian NLL with fixed Stage9 NMSE optimization scaling; thresholds and curvature use raw NLL'},
      'global_starts':'Declared Stage9 main start, clean/reference center, alternate [.8,log(.004),.2,1.8]; minimum converged NLL only.',
      'profile_starts':'Central optimum, alternate [.8,log(.004),.2,1.8], then previous-point best preliminary optimum (batched warm pass). All points use all three.',
      'profile_grid':'q: 61 log-spaced points over [.0005,.005] for full; full inverse bounds for sparse; includes center and pseudo target. c6:61, c3:41, v:41 full bounds. Eight adaptive bisection rounds at nominal crossings. Full q range expands to inverse bounds if accepted at interior edges.',
      'profile_failures':'Retained; excluded from interpolation. Coverage unresolved intervals count as indeterminate, with explicit denominators and bounds.',
      'surface':'31x31 q in pseudo q +/- .15, c6 in [0,2], plus threshold range supplied by full 1D q profiles',
      'bound_diagnostics':'Continuum A moderate clean reference, B1/B2/B3, common 2048-step graph; q and c6 profiles. Main bounds remain [0,2].',
      'optional_skipped':['offnominal truth','prediction envelopes'],
      'baseline_reconciliation':'If a converged fixed-point profile improves unrestricted NLL by >1e-5, try one full 2000/4000 fit initialized there; retain original and reconciliation; no truth-based selection.'},
      'pseudo':{},'profiles':{},'bootstrap':{},'datasets':{},'coverage':{},'failures':{}}
    save(r);return r


def context(truth,budget,bound=2):
    ctx=stage9_context(OLD,truth,budget)
    if bound!=2:
        # All bound-diagnostic graphs, including B1, use this common schedule.
        ctx['dt']=2/2048;ctx['steps']=2048
        times=np.linspace(0,2,9) if budget=='full' else np.array(OLD['protocol']['designs'][budget]['times'])
        ctx['ti']=jnp.array(np.rint(times/ctx['dt']).astype(int))
    _,meta=add_measurement_noise(ctx['clean'],ctx['ti'],.05 if budget=='4' else .02,0)
    ctx['sigma']=meta['sigma_abs']
    return ctx


def predict(theta,ctx,model,ti=None,sensors=None):
    return rk4_observations(unpack_parameters(theta,model,CAL),ctx['psi'],ctx['dt'],ctx['dx'],ctx['ti'] if ti is None else ti,ctx['sensors'] if sensors is None else sensors)


def loss_fn(ctx,model,data):
    return lambda theta:gaussian_nll(predict(theta,ctx,model),data,ctx['ti'],ctx['sigma'])


@lru_cache(None)
def kernel(model,dt,dx,fixed_indices,bound,iterations):
    lo,hi=parameter_bounds(model)
    if model=='M4':hi=hi.at[2:].set(bound)
    def one(start,state,fixed,data,psi,ti,sensors,sigma):
        def loss(theta):
            pred=rk4_observations(unpack_parameters(theta,model,CAL),psi,dt,dx,ti,sensors)
            return gaussian_nll(pred,data,ti,sigma)
        scale=2*sigma**2/(data.size*(jnp.mean(data**2)+1e-12))
        run,kkt=constrained_adam(lambda z:scale*loss(z),start,lo,hi,fixed_indices,fixed,iterations,state)
        return {'theta':run['theta'],'NLL':loss(run['theta']),'KKT':kkt,'NLL_KKT':kkt/scale,'optimization_scale':scale,
          'contacts':run['parameter_bound_contacts'],'state':run['optimizer_state'],
          'finite':jnp.all(jnp.isfinite(run['loss_history']))&jnp.all(jnp.isfinite(run['theta_history']))}
    return jax.jit(jax.vmap(one,in_axes=(0,0,0,0,None,None,None,None)))


def optimize(ctx,model,starts,data=None,fixed_indices=(),fixed_values=None,bound=2):
    starts=np.asarray(starts,dtype=float);n=len(starts)
    if fixed_values is None:fixed_values=np.zeros((n,0))
    fixed=np.asarray(fixed_values)
    starts[:,list(fixed_indices)]=fixed
    data=np.asarray(ctx['clean'])[None] if data is None else np.asarray(data)
    if len(data)==1 and n>1:data=np.repeat(data,n,axis=0)
    state={'theta':jnp.array(starts),'first_moment':jnp.zeros_like(jnp.array(starts)),
      'second_moment':jnp.zeros_like(jnp.array(starts)),'update_count':jnp.zeros(n,dtype=jnp.int64)}
    args=(model,ctx['dt'],ctx['dx'],tuple(fixed_indices),bound,2000)
    initial=jax.device_get(kernel(*args)(jnp.array(starts),state,jnp.array(fixed),jnp.array(data),ctx['psi'],ctx['ti'],ctx['sensors'],ctx['sigma']))
    fail=np.where(initial['KKT']>=TOL)[0]
    final=jax.tree.map(lambda a:np.copy(a),initial)
    if len(fail):
        sub=jax.tree.map(lambda a:jnp.asarray(a[fail]),initial['state'])
        continuation=jax.device_get(kernel(*args)(jnp.array(starts[fail]),sub,jnp.array(fixed[fail]),jnp.array(data[fail]),ctx['psi'],ctx['ti'],ctx['sensors'],ctx['sigma']))
        for k,v in final.items():
            if isinstance(v,dict):
                for key in v:v[key][fail]=continuation[k][key]
            else:v[fail]=continuation[k]
    rows=[]
    for i in range(n):
        if not(initial['finite'][i] and final['finite'][i]):raise FloatingPointError('Nonfinite likelihood optimization')
        rows.append({'start':starts[i].tolist(),'fixed_indices':list(fixed_indices),'fixed_values':fixed[i].tolist(),
          'theta':final['theta'][i].tolist(),'NLL':float(final['NLL'][i]),'KKT':float(final['KKT'][i]),'NLL_KKT':float(final['NLL_KKT'][i]),'optimization_scale':float(final['optimization_scale'][i]),
          'contacts':final['contacts'][i].tolist(),'initial_updates':2000,'updates':int(final['state']['update_count'][i]),
          'continued':bool(i in fail),'initial_KKT':float(initial['KKT'][i]),'initial_NLL':float(initial['NLL'][i]),
          'continuation_change':(final['theta'][i]-initial['theta'][i]).tolist(),
          'initial_optimizer_state':{k:plain(v[i]) for k,v in initial['state'].items()},
          'final_optimizer_state':{k:plain(v[i]) for k,v in final['state'].items()},
          'converged':bool(final['KKT'][i]<TOL)})
    return rows


def best_run(rows):
    valid=[a for a in rows if a['converged']]
    chosen=min(valid or rows,key=lambda a:a['NLL'])
    return dict(chosen,attempts=rows,valid=bool(valid))


def attach_metrics(row,ctx,model,data,bound=2):
    theta=jnp.array(row['theta']);p=np.array(unpack_parameters(theta,model,CAL),float)
    obs=predict(theta,ctx,model);field=predict(theta,ctx,model,jnp.array([ctx['steps']]),jnp.arange(32))[0]
    row.update(parameters=p.tolist(),sigma_abs=ctx['sigma'],normalized_mse=float(jnp.mean((obs-data)**2)/(jnp.mean(jnp.asarray(data)**2)+1e-12)),
      clean_observed_nmse=float(jnp.mean((obs-ctx['clean'])**2)/(jnp.mean(ctx['clean']**2)+1e-12)),
      final_field_l2=float(jnp.linalg.norm(field-ctx['truth_final'])/jnp.linalg.norm(ctx['truth_final'])),
      physical_error=(p[:2]-ctx['truth_parameters']).tolist(),
      discrepancy_bound_hit=bool(model=='M4' and np.any((p[2:]<=1e-8)|(p[2:]>=bound-1e-8))))
    return row


def global_fit(ctx,model,center,data=None,bound=2):
    dim=len(parameter_bounds(model)[0]);main=[.9,np.log(.0015),1.,1.][:dim];alt=[.8,np.log(.004),.2,1.8][:dim]
    runs=optimize(ctx,model,[main,center,alt],None if data is None else np.array(data)[None],bound=bound)
    return attach_metrics(best_run(runs),ctx,model,ctx['clean'] if data is None else data,bound)


def curvature(ctx,model,theta):
    theta=jnp.array(theta);loss=loss_fn(ctx,model,ctx['clean']);gradient=jax.grad(loss)(theta);h=jax.hessian(loss)(theta)
    physical=theta.at[1].set(jnp.exp(theta[1]))
    hp=jax.hessian(lambda z:loss(z.at[1].set(jnp.log(z[1]))))(physical)
    transformed=transform_hessian_to_physical(h,gradient,theta)
    q_summary=curvature_summary(h);p_summary=curvature_summary(hp)
    q_summary['gradient']=plain(gradient)
    return {'log_coordinates':q_summary,'physical_coordinates':p_summary,
      'transformation_relative_error':float(jnp.max(jnp.abs(hp-transformed))/jnp.maximum(1.,jnp.max(jnp.abs(hp))))}


def pseudo(r):
    for truth in ['continuum_A','continuum_C','matched_A']:
        for budget in ['full','8','4']:
            ctx=context(truth,budget)
            for model in MODELS:
                key=f'{truth}/{budget}/{model}'
                if key in r['pseudo']:continue
                old=OLD['clean'].get(key)
                center=old['final']['theta'] if old else [.9,np.log(.0015),1.,1.][:len(parameter_bounds(model)[0])]
                row=global_fit(ctx,model,center);row['curvature']=curvature(ctx,model,row['theta'])
                r['pseudo'][key]=row;save(r);print('Pseudo',key,row['KKT'],row['valid'],flush=True)


def profile(ctx,model,central,target,data=None,bound=2,extra=(),points=None):
    data=np.array(ctx['clean'] if data is None else data);theta=np.array(central['theta'])
    lo,hi=map(np.asarray,parameter_bounds(model));hi=hi.copy()
    if model=='M4':hi[2:]=bound
    if points is None:
        if target==1 and ctx['budget']=='full':axis=np.linspace(np.log(.0005),np.log(.005),61)
        else:axis=np.linspace(lo[target],hi[target],61 if target in [1,3] else 41)
    else:axis=np.asarray(points)
    axis=np.unique(np.r_[axis,theta[target],extra]);dim=len(theta)
    alternate=np.array([.8,np.log(.004),.2,1.8][:dim]);records={}
    def evaluate(xs,warm=None):
        if not len(xs):return
        xs=np.asarray(xs);starts=np.repeat(np.stack([theta,alternate])[None],len(xs),axis=0).reshape(-1,dim)
        fixed=np.repeat(xs,2)[:,None]
        attempts=optimize(ctx,model,starts,data[None],(target,),fixed,bound)
        initial_best=[best_run(attempts[2*i:2*i+2]) for i in range(len(xs))]
        previous=[theta]+[a['theta'] for a in initial_best[:-1]] if warm is None else warm
        warmed=optimize(ctx,model,np.array(previous),data[None],(target,),xs[:,None],bound)
        for i,x in enumerate(xs):
            row=best_run(attempts[2*i:2*i+2]+[warmed[i]])
            records[float(x)]=dict(row,target=float(x),warm_start_source='previous preliminary optimum' if warm is None else 'nearest existing optimum')
    evaluate(axis)
    # Full-clean initial q range must not silently truncate an accepted region.
    if target==1 and ctx['budget']=='full':
        ends=[records[float(axis[0])],records[float(axis[-1])]]
        if any(a['valid'] and 2*(a['NLL']-central['NLL'])<=3.841459 for a in ends):
            more=np.unique(np.r_[np.linspace(lo[target],axis[0],21),np.linspace(axis[-1],hi[target],21)])
            evaluate([x for x in more if float(x) not in records])
    for _ in range(8):
        xs=sorted(records);extra_x=[]
        for a,b in zip(xs,xs[1:]):
            ra,rb=records[a],records[b]
            if not(ra['valid'] and rb['valid']):continue
            la,lb=2*(ra['NLL']-central['NLL']),2*(rb['NLL']-central['NLL'])
            if any((la-th)*(lb-th)<0 for th in LR_LEVELS.values()):extra_x.append((a+b)/2)
        warm=[records[min(xs,key=lambda y:abs(y-x))]['theta'] for x in extra_x]
        evaluate(extra_x,warm)
    rows=[records[x] for x in sorted(records)];baseline=central['NLL'];reconciliation=None
    valid_rows=[a for a in rows if a['valid']]
    if valid_rows and min(a['NLL'] for a in valid_rows)<baseline-1e-5:
        candidate=min(valid_rows,key=lambda a:a['NLL'])
        reconciliation=best_run(optimize(ctx,model,[candidate['theta']],data[None],bound=bound))
        if reconciliation['valid'] and reconciliation['NLL']<baseline:baseline=reconciliation['NLL']
    baseline_ok=central['valid'] and (not valid_rows or min(a['NLL'] for a in valid_rows)>=baseline-1e-5)
    xs=[a['target'] for a in rows];lr=[2*(a['NLL']-baseline) for a in rows];valid=[a['valid'] and baseline_ok for a in rows]
    intervals={level:profile_intervals(xs,lr,valid,threshold,(lo[target],hi[target])) for level,threshold in LR_LEVELS.items()}
    if target==1:
        for value in intervals.values():value['nu4']=transform_q_interval(value)
    # Report raw jumps; do not smooth or discard branch switches.
    span=hi-lo;jumps=[]
    for i in range(len(rows)-1):
        change=np.abs(np.array(rows[i+1]['theta'])-rows[i]['theta'])/span
        change[target]=0
        if max(change)>.15:jumps.append({'left':xs[i],'right':xs[i+1],'normalized_nuisance_jump':change.tolist()})
    return {'target_index':target,'axis':xs,'nu4_axis':np.exp(xs).tolist() if target==1 else None,
      'rows':rows,'LR':lr,'valid_points':valid,'baseline_NLL':baseline,'baseline_valid':baseline_ok,
      'baseline_reconciliation':reconciliation,'intervals':intervals,'jumps':jumps,
      'failed_points':sum(not a for a in valid),'bounds':[float(lo[target]),float(hi[target])]}


def clean_profiles(r):
    pseudo(r)
    for budget in ['full','8','4']:
        for model in MODELS:
            key=f'continuum_A/{budget}/{model}';ctx=context('continuum_A',budget)
            for target in range(4 if model=='M4' else 2):
                pkey=f'{key}/{target}'
                if pkey in r['profiles']:continue
                print('Profile',pkey,flush=True)
                r['profiles'][pkey]=profile(ctx,model,r['pseudo'][key],target)
                save(r);print('  failed points',r['profiles'][pkey]['failed_points'],flush=True)


def surfaces(r):
    if not r['pseudo']:raise RuntimeError('Run clean profiles first')
    if 'surface' not in r:
        ctx=context('continuum_A','full');center=r['pseudo']['continuum_A/full/M4'];q0=center['theta'][1]
        qa=np.linspace(q0-.15,q0+.15,31);ca=np.linspace(0,2,31);fixed=np.array([(q,c) for q in qa for c in ca]);dim=4
        starts=np.tile(center['theta'],(len(fixed),1));first=optimize(ctx,'M4',starts,fixed_indices=(1,3),fixed_values=fixed)
        alt=np.tile([.8,np.log(.004),.2,1.8],(len(fixed),1));second=optimize(ctx,'M4',alt,fixed_indices=(1,3),fixed_values=fixed)
        warm=np.array([center['theta']]+[best_run([a,b])['theta'] for a,b in zip(first,second)][:-1])
        third=optimize(ctx,'M4',warm,fixed_indices=(1,3),fixed_values=fixed)
        rows=[best_run([a,b,c]) for a,b,c in zip(first,second,third)]
        vals=np.array([a['NLL'] for a in rows]).reshape(31,31);valid=np.array([a['valid'] for a in rows]).reshape(31,31)
        valley=[]
        for i,q in enumerate(qa):
            choices=np.where(valid[i])[0]
            if len(choices):
                j=choices[np.argmin(vals[i,choices])];valley.append({'q':q,'nu4':np.exp(q),'c6':ca[j],'NLL':vals[i,j],'LR':2*(vals[i,j]-center['NLL'])})
        r['surface']=plain({'q_axis':qa,'nu4_axis':np.exp(qa),'c6_axis':ca,'rows':rows,'NLL':vals,'valid':valid,
          'valley':valley,'reference_NLL':center['NLL'],'range_limited':'Finite local surface; use the full 1D q profile for interval extent.',
          'level_ranges':{k:[min([a['nu4'] for a in valley if a['LR']<=t],default=None),max([a['nu4'] for a in valley if a['LR']<=t],default=None)] for k,t in LR_LEVELS.items()}});save(r)
    if 'bound_sensitivity' not in r:r['bound_sensitivity']={}
    for bound in [2,3,4]:
        key=str(bound)
        if key in r['bound_sensitivity']:continue
        # marker bound=0 requests the common diagnostic timestep also for B1.
        ctx=context('continuum_A','8',bound=0)
        angles=jnp.linspace(0,jnp.pi,8193)
        amplification=max(float(jnp.max(jnp.abs(rk4_amplification(ctx['dt']*discrepancy_symbol(angles,ctx['dx'],v,nu,c3,c6))))) for v in [.5,1.,1.5] for nu in [1e-4,.002,.02] for c3 in [0,bound/2,bound] for c6 in [0,bound/2,bound])
        if amplification>1+1e-10:raise RuntimeError('Diagnostic bound graph unstable')
        center=global_fit(ctx,'M4',r['pseudo']['continuum_A/8/M4']['theta'],bound=bound)
        results={'center':center,'num_steps':2048,'max_sampled_amplification':amplification}
        for target in [1,3]:results[str(target)]=profile(ctx,'M4',center,target,bound=bound)
        r['bound_sensitivity'][key]=results;save(r);print('Bounds',bound,flush=True)


def distribution(values):
    x=np.array(values);return {'mean':x.mean(0).tolist(),'median':np.median(x,axis=0).tolist(),'sample_sd':x.std(0,ddof=1).tolist(),
      'p05':np.quantile(x,.05,axis=0).tolist(),'p25':np.quantile(x,.25,axis=0).tolist(),'p75':np.quantile(x,.75,axis=0).tolist(),'p95':np.quantile(x,.95,axis=0).tolist()}


def ensemble_summary(rows):
    return {'n':len(rows),'parameters':distribution([a['parameters'] for a in rows]),
      'physical_errors':distribution([a['physical_error'] for a in rows]),
      'pseudo_deviations':distribution([a['decomposition']['noise_induced_shift'] for a in rows]),
      'field_error':distribution([a['final_field_l2'] for a in rows]),'clean_observed_nmse':distribution([a['clean_observed_nmse'] for a in rows]),
      'discrepancy_bound_fraction':np.mean([a['discrepancy_bound_hit'] for a in rows]),
      'continuation_fraction':np.mean([a['continued'] for a in rows]),'unconverged_fraction':np.mean([not a['valid'] for a in rows])}


def bootstrap(r,truth,budget,seeds,coverage_seeds,models=MODELS):
    key=f'{truth}/{budget}';ctx=context(truth,budget)
    if key not in r['datasets']:
        records=paired_observation_records(ctx['clean'],ctx['ti'],.05 if budget=='4' else .02,seeds)
        r['datasets'][key]=records;save(r)
    records=r['datasets'][key]
    if [a['seed'] for a in records]!=list(seeds):raise RuntimeError('Seed protocol changed')
    for model in models:
        groupkey=f'{key}/{model}';pseudo_row=r['pseudo'][groupkey]
        if groupkey in r['bootstrap']:
            existing=r['bootstrap'][groupkey]['runs']
            if [a['seed'] for a in existing]!=list(seeds):raise RuntimeError('Incomplete bootstrap checkpoint '+groupkey)
            if any(a['observation_hash']!=b['hash'] for a,b in zip(existing,records)):
                raise RuntimeError('Observation mismatch in checkpoint '+groupkey)
        if groupkey not in r['bootstrap']:
            dim=len(pseudo_row['theta']);starts=[[.9,np.log(.0015),1.,1.][:dim],pseudo_row['theta'],[.8,np.log(.004),.2,1.8][:dim]]
            allstarts=np.tile(starts,(len(records),1));all_data=np.repeat(np.array([a['observations'] for a in records]),3,axis=0)
            runs=optimize(ctx,model,allstarts,all_data);rows=[]
            for i,record in enumerate(records):
                row=attach_metrics(best_run(runs[3*i:3*i+3]),ctx,model,jnp.array(record['observations']))
                row.update(seed=record['seed'],observation_hash=record['hash'],decomposition=bias_decomposition(row['parameters'][:2],pseudo_row['parameters'][:2],ctx['truth_parameters']),
                  discrepancy_pseudo_deviation=(np.array(row['parameters'][2:])-pseudo_row['parameters'][2:]).tolist())
                rows.append(row)
            r['bootstrap'][groupkey]={'runs':rows,'summary':plain(ensemble_summary(rows))};save(r)
            print('Bootstrap',groupkey,'unconverged',sum(not a['valid'] for a in rows),flush=True)
        for seed in coverage_seeds:
            pkey=f'{groupkey}/{seed}'
            if pkey in r['coverage']:
                existing=r['coverage'][pkey]
                if len(existing['rows'])<61 or 'coverage' not in existing:
                    raise RuntimeError('Incomplete coverage checkpoint '+pkey)
                continue
            record=next(a for a in records if a['seed']==seed);central=next(a for a in r['bootstrap'][groupkey]['runs'] if a['seed']==seed)
            print('Coverage',pkey,flush=True)
            prof=profile(ctx,model,central,1,jnp.array(record['observations']),extra=[pseudo_row['theta'][1],np.log(.002)])
            exact=next(a for a in prof['rows'] if a['target']==pseudo_row['theta'][1])
            prof['LR_at_pseudo']=2*(exact['NLL']-prof['baseline_NLL']) if exact['valid'] and prof['baseline_valid'] else None
            prof['coverage']={level:{'pseudo':interval_contains(interval,pseudo_row['theta'][1]),'physical':interval_contains(interval,np.log(.002)),
                'resolved':not interval['unresolved'] and prof['baseline_valid']} for level,interval in prof['intervals'].items()}
            r['coverage'][pkey]=prof;save(r)



def representative_profiles(r):
    """First declared A seed, all non-q coordinates; q already in coverage."""
    groups=r.setdefault('noisy_profiles',{})
    for budget in ['8','4']:
        ctx=context('continuum_A',budget)
        data=next(a['observations'] for a in r['datasets']['continuum_A/'+budget] if a['seed']==800)
        for model in MODELS:
            key=f'continuum_A/{budget}/{model}'
            central=next(a for a in r['bootstrap'][key]['runs'] if a['seed']==800)
            for target in ([0,2,3] if model=='M4' else [0]):
                pkey=f'{key}/800/{target}'
                if pkey in groups:continue
                groups[pkey]=profile(ctx,model,central,target,jnp.array(data))
                save(r);print('Representative noisy profile',pkey,flush=True)

def main():
    parser=argparse.ArgumentParser()
    names=['clean-profiles','profile-surfaces','bootstrap-moderate','bootstrap-severe','heldout-transfer','matched-controls','representative-profiles','finalize']
    for name in names:parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args();r=load()
    functions={'clean_profiles':lambda:clean_profiles(r),'profile_surfaces':lambda:surfaces(r),
      'bootstrap_moderate':lambda:bootstrap(r,'continuum_A','8',range(800,900),range(800,820)),
      'bootstrap_severe':lambda:bootstrap(r,'continuum_A','4',range(800,850),range(800,810)),
      'heldout_transfer':lambda:bootstrap(r,'continuum_C','8',range(900,950),range(900,910)),
      'matched_controls':lambda:[bootstrap(r,'matched_A',b,range(800,820),[],['M1','M4']) for b in ['8','4']],
      'representative_profiles':lambda:representative_profiles(r)}
    for name,fn in functions.items():
        if getattr(args,name):
            try:fn()
            except Exception as exc:r['failures'][name]=repr(exc);save(r);raise
            else:r['failures'].pop(name,None);save(r)
    if args.finalize:
        from experiments.stage10_report import finalize
        finalize(r);save(r)

if __name__=='__main__':main()
