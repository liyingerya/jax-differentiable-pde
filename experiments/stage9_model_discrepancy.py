"""Reproducible, checkpointed Stage 9; run phases in the documented order."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src.discrepancy import (
    discrepancy_integrate, discrepancy_symbol, discrepancy_loss,
    rk4_observations, unpack_parameters, parameter_bounds,
    projected_residual, stability_policy,
)
from src.operators import d3_dx3,d6_dx6
from src.solver import integrate
from src.joint_inverse import projected_adam_kernel
from src.robust_design import INITIAL_MODES,initial_field,continuum_predict,frozen_hash,bias_decomposition
from src.observations import add_measurement_noise

PATH=Path('docs/stage9_results.json')
THRESHOLD=1e-7
MODELS=['M0','M1','M2','M3','M4','M5']
CAL_STARTS=[[0,0],[.5,.5],[1,1],[1.5,1.5],[0,2],[2,0]]
M4_STARTS=[[.8,np.log(.001),.1,.1],[1.2,np.log(.004),.1,.1],
 [.8,np.log(.004),1,1],[1.2,np.log(.001),1,1],
 [.8,np.log(.001),1.8,1.8],[1.2,np.log(.004),1.8,1.8],
 [.8,np.log(.004),.2,1.8],[1.2,np.log(.001),1.8,.2]]
TRUTHS={'matched_A':('A',1.,.002,'fd'),'continuum_A':('A',1.,.002,'continuum'),
 'continuum_C':('C',1.,.002,'continuum'),'continuum_offnominal':('A',1.05,.0018,'continuum'),
 'continuum_B':('B',.9,.0015,'continuum')}


def plain(x):
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if hasattr(x,'tolist'): return x.tolist()
    return x


def save(r):
    tmp=PATH.with_suffix('.tmp');tmp.write_text(json.dumps(plain(r),allow_nan=False,separators=(',',':')));tmp.replace(PATH)


def load():
    if PATH.exists():
        result=json.loads(PATH.read_text())
        protocol=result['protocol']
        if protocol['models']!=MODELS or protocol['calibration_starts']!=CAL_STARTS or protocol['M4_starts']!=M4_STARTS:
            raise RuntimeError('Frozen model/start protocol changed; refusing resume')
        old=json.loads(Path('docs/stage8_results.json').read_text())
        designs={b:old['frozen']['selected'][b]['robust_noise_weighted'] for b in ['8','4']}
        if frozen_hash(designs)!=protocol['design_hash']:
            raise RuntimeError('Frozen Stage 8 designs changed; refusing resume')
        return result
    old=json.loads(Path('docs/stage8_results.json').read_text())
    designs={b:old['frozen']['selected'][b]['robust_noise_weighted'] for b in ['8','4']}
    return {'stage':9,'protocol':{'models':MODELS,'calibration_starts':CAL_STARTS,'M4_starts':M4_STARTS,
      'initial_updates':2000,'threshold':THRESHOLD,'default_cap':4000,'default_learning_rate':.05,
      'beta1':.9,'beta2':.999,'epsilon':1e-8,'designs':designs,'design_hash':frozen_hash(designs),
      'primary_seeds':list(range(700,720)),'transfer_seeds':list(range(720,740)),
      'main_start':[.9,np.log(.0015),1.,1.],
      'selection_rule':'Main fit uses the declared main start; no truth-based selection. Calibration uses lowest loss only after multistart agreement.',
      'production_evaluator':'Exact diagonalization of the fixed-step stencil RK4 polynomial, verified against lax.scan; not exact continuum evolution.',
      'optional_skipped':['profile loss curves','exact resolved-mode correction','offnominal noisy transfer'],
      'optimizer_calibration_rule':'Before production, evaluate clean matched/full continuum and continuum sparse controls at lr .05, .02, .01, .005. Select per family the setting with fewest failures at the allowed cap, then lowest maximum KKT; all candidates retained. M4 may use 6000 only if 4000 failures are systematic.'},
      'completed':[],'failures':{}}


def context(r,truth='continuum_A',budget='full',n=32):
    pol=r['forward']['stability'][str(n)];dx=pol['dx'];dt=pol['dt'];steps=pol['num_steps']
    x=jnp.arange(n)*dx
    ic,v,nu,kind=TRUTHS[truth];psi=initial_field(x,ic)
    times=np.linspace(0,2,9) if budget=='full' else np.array(r['protocol']['designs'][budget]['times'])
    sensors=jnp.arange(n) if budget=='full' else jnp.array(r['protocol']['designs'][budget]['sensor_indices'])
    ti=jnp.array(np.rint(times/dt).astype(int));assert np.max(np.abs(np.array(ti)*dt-times))<1e-14
    if kind=='fd':
        traj=integrate(psi,dt,steps,v,nu,dx);clean=traj[ti][:,sensors];final=traj[-1]
    else:
        clean=continuum_predict(jnp.array([v,np.log(nu)]),x,times,INITIAL_MODES[ic])[:,sensors]
        final=continuum_predict(jnp.array([v,np.log(nu)]),x,jnp.array([2.]),INITIAL_MODES[ic])[0]
    return dict(psi=psi,dt=dt,dx=dx,steps=steps,ti=ti,sensors=sensors,clean=clean,truth_final=final,truth_parameters=[v,nu],truth=truth,budget=budget,n=n)


def forward(r):
    if 'gradient_checks' in r.get('forward',{}):return
    out={'derivative_convergence':[],'symbols':{'D1':'i sin(theta)/dx','D4':'16 sin(theta/2)^4/dx^4','D3':'-4 i sin(theta) sin(theta/2)^2/dx^3','D6':'-64 sin(theta/2)^6/dx^6'},'rate_convergence':[],'stability':{}}
    for n in [16,24,32,48,64]:
        dx=2*np.pi/n;x=jnp.arange(n)*dx
        out['derivative_convergence'].append({'N':n,'D3_relative_error':float(jnp.linalg.norm(d3_dx3(jnp.sin(3*x),dx)+27*jnp.cos(3*x))/jnp.linalg.norm(27*jnp.cos(3*x))), 'D6_relative_error':float(jnp.linalg.norm(d6_dx6(jnp.sin(3*x),dx)+729*jnp.sin(3*x))/jnp.linalg.norm(729*jnp.sin(3*x)))})
        for k in [1,2,3]:
            for c in [0,1]:
                lam=complex(discrepancy_symbol(k*dx,dx,1.,.002,c,c))
                out['rate_convergence'].append(dict(N=n,k=k,c=c,phase_error=abs(lam.imag+k),damping_error=abs(lam.real+.002*k**4)))
        print('Stability N=',n,flush=True);out['stability'][str(n)]=stability_policy(n)
    out['orders']={}
    for k in [1,2,3]:
        for c in [0,1]:
            rows=[a for a in out['rate_convergence'] if a['k']==k and a['c']==c and a['N']>=32]
            out['orders'][f'k{k}/c{c}']={field:float(np.polyfit(np.log([2*np.pi/a['N'] for a in rows]),np.log([a[field] for a in rows]),1)[0]) for field in ['phase_error','damping_error']}
    r['forward']=out;save(r)
    ctx=context(r);theta=jnp.array([.9,np.log(.0015),.7,1.3]);checks={}
    for label,budget in [('full','full'),('sparse_noisy','8')]:
        ctx=context(r,budget=budget);data=ctx['clean']
        if budget!='full':data,_=add_measurement_noise(data,ctx['ti'],.02,699)
        loss=jax.jit(lambda z:discrepancy_loss(z,ctx['psi'],ctx['dt'],ctx['dx'],ctx['ti'],data,ctx['sensors']))
        ad=np.array(jax.grad(loss)(theta));fd=[]
        for i in range(4):
            h=1e-5;d=jnp.eye(4)[i]*h;fd.append(float((loss(theta+d)-loss(theta-d))/(2*h)))
        fd=np.array(fd);checks[label]=dict(autodiff=ad,finite_difference=fd,absolute_error=abs(ad-fd),relative_error=abs(ad-fd)/np.maximum(abs(ad),1e-12))
    ctx=context(r);pars=(.9,.0015,.7,1.3)
    scan=discrepancy_integrate(ctx['psi'],ctx['dt'],ctx['steps'],*pars,ctx['dx'])
    spec=rk4_observations(pars,ctx['psi'],ctx['dt'],ctx['dx'],jnp.arange(ctx['steps']+1),jnp.arange(32))
    out['scan_spectral_max_error']=float(jnp.max(jnp.abs(scan-spec)))
    # Compare derivatives of the independent evaluators, not just their values.
    scanloss=lambda z:jnp.mean((discrepancy_integrate(ctx['psi'],ctx['dt'],ctx['steps'],*unpack_parameters(z),ctx['dx'])[ctx['ti']]-ctx['clean'])**2)/(jnp.mean(ctx['clean']**2)+1e-12)
    fastloss=lambda z:discrepancy_loss(z,ctx['psi'],ctx['dt'],ctx['dx'],ctx['ti'],ctx['clean'],ctx['sensors'])
    out['scan_spectral_gradient_max_error']=float(jnp.max(jnp.abs(jax.grad(scanloss)(theta)-jax.grad(fastloss)(theta))))
    out['gradient_checks']=checks;save(r)


@lru_cache(None)
def fit_kernel(model,n,dt,dx,lr,updates,cal,calibration=False,history=False):
    lower,upper=(jnp.zeros(2),jnp.ones(2)*2) if calibration else parameter_bounds(model)
    def one(start,state,data,psi,ti,sensors):
        def loss(z):
            if calibration:
                pred=rk4_observations((1.,.002,z[0],z[1]),psi,dt,dx,ti,sensors)
                return jnp.mean((pred-data)**2)/(jnp.mean(data**2)+1e-12)
            return discrepancy_loss(z,psi,dt,dx,ti,data,sensors,model,cal)
        run=projected_adam_kernel(loss,start,lower,upper,updates,lr,state)
        final=run['theta'];value,gradient=jax.value_and_grad(loss)(final)
        out={'state':run['optimizer_state'],'theta':final,'loss':value,'gradient':gradient,
          'kkt':projected_residual(final,gradient,lower,upper),'proposal_hits':run['proposal_boundary_hits'],
          'bound_contacts':run['parameter_bound_contacts'],'at_bound':(final<=lower+1e-8)|(final>=upper-1e-8),
          'finite':jnp.all(jnp.isfinite(run['theta_history']))&jnp.all(jnp.isfinite(run['loss_history']))&jnp.all(jnp.isfinite(run['gradient_history']))}
        if history:out['theta_history']=run['theta_history'];out['loss_history']=run['loss_history']
        return out
    return jax.jit(jax.vmap(one,in_axes=(0,0,0,None,None,None)))


def fit(r,ctx,model,data=None,starts=None,lr=None,cap=None,calibration=False,history=False):
    if lr is None:lr=r['optimizer_calibration']['chosen'][model]['learning_rate']
    if cap is None:cap=r['optimizer_calibration']['chosen'][model]['cap']
    cal=tuple(r.get('calibrated_coefficients',[1.,1.]))
    data=np.array(ctx['clean'])[None] if data is None else np.asarray(data)
    if starts is None:
        dim=len(parameter_bounds(model)[0]);starts=np.tile(np.array(r['protocol']['main_start'][:dim]),(len(data),1))
    starts=jnp.array(starts);b=len(starts)
    if len(data)==1 and b>1:data=np.repeat(data,b,axis=0)
    state={'theta':starts,'first_moment':jnp.zeros_like(starts),'second_moment':jnp.zeros_like(starts),'update_count':jnp.zeros(b,dtype=jnp.int64)}
    args=(model,ctx['n'],ctx['dt'],ctx['dx'],lr,2000,cal,calibration,history)
    initial=jax.device_get(fit_kernel(*args)(starts,state,jnp.array(data),ctx['psi'],ctx['ti'],ctx['sensors']))
    failing=np.where(initial['kkt']>=THRESHOLD)[0]
    final={key:np.copy(val) if not isinstance(val,dict) else {k:np.copy(v) for k,v in val.items()} for key,val in initial.items()}
    continuation=None
    if len(failing):
        sub=jax.tree.map(lambda a:jnp.array(a[failing]),initial['state'])
        continuation=jax.device_get(fit_kernel(model,ctx['n'],ctx['dt'],ctx['dx'],lr,cap-2000,cal,calibration,history)(starts[failing],sub,jnp.array(data[failing]),ctx['psi'],ctx['ti'],ctx['sensors']))
        for key in final:
            if key.endswith('_history'):continue
            if isinstance(final[key],dict):
                for k in final[key]:final[key][k][failing]=continuation[key][k]
            else:final[key][failing]=continuation[key]
    rows=[]
    for i in range(b):
        extract=lambda obj:{k:({a:plain(v[i]) for a,v in val.items()} if isinstance(val,dict) else plain(val[i])) for k,val in obj.items() if not k.endswith('_history')}
        row={'initial':extract(initial),'final':extract(final),'initial_update_budget':2000,
          'continuation_required':bool(i in failing),'final_total_updates':int(final['state']['update_count'][i]),
          'parameter_changes_during_continuation':(final['theta'][i]-initial['theta'][i]).tolist(),
          'start':plain(starts[i]),'learning_rate':lr,'cap':cap}
        if history:
            row['initial_history']={k:plain(initial[k][i]) for k in ['theta_history','loss_history']}
            if i in failing:
                j=list(failing).index(i);row['continuation_history']={k:plain(continuation[k][j]) for k in ['theta_history','loss_history']}
        if not row['final']['finite']:raise FloatingPointError('Nonfinite optimizer history')
        if not calibration:
            p=np.array([float(x) for x in unpack_parameters(jnp.array(final['theta'][i]),model,cal)])
            pred=rk4_observations(p,ctx['psi'],ctx['dt'],ctx['dx'],ctx['ti'],ctx['sensors'])
            field=rk4_observations(p,ctx['psi'],ctx['dt'],ctx['dx'],jnp.array([ctx['steps']]),jnp.arange(ctx['n']))[0]
            row['final'].update(parameters=p.tolist(),relative_physical_errors=(abs(p[:2]-ctx['truth_parameters'])/ctx['truth_parameters']).tolist(),
                clean_observed_nmse=float(jnp.mean((pred-ctx['clean'])**2)/(jnp.mean(ctx['clean']**2)+1e-12)),
                full_final_l2=float(jnp.linalg.norm(field-ctx['truth_final'])/jnp.linalg.norm(ctx['truth_final'])))
        rows.append(row)
    return rows


def calibrate(r):
    if 'forward' not in r:raise RuntimeError('Run forward validation first')
    if 'calibration' not in r:r['calibration']={}
    # Known-physics calibration policy frozen before these runs.
    for n in [32,16,24,48,64]:
        if str(n) in r['calibration']:continue
        print('Known-physics calibration N=',n,flush=True)
        rows=fit(r,context(r,n=n),'CAL',starts=CAL_STARTS,lr=.05,cap=4000,calibration=True,history=True)
        values=np.array([a['final']['theta'] for a in rows]);best=min(rows,key=lambda a:a['final']['loss'])
        r['calibration'][str(n)]={'runs':rows,'spread':np.ptp(values,axis=0).tolist(),'selected':best['final']['theta'],
          'common_optimum_supported':bool(np.max(np.ptp(values,axis=0))<1e-4 and max(a['final']['kkt'] for a in rows)<THRESHOLD)}
        save(r)
    if not r['calibration']['32']['common_optimum_supported']:raise RuntimeError('N32 calibration does not support common M5 coefficients')
    r['calibrated_coefficients']=r['calibration']['32']['selected'];save(r)
    if 'optimizer_calibration' not in r:r['optimizer_calibration']={'trials':{},'chosen':{},'production_started':False}
    oc=r['optimizer_calibration']
    contexts=[context(r,'matched_A'),context(r),context(r,budget='8'),context(r,budget='4')]
    for model in MODELS:
        if model in oc['chosen']:continue
        trials=[]
        for lr in [.05,.02,.01,.005]:
            rows=[]
            for ctx in contexts:rows+=fit(r,ctx,model,lr=lr,cap=4000)
            failures=sum(a['final']['kkt']>=THRESHOLD for a in rows)
            trials.append({'learning_rate':lr,'cap':4000,'failures':failures,'max_kkt':max(a['final']['kkt'] for a in rows),'runs':rows})
            print('Optimizer pilot',model,lr,'failures',failures,'max KKT',trials[-1]['max_kkt'],flush=True)
        if model=='M4' and min(a['failures'] for a in trials)>0:
            for lr in [.05,.02,.01,.005]:
                rows=[]
                for ctx in contexts:rows+=fit(r,ctx,model,lr=lr,cap=6000)
                trials.append({'learning_rate':lr,'cap':6000,'failures':sum(a['final']['kkt']>=THRESHOLD for a in rows),'max_kkt':max(a['final']['kkt'] for a in rows),'runs':rows})
        best=min(trials,key=lambda a:(a['failures'],a['max_kkt']))
        # Preserve the default if it already passes all controls.
        if trials[0]['failures']==0:best=trials[0]
        oc['trials'][model]=trials;oc['chosen'][model]={k:best[k] for k in ['learning_rate','cap','failures','max_kkt']};save(r)
    if 'calibrate' not in r['completed']:r['completed'].append('calibrate')
    save(r)


def clean(r):
    if len(r.get('optimizer_calibration',{}).get('chosen',{}))!=6:raise RuntimeError('Complete optimizer calibration first')
    r['optimizer_calibration']['production_started']=True
    groups=r.setdefault('clean',{})
    requests=[]
    for truth in ['matched_A','continuum_A']:
        for model in MODELS:requests.append((truth,'full',model))
    for truth in ['continuum_C','continuum_offnominal','continuum_B']:
        for model in ['M0','M1','M5']:requests.append((truth,'full',model))
    for truth in ['matched_A','continuum_A']:
        for budget in ['8','4']:
            for model in (['M0','M1','M4','M5'] if truth=='matched_A' else MODELS):requests.append((truth,budget,model))
    for model in ['M0','M1','M5']:requests.append(('continuum_C','8',model))
    for truth,budget,model in requests:
        key=f'{truth}/{budget}/{model}'
        if key in groups:continue
        row=fit(r,context(r,truth,budget),model)[0];groups[key]=row;save(r)
        print('Clean',key,'KKT',row['final']['kkt'],flush=True)
    if 'M4_multistart' not in r:
        r['M4_multistart']=fit(r,context(r),'M4',starts=M4_STARTS,history=True);save(r)
    if 'geometry' not in r:
        ctx=context(r);theta=jnp.array(groups['continuum_A/full/M4']['final']['theta'])
        loss=jax.jit(lambda z:discrepancy_loss(z,ctx['psi'],ctx['dt'],ctx['dx'],ctx['ti'],ctx['clean'],ctx['sensors']))
        h=np.array(jax.hessian(loss)(theta));h=(h+h.T)/2;ev,evec=np.linalg.eigh(h)
        tol=100*np.finfo(float).eps*max(1.,max(abs(ev)));coupling=h/np.sqrt(np.outer(np.diag(h),np.diag(h)))
        geo={'matrix':h,'eigenvalues':ev,'eigenvectors_columns':evec,'positive_definiteness_threshold':tol,
            'positive_definite':bool(min(ev)>tol),'condition_number':float(max(ev)/min(ev)) if min(ev)>tol else None,'normalized_hessian_couplings':coupling,'slices':{}}
        for name,i,j,wi,wj in [('q_c6',1,3,.15,.6),('v_c3',0,2,.02,.6)]:
            lo,hi=parameter_bounds('M4');a=np.linspace(max(float(lo[i]),float(theta[i])-wi),min(float(hi[i]),float(theta[i])+wi),41)
            b=np.linspace(max(float(lo[j]),float(theta[j])-wj),min(float(hi[j]),float(theta[j])+wj),41)
            aa,bb=np.meshgrid(a,b,indexing='ij');zs=np.tile(np.array(theta),(1681,1));zs[:,i]=aa.ravel();zs[:,j]=bb.ravel()
            vals=jax.jit(jax.vmap(loss))(jnp.array(zs)).reshape(41,41)
            geo['slices'][name]={'axis1':a,'axis2':b,'loss':vals,'fixed_theta':theta,'type':'conditional loss slice, not profile likelihood'}
        r['geometry']=plain(geo);save(r)
    if 'cross_grid_transfer' not in r:
        r['cross_grid_transfer']={}
    for n in [16,24,32,48,64]:
        if str(n) in r['cross_grid_transfer']:continue
        r['cross_grid_transfer'][str(n)]={model:fit(r,context(r,n=n),model)[0] for model in ['M0','M1','M5']};save(r)
    if 'clean-inference' not in r['completed']:r['completed'].append('clean-inference')
    save(r)


def summarize(rows):
    p=np.array([a['final']['parameters'] for a in rows]);e=np.array([a['final']['relative_physical_errors'] for a in rows])
    return {'count':len(rows),'mean_relative_errors':e.mean(0),'median_relative_errors':np.median(e,axis=0),
      'parameter_mean':p.mean(0),'parameter_sd':p.std(0,ddof=1),'p90_nu4_relative_error':np.quantile(e[:,1],.9),
      'mean_final_field_l2':np.mean([a['final']['full_final_l2'] for a in rows]),
      'mean_clean_observed_nmse':np.mean([a['final']['clean_observed_nmse'] for a in rows]),
      'mean_noisy_loss':np.mean([a['final']['loss'] for a in rows]),
      'bound_rate':np.mean([any(a['final']['at_bound']) for a in rows]),
      'continuation_rate':np.mean([a['continuation_required'] for a in rows]),
      'unconverged_rate':np.mean([a['final']['kkt']>=THRESHOLD for a in rows]),
      'max_kkt':max(a['final']['kkt'] for a in rows),
      'mean_decomposition':{k:np.mean([a['decomposition'][k] for a in rows],axis=0) for k in ['physical_error','clean_model_design_bias','noise_induced_shift']}}


def noisy(r):
    groups=r.setdefault('noisy',{});datasets=r.setdefault('datasets',{})
    requests=[(truth,budget) for truth in ['matched_A','continuum_A'] for budget in ['8','4']]+[('continuum_C','8')]
    for truth,budget in requests:
        ctx=context(r,truth,budget);key=f'{truth}/{budget}'
        seeds=list(range(720,740)) if truth=='continuum_C' else list(range(700,720))
        models=['M0','M1','M5'] if truth=='continuum_C' else ['M0','M1','M4','M5'] if truth=='matched_A' else MODELS
        if key not in datasets:
            entries=[]
            for seed in seeds:
                data,meta=add_measurement_noise(ctx['clean'],ctx['ti'],.02 if budget=='8' else .05,seed)
                entries.append(dict(seed=seed,observations=plain(data),hash=frozen_hash(plain(data)),noise=meta))
            datasets[key]=entries;save(r)
        data=np.array([a['observations'] for a in datasets[key]])
        for model in models:
            groupkey=f'{key}/{model}'
            if groupkey in groups:
                if len(groups[groupkey]['runs'])!=20:raise RuntimeError('Incomplete checkpoint '+groupkey)
                continue
            if groupkey not in r['clean']:raise RuntimeError('Missing clean reference '+groupkey)
            print('Noisy',groupkey,flush=True);rows=fit(r,ctx,model,data=data)
            cp=r['clean'][groupkey]['final']['parameters']
            for row,entry in zip(rows,datasets[key]):
                row.update(seed=entry['seed'],observation_hash=entry['hash'])
                row['decomposition']=bias_decomposition(row['final']['parameters'][:2],cp[:2],ctx['truth_parameters'])
                row['discrepancy_noise_shift']=(np.array(row['final']['parameters'][2:])-cp[2:]).tolist()
            groups[groupkey]={'runs':rows,'summary':plain(summarize(rows))};save(r)
            print('  maximum KKT',groups[groupkey]['summary']['max_kkt'],'continuations',sum(a['continuation_required'] for a in rows),flush=True)
    if 'noisy-inference' not in r['completed']:r['completed'].append('noisy-inference')
    save(r)


def main():
    parser=argparse.ArgumentParser()
    for phase in ['forward-validation','calibrate','clean-inference','noisy-inference','finalize']:parser.add_argument('--'+phase,action='store_true')
    args=parser.parse_args();r=load()
    funcs=[('forward_validation',forward),('calibrate',calibrate),('clean_inference',clean),('noisy_inference',noisy)]
    for name,fn in funcs:
        if getattr(args,name):
            try:fn(r)
            except Exception as exc:r['failures'][name]=repr(exc);save(r);raise
            else:r['failures'].pop(name,None);save(r)
    if args.finalize:
        from experiments.stage9_report import finalize
        finalize(r);save(r)

if __name__=='__main__':main()
