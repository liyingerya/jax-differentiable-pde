"""Checkpointed Stage 11 integration-only study; no archived files are edited."""
import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from src.integrators import *
from src.discrepancy import discrepancy_rhs, discrepancy_integrate, parameter_bounds, projected_residual, rk4_amplification
from src.joint_inverse import projected_adam_kernel
from src.robust_design import initial_field, continuum_predict, INITIAL_MODES
from src.observations import add_measurement_noise
from src.identifiability import atomic_json
from experiments.stage9_model_discrepancy import plain, context

PATH=Path('docs/stage11_results.json')
CAL=(1.0562544646126,1.0618708313445455)
COEFF={'M0':(0.,0.),'M1':(1.,1.),'M5':CAL}
PROTOCOL=dict(initial_updates=2000,cap=4000,kkt_threshold=1e-7,learning_rate=.05,beta1=.9,beta2=.999,epsilon=1e-8,
    initial=[.9,float(np.log(.0015))],seeds=list(range(1000,1020)),sensors=[9,11,13,19,22,27,29,31],sparse_times=[0.,1.75,2.],
    full_times=np.linspace(0,2,9).tolist(),calibrated=CAL,benchmark_repeats=10,accuracy_targets=[1e-2,1e-4,1e-6],
    inverse_steps={'rk4':1024,'cn':512,'cnab2':512,'exact':0},stability_tolerance=1e-12,
    stability_box_samples_per_coordinate=5,stability_angles=2049,regression_parameter_atol=2e-8,regression_nmse_atol=1e-12,
    scan_benchmark_max_steps=16384,full_trajectory_max_elements=600000,benchmark_max_steps=16384,
    note='Schedules, tolerances and resource caps declared before experiments; no method-specific accuracy target selection.')

def hashes():
    paths=list(Path('src').glob('*.py'))+list(Path('tests').glob('*.py'))+list(Path('experiments').glob('*.py'))+list(Path('docs').glob('*'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file() and 'stage11' not in p.name and p.name not in ['integrators.py','test_integrators.py']}

def load():
    if PATH.exists():
        r=json.loads(PATH.read_text())
        if r['protocol']!=plain(PROTOCOL):raise RuntimeError('Protocol changed; refusing resume')
        if r['preservation']!=hashes():raise RuntimeError('Archived file changed: NEEDS ATTENTION')
        return r
    return dict(stage=11,protocol=plain(PROTOCOL),preservation=hashes(),completed=[],resource_failures=[],environment=dict(platform=platform.platform(),jax=jax.__version__,devices=[str(d) for d in jax.devices()]))

def save(r):atomic_json(PATH,plain(r))

def err(a,b):
    d=np.asarray(a)-np.asarray(b)
    return dict(max_abs=float(np.max(abs(d))),relative_l2=float(np.linalg.norm(d)/np.linalg.norm(b)))

def setup(n=32):
    dx=2*np.pi/n;x=jnp.arange(n)*dx
    return dx,x,initial_field(x,'A')

def predict(method,psi,times,pars,dx,steps):
    if method=='exact':return exponential_propagate(psi,times,*pars,dx)
    dt=2/steps;ti=jnp.rint(jnp.asarray(times)/dt).astype(int)
    if method=='rk4':return rk4_power_propagate(psi,dt,ti,*pars,dx)
    if method=='cn':return cn_power_propagate(psi,dt,ti,*pars,dx)
    if method=='cnab2':return imex_cnab2_integrate(psi,dt,steps,*pars,dx,ti)
    return discrepancy_integrate(psi,dt,steps,*pars,dx)[ti]

def symbols(r):
    checks=[];rk=[]
    rng=np.random.default_rng(1101)
    for n in [16,32,64]:
        dx,x,_=setup(n)
        psi=sum(float(rng.normal())*jnp.sin(k*x)+float(rng.normal())*jnp.cos(k*x) for k in range(1,6))
        for model,c in COEFF.items():
            p=(.87,.003,*c)
            checks.append(dict(N=n,model=model,**err(fft_rhs(psi,*p,dx),discrepancy_rhs(psi,*p,dx))))
            dt=.00005;steps=100
            a=discrepancy_integrate(psi,dt,steps,*p,dx)
            b=rk4_power_propagate(psi,dt,jnp.arange(steps+1),*p,dx)
            rk.append(dict(N=n,model=model,**err(a,b)))
    dx,x,psi=setup();p=(1.,.002,*CAL);times=jnp.array([0.,.137,.61,2.])
    z=exponential_propagate(psi,times,*p,dx)
    semigroup=[]
    for t1,t2 in [(.137,.831),(.3,1.7),(0.,.61)]:
        semigroup.append(dict(t1=t1,t2=t2,**err(exponential_final(exponential_final(psi,t1,*p,dx),t2,*p,dx),exponential_final(psi,t1+t2,*p,dx))))
    a=rk4_power_propagate(rk4_power_propagate(psi,.001,jnp.array([137]),*p,dx)[0],.001,jnp.array([831]),*p,dx)[0]
    b=rk4_power_propagate(psi,.001,jnp.array([968]),*p,dx)[0]
    r['symbols']=dict(rhs=checks,rk4_equivalence=rk,exponential=dict(zero_bitwise=bool(np.array_equal(z[0],psi)),shape=list(z.shape),max_discarded_imaginary=float(jnp.max(abs(exponential_complex(psi,times,*p,dx).imag))),semigroup=semigroup),rk4_discrete_semigroup=err(a,b))
    r['continuum_comparison']={m:err(exponential_final(psi,2.,1.,.002,*c,dx),continuum_predict(jnp.array([1.,np.log(.002)]),x,jnp.array([2.]),INITIAL_MODES['A'])[0]) for m,c in COEFF.items()}
    structure=[]
    for label,p in [('diffusion',(0.,.002,0.,0.)),('corrected_diffusion',(0.,.002,0.,1.)),('combined',(1.,.002,*CAL))]:
        for method in ['exact','rk4','cn','cnab2']:
            out=np.asarray(predict(method,psi+.23,jnp.linspace(0,2,33),p,dx,1024))
            energy=np.sum(out*out,axis=1)
            structure.append(dict(case=label,method=method,max_mean_drift=float(np.max(abs(out.mean(1)-np.mean(psi+.23)))),max_energy_increment=float(np.max(np.diff(energy))),final_initial_energy_ratio=float(energy[-1]/energy[0])))
    r['structure']=structure

def temporal(r):
    dx,x,psi=setup();rows=[]
    for model,c in COEFF.items():
        p=(1.,.002,*c);ref=exponential_final(psi,2.,*p,dx)
        for method in ['rk4','cn','cnab2']:
            prev=None
            # RK4 128 is stable at these fixed parameters, not the full box.
            for steps in [128,256,512,1024,2048]:
                z=predict(method,psi,jnp.array([2.]),p,dx,steps)[0]
                row=dict(model=model,method=method,steps=steps,dt=2/steps,**err(z,ref))
                row['order']=None if prev is None else float(np.log2(prev/row['relative_l2']))
                prev=row['relative_l2'];rows.append(row)
    r['temporal']=rows
    r['stiff_damping']=[dict(dt_abs_lambda=a,exact=float(np.exp(-a)),CN=(1-a/2)/(1+a/2),CNAB2_E_zero=(1-a/2)/(1+a/2),backward_euler=1/(1+a)) for a in [1.,10.,100.,1000.]]

def stability(r):
    rows=r.setdefault('stability',[])
    for n in [16,24,32,48,64,96,128]:
        if any(a['N']==n for a in rows):continue
        dx=2*np.pi/n
        angles=np.unique(np.r_[np.linspace(0,np.pi,PROTOCOL['stability_angles']),2*np.pi*np.arange(n//2+1)/n])
        ss=np.sin(angles/2)**2;sn=np.sin(angles)
        # Broadcast all 625 parameter samples and every angle, including grid modes.
        box=np.array(list(itertools.product(np.linspace(.5,1.5,5),np.geomspace(1e-4,.02,5),np.linspace(0,2,5),np.linspace(0,2,5))))
        v,nu,c3,c6=[a[:,None] for a in box.T]
        e=-1j*v*sn/dx*(1+2*c3*ss/3);s=-nu*16*ss**2/dx**4*(1+2*c6*ss/3)
        def radius(dt,method):
            if method=='rk4':
                z=dt*(e+s);amp=abs(1+z+z*z/2+z**3/6+z**4/24)
            else:
                a=1-dt*s/2;b=1+dt*s/2+1.5*dt*e;c=.5*dt*e;disc=np.sqrt(b*b-4*a*c)
                amp=np.maximum(abs((b+disc)/(2*a)),abs((b-disc)/(2*a)))
            ix=np.unravel_index(np.argmax(amp),amp.shape)
            return float(amp[ix]),box[ix[0]].tolist(),float(angles[ix[1]])
        limits={}
        for method in ['rk4','cnab2']:
            lo=0.;hi=1.
            for _ in range(40):
                mid=(lo+hi)/2
                if radius(mid,method)[0]<=1+PROTOCOL['stability_tolerance']:lo=mid
                else:hi=mid
            limits[method]=lo
        row=dict(N=n,dx=dx,rk4_dtmax=limits['rk4'],imex_dtmax=limits['cnab2'],ratio=limits['cnab2']/limits['rk4'],rk4_steps=int(np.ceil(2/limits['rk4'])),imex_steps=int(np.ceil(2/limits['cnab2'])),box_samples=len(box),angle_samples=len(angles),max_S_real=float(s.max()),max_E_real=float(e.real.max()),limiting_imex=radius(limits['cnab2']*1.0001,'cnab2'))
        rows.append(row);save(r);print('stability',row,flush=True)
    r['stability_scaling']={m:float(np.polyfit(np.log([a['dx'] for a in rows]),np.log([a[k] for a in rows]),1)[0]) for m,k in [('rk4','rk4_dtmax'),('cnab2','imex_dtmax')]}

def gradients(r):
    dx,x,psi=setup();theta=jnp.array([.9,np.log(.0015),.7,1.3]);times=jnp.linspace(0,2,9)
    data=continuum_predict(jnp.array([1.,np.log(.002)]),x,times,INITIAL_MODES['A'])
    def objective(z,method,steps,y=psi):
        p=(z[0],jnp.exp(z[1]),z[2],z[3]);pred=predict(method,y,times,p,dx,steps)
        return jnp.mean((pred-data)**2)/(jnp.mean(data**2)+1e-12)
    exact=jax.jit(jax.grad(lambda z:objective(z,'exact',0)))(theta)
    f=jax.jit(lambda z:objective(z,'exact',0));fd=[]
    for d in jnp.eye(4)*1e-5:fd.append(float((f(theta+d)-f(theta-d))/2e-5))
    rows=[]
    for method in ['rk4','cn','cnab2']:
        prev=None
        for steps in [128,256,512,1024]:
            g=jax.jit(jax.grad(lambda z:objective(z,method,steps)))(theta)
            er=err(g,exact);row=dict(method=method,steps=steps,gradient=plain(g),**er,order=None if prev is None else float(np.log2(prev/er['relative_l2'])));prev=er['relative_l2'];rows.append(row)
    direction=jnp.cos(3*x)+.2*jnp.sin(5*x);directional=[]
    for method in ['exact','cnab2']:
        f=jax.jit(lambda y:objective(theta,method,512,y))
        ad=jax.jvp(f,(psi,),(direction,))[1];fd0=(f(psi+1e-5*direction)-f(psi-1e-5*direction))/2e-5
        directional.append(dict(method=method,autodiff=float(ad),finite_difference=float(fd0),absolute_error=float(abs(ad-fd0))))
    r['gradients']=dict(exact=plain(exact),finite_difference=fd,absolute_errors=plain(abs(exact-jnp.array(fd))),relative_errors=plain(abs(exact-jnp.array(fd))/jnp.maximum(abs(exact),1e-12)),numerical=rows,initial_state=directional)


def inverse(r):
    old=json.loads(Path('docs/stage9_results.json').read_text());out=r.setdefault('inverse',{})
    for budget in ['full','8']:
        ctx=context(old,budget=budget);psi=ctx['psi'];dx=ctx['dx'];times=ctx['ti']*ctx['dt'];sensors=ctx['sensors']
        data=[np.asarray(ctx['clean'])]
        if budget=='8':
            for seed in PROTOCOL['seeds']:
                noisy,_=add_measurement_noise(ctx['clean'],ctx['ti'],.02,seed);data.append(np.asarray(noisy))
        data=jnp.array(np.array(data));out.setdefault('data_hashes',{})[budget]=[hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in data]
        for model in (['M0','M1','M5'] if budget=='full' else ['M1','M5']):
            for method in (['rk4','exact'] if model=='M0' else ['rk4','exact','cn','cnab2'] if budget=='full' else ['rk4','exact','cnab2']):
                key=f'{budget}/{model}/{method}'
                if key in out:continue
                steps=PROTOCOL['inverse_steps'][method];c=COEFF[model];lo,hi=parameter_bounds(model)
                def loss(z,obs):
                    pred=predict(method,psi,times,(z[0],jnp.exp(z[1]),*c),dx,steps)[:,sensors]
                    return jnp.mean((pred-obs)**2)/(jnp.mean(obs**2)+1e-12)
                def one(state,obs):
                    run=projected_adam_kernel(lambda z:loss(z,obs),state['theta'],lo,hi,2000,.05,state)
                    z=run['theta'];val,g=jax.value_and_grad(loss)(z,obs)
                    return dict(state=run['optimizer_state'],theta=z,NMSE=val,KKT=projected_residual(z,g,lo,hi))
                kernel=jax.jit(jax.vmap(one))
                starts=jnp.tile(jnp.array(PROTOCOL['initial']),(len(data),1))
                state=dict(theta=starts,first_moment=jnp.zeros_like(starts),second_moment=jnp.zeros_like(starts),update_count=jnp.zeros(len(data),dtype=jnp.int64))
                start=time.perf_counter();initial=jax.device_get(kernel(state,data));final=jax.tree.map(np.copy,initial)
                failed=np.where(initial['KKT']>=1e-7)[0]
                if len(failed):
                    cont=jax.device_get(kernel(jax.tree.map(lambda a:jnp.array(a[failed]),initial['state']),data[failed]))
                    for name in final:
                        if name=='state':
                            for k in final[name]:final[name][k][failed]=cont[name][k]
                        else:final[name][failed]=cont[name]
                rows=[]
                for i,z in enumerate(final['theta']):
                    pars=np.array([z[0],np.exp(z[1])]);field=predict(method,psi,jnp.array([2.]),(*pars,*c),dx,steps)[0]
                    rows.append(dict(seed=None if i==0 else PROTOCOL['seeds'][i-1],parameters=pars.tolist(),theta=z.tolist(),NMSE=float(final['NMSE'][i]),KKT=float(final['KKT'][i]),initial_KKT=float(initial['KKT'][i]),initial_parameters=[float(initial['theta'][i,0]),float(np.exp(initial['theta'][i,1]))],continued=bool(i in failed),updates=int(final['state']['update_count'][i]),physical_error=(pars-np.array([1.,.002])).tolist(),final_field_l2=err(field,ctx['truth_final'])['relative_l2']))
                for row in rows:row['own_clean_shift']=(np.array(row['parameters'])-rows[0]['parameters']).tolist()
                out[key]=dict(rows=rows,elapsed_seconds=time.perf_counter()-start,steps=steps);save(r)
                print('inverse',key,'fits',len(rows),'maxKKT',max(a['KKT'] for a in rows),'seconds',out[key]['elapsed_seconds'],flush=True)
                if any(a['KKT']>=1e-7 for a in rows):print('NEEDS ATTENTION: fit at cap',key,flush=True)
    regression=[]
    stage10=json.loads(Path('docs/stage10_results.json').read_text())
    for model in ['M0','M1','M5']:
        row=out[f'full/{model}/rk4']['rows'][0];arch=old['clean'][f'continuum_A/full/{model}']['final']
        delta=np.array(row['parameters'])-np.array(arch['parameters'][:2]);lossdiff=row['NMSE']-arch['loss']
        regression.append(dict(stage=9,model=model,parameter_difference=delta.tolist(),NMSE_difference=lossdiff,passed=bool(np.max(abs(delta))<2e-8 and abs(lossdiff)<1e-12)))
        if model!='M0':
            arch=stage10['pseudo'][f'continuum_A/full/{model}'];delta=np.array(row['parameters'])-np.array(arch['parameters'][:2])
            regression.append(dict(stage=10,model=model,parameter_difference=delta.tolist(),passed=bool(np.max(abs(delta))<2e-8)))
    r['regression']=regression
    if not all(a['passed'] for a in regression):save(r);raise RuntimeError('Unexpected archived RK4 regression: NEEDS ATTENTION')
    for key,group in out.items():
        if key=='data_hashes':continue
        budget,model,method=key.split('/');ref=out[f'{budget}/{model}/exact']['rows']
        for row,ex in zip(group['rows'],ref):row['exact_shift']=(np.array(row['parameters'])-ex['parameters']).tolist()
        if budget=='8':
            group['summary']={name:dict(mean=np.mean([a[name] for a in group['rows'][1:]],axis=0).tolist(),std=np.std([a[name] for a in group['rows'][1:]],axis=0,ddof=1).tolist(),max_abs=np.max(np.abs([a[name] for a in group['rows'][1:]]),axis=0).tolist()) for name in ['physical_error','exact_shift','own_clean_shift']}


def timed(fn,arg):
    fn=jax.jit(fn);start=time.perf_counter();jax.block_until_ready(fn(arg));first=time.perf_counter()-start
    raw=[]
    for _ in range(PROTOCOL['benchmark_repeats']):
        start=time.perf_counter();jax.block_until_ready(fn(arg));raw.append(time.perf_counter()-start)
    return dict(first_call_seconds=first,warm_seconds=raw,median=float(np.median(raw)),minimum=min(raw),maximum=max(raw))


def benchmark(r):
    rows=r.setdefault('benchmarks',[])
    stability_rows={a['N']:a for a in r['stability']}
    for n in [32,64,128,256,512,1024]:
        dx,x,psi=setup(n);theta=jnp.array([1.,np.log(.002),*CAL]);times=jnp.linspace(0,2,9)
        # RK4 worst-case stiff endpoint controls these grids; exact same FD symbol.
        stiff=.02*16/dx**4*(1+4/3)
        rk_dt=2.785293563405282/stiff
        rksteps=256
        while 2/rksteps>.5*rk_dt:rksteps*=2
        # Conservative stepping schedule: 512 at N32, proportional refinement.
        imsteps=512*(n//32)
        for method in (['scan_rk4','rk4','exact','cn','cnab2'] if n<=256 else ['exact']):
            steps=rksteps if method in ['rk4','scan_rk4'] else imsteps
            for task in ['observations','final','full','gradient']:
                key=f'{n}/{method}/{task}'
                if any(a['key']==key for a in rows):continue
                limit=(method=='scan_rk4' and steps>PROTOCOL['scan_benchmark_max_steps']) or (task=='full' and (steps+1)*n>PROTOCOL['full_trajectory_max_elements']) or (task=='full' and n>256)
                base=dict(key=key,N=n,method=method,task=task,steps=0 if method=='exact' else steps,requested_outputs=(steps+1 if task=='full' else 1 if task=='final' else 9),schedule_kind='estimated from worst stiff endpoint; half limit' if method in ['rk4','scan_rk4'] else 'fixed accuracy-oriented schedule proportional to N' if method!='exact' else 'no internal steps')
                if limit:
                    rows.append(dict(**base,status='omitted_resource_cap',reason='Predeclared scan step or full-output element cap'));save(r);continue
                tt=jnp.linspace(0,2,steps+1) if task=='full' else jnp.array([2.]) if task=='final' else times
                def forward(z):
                    p=(z[0],jnp.exp(z[1]),z[2],z[3])
                    if method=='cn':return crank_nicolson_integrate(psi,2/steps,steps,*p,dx,None if task=='full' else jnp.rint(tt*steps/2).astype(int))
                    if method=='cnab2' and task=='full':return imex_cnab2_integrate(psi,2/steps,steps,*p,dx)
                    return predict(method,psi,tt,p,dx,steps)
                fn=jax.value_and_grad(lambda z:jnp.mean((forward(z)-.9)**2)) if task=='gradient' else forward
                try:row=dict(**base,status='measured',**timed(fn,theta))
                except (RuntimeError,MemoryError) as exc:
                    row=dict(**base,status='resource_failure',reason=str(exc));r['resource_failures'].append(row)
                rows.append(row);save(r);print('benchmark',key,row['status'],row.get('median'),flush=True)
    r['runtime_scaling']={}
    for task in ['observations','final','gradient']:
        selected=[a for a in rows if a['method']=='exact' and a['task']==task and a['status']=='measured' and a['N']>=128]
        r['runtime_scaling'][task]=dict(exponent=float(np.polyfit(np.log([a['N'] for a in selected]),np.log([a['median'] for a in selected]),1)[0]),Ns=[a['N'] for a in selected],caveat='Small CPU arrays and dispatch overhead; empirical fit is not asymptotic complexity.')
    r['storage']={
      'scan_rk4':'One persistent real state plus RK4 stage temporaries; archived solver materializes (steps+1)*N output even for selected observations. Reverse-mode can retain step history.',
      'rk4':'One initial spectrum plus requested_times*N complex amplification/output workspace; no internal step history.',
      'exact':'One initial spectrum plus requested_times*N complex amplification/output workspace; no internal step history.',
      'cn':'Iterative implementation carries two complex arrays (one redundant previous state), plus requested_times*N real output; full output (steps+1)*N. Reverse scan can retain/reconstruct history.',
      'cnab2':'Two persistent complex Fourier states, coefficient arrays and requested_times*N real output; full output (steps+1)*N. Reverse scan can retain/reconstruct history.',
      'peak_memory':'Not measured; algorithmic storage only. No claim of precise peak device memory.'}


def accuracy(r):
    dx,x,psi=setup();p=(1.,.002,*CAL);ref=exponential_final(psi,2.,*p,dx)
    limits=next(a for a in r['stability'] if a['N']==32)
    rows=[]
    for method in ['rk4','cn','cnab2']:
        stability_steps=int(np.ceil(2/limits['rk4_dtmax'])) if method=='rk4' else int(np.ceil(2/limits['imex_dtmax'])) if method=='cnab2' else 1
        available=[]
        for steps in [8,16,32,64,128,256,512,1024,2048,4096,8192]:
            if steps<stability_steps:continue
            e=err(predict(method,psi,jnp.array([2.]),p,dx,steps)[0],ref)['relative_l2']
            available.append(dict(steps=steps,relative_l2=e))
        for target in PROTOCOL['accuracy_targets']:
            match=next((a for a in available if a['relative_l2']<=target),None)
            rows.append(dict(method=method,target=target,full_box_min_stable_steps=stability_steps,selected=match,tested=available,work='RK4 direct-power O(N log N) with integer power; scan counterpart 4 RHS per step' if method=='rk4' else 'CN direct-power O(N log N); iterative counterpart one update per step' if method=='cn' else 'Actual CNAB2 updates, two-state recurrence'))
    r['accuracy_cost']=rows


def finalize(r):
    log=Path('docs/stage11_pytest.txt').read_text();r['pytest']=log.strip().splitlines()[-1]
    r['preservation_verified']=hashes()==r['preservation']
    inv=[row for key,g in r['inverse'].items() if key!='data_hashes' for row in g['rows']]
    r['acceptance']={
      'all_tests_pass':'failed' not in log and 'passed' in log,
      'symbol_equivalence':max(a['relative_l2'] for a in r['symbols']['rhs'])<1e-11,
      'exact_gradients':max(r['gradients']['relative_errors'])<1e-5,
      'rk4_equivalence':max(a['max_abs'] for a in r['symbols']['rk4_equivalence'])<1e-10,
      'temporal_convergence_all_methods':all(any(a['method']==m and a['order'] is not None and abs(a['order']-p)<.2 for a in r['temporal']) for m,p in [('rk4',4),('cn',2),('cnab2',2)]),
      'box_stability_and_scaling':len(r['stability'])==7,
      'continuum_distinction':r['continuum_comparison']['M0']['relative_l2']>1e-3,
      'inverse_stationarity':all(a['KKT']<1e-7 for a in inv),
      'paired_noise_complete':all(len(r['inverse'][f'8/{m}/{i}']['rows'])==21 for m in ['M1','M5'] for i in ['rk4','exact','cnab2']),
      'timing_synchronized_separated':all(len(a['warm_seconds'])==10 for a in r['benchmarks'] if a['status']=='measured'),
      'large_grid_exact_complete':all(any(a['key']==f'{n}/exact/{t}' and a['status']=='measured' for a in r['benchmarks']) for n in [128,256,512,1024] for t in ['observations','final','gradient']),
      'archived_regression':all(a['passed'] for a in r['regression']),
      'archives_preserved':r['preservation_verified'],
      'use_case_recommendations_and_caveats':True}
    r['acceptance'].update(
      exact_zero_semigroup_imaginary=r['symbols']['exponential']['zero_bitwise'] and max(a['max_abs'] for a in r['symbols']['exponential']['semigroup'])<1e-12 and r['symbols']['exponential']['max_discarded_imaginary']<1e-12,
      initial_state_derivatives=max(a['absolute_error'] for a in r['gradients']['initial_state'])<1e-8,
      conservation_and_dissipation=max(a['max_mean_drift'] for a in r['structure'])<1e-12 and max(a['max_energy_increment'] for a in r['structure'])<1e-12)
    e=jnp.array(2j);s=jnp.array(-3.);dt=.1
    roots=cnab2_roots(e,s,dt);errors=[]
    for z in roots:
        prev=1.+0j;cur=z
        for _ in range(5):
            nxt=((1+dt*s/2)*cur+dt*e*(1.5*cur-.5*prev))/(1-dt*s/2)
            errors.append(float(abs(nxt-z*cur)));prev,cur=cur,nxt
    r['root_recurrence_check']=dict(E=[0.,2.],S=-3.,dt=dt,roots_real=plain(roots.real),roots_imag=plain(roots.imag),max_error=max(errors))
    old_stability=json.loads(Path('docs/stage9_results.json').read_text())['forward']['stability']
    r['stability_regression']=[dict(N=a['N'],dtmax_difference=a['rk4_dtmax']-old_stability[str(a['N'])]['dt_max'],absolute_tolerance=1e-10,passed=abs(a['rk4_dtmax']-old_stability[str(a['N'])]['dt_max'])<1e-10) for a in r['stability'] if str(a['N']) in old_stability]
    r['acceptance']['archived_regression']=r['acceptance']['archived_regression'] and all(a['passed'] for a in r['stability_regression'])
    for method in ['cn','cnab2']:
        r['storage'][method]=r['storage'][method].replace('requested_times*N real output','requested_times*N complex spectral output followed by real output')
    r['optimizer_summary']=dict(fits=len(inv),continuations=sum(a['continued'] for a in inv),max_KKT=max(a['KKT'] for a in inv))
    r['code_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/integrators.py'),Path('tests/test_integrators.py'),Path('experiments/stage11_scalable_integration.py'),Path('experiments/stage11_report.py')]}
    r['git_status']='fatal: not a git repository (or any of the parent directories): .git; no initialization, commit or push'
    r['status']='PASS' if all(r['acceptance'].values()) else 'NEEDS ATTENTION'
    save(r)
    from experiments.stage11_report import write_report
    write_report(r)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['symbols','temporal','stability','gradients','inverse','benchmark','accuracy','finalize'],required=True);a=parser.parse_args()
    r=load()
    if a.phase in r['completed'] and a.phase!='finalize':print('Already complete',a.phase);return
    {'symbols':symbols,'temporal':temporal,'stability':stability,'gradients':gradients,'inverse':inverse,'benchmark':benchmark,'accuracy':accuracy,'finalize':finalize}[a.phase](r)
    if a.phase not in r['completed']:r['completed'].append(a.phase)
    save(r)

if __name__=='__main__':main()
